"""Camera-ready add-on: train mBERT + XLM-R on the SAME deterministic split and
return PER-EXAMPLE calibration/test probabilities, so the concept-smoothed method,
the full Mondrian suite, and PAIRED significance tests can be run locally on the
transformer backbones (not just aggregate counts). Run via run_cr_beam_probs.py."""
from beam import function, Image

image = (Image(python_version="python3.11")
         .add_python_packages(["torch", "transformers", "scikit-learn", "pandas",
                               "pyarrow", "numpy", "huggingface_hub", "sentencepiece"]))


@function(gpu="RTX4090", cpu=8, memory="24Gi", image=image, timeout=3000)
def run():
    import sys, time, json, gc, copy, urllib.request
    from pathlib import Path
    import numpy as np, torch
    from sklearn.metrics import f1_score, accuracy_score
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    root = Path(__file__).resolve().parent; sys.path.insert(0, str(root))
    DATA = root / "data"; DATA.mkdir(exist_ok=True)
    urls = {"train.parquet": "https://huggingface.co/datasets/community-datasets/disaster_response_messages/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
            "validation.parquet": "https://huggingface.co/datasets/community-datasets/disaster_response_messages/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet",
            "test.parquet": "https://huggingface.co/datasets/community-datasets/disaster_response_messages/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet"}
    for fn, u in urls.items():
        p = DATA / fn
        if not p.exists() or p.stat().st_size < 100000:
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}); p.write_bytes(urllib.request.urlopen(req, timeout=180).read())
    from src.data import build_unified_dataframe, stratified_split

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    K, MAXLEN, EPOCHS, BS = 5, 96, 3, 32
    df = build_unified_dataframe(); tr, cal, te = stratified_split(df)
    y_tr, y_cal, y_te = tr["y_crisis_type"].values, cal["y_crisis_type"].values, te["y_crisis_type"].values
    lang_cal, lang_te = cal["lang"].values.astype(str), te["lang"].values.astype(str)

    def enc(tok, texts): return tok(list(texts), truncation=True, padding="max_length", max_length=MAXLEN, return_tensors="pt")
    R = {"device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "epochs": EPOCHS,
         "y_cal": y_cal.tolist(), "y_te": y_te.tolist(), "lang_cal": lang_cal.tolist(), "lang_te": lang_te.tolist(),
         "models": {}}

    for MODEL in ["bert-base-multilingual-cased", "xlm-roberta-base"]:
        torch.manual_seed(42); np.random.seed(42)
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=K).to(dev)
        etr = enc(tok, tr["text"].tolist()); yt = torch.tensor(y_tr, dtype=torch.long)
        yval = y_cal[:1200]; eva = enc(tok, cal["text"].tolist()[:1200])
        opt = torch.optim.AdamW(model.parameters(), lr=3e-5); lossf = torch.nn.CrossEntropyLoss()
        n = len(tr); best = None; bestf1 = -1; t0 = time.time()
        for ep in range(EPOCHS):
            model.train(); perm = torch.randperm(n)
            for s in range(0, n, BS):
                idx = perm[s:s + BS]; opt.zero_grad()
                out = model(input_ids=etr["input_ids"][idx].to(dev), attention_mask=etr["attention_mask"][idx].to(dev))
                lossf(out.logits, yt[idx].to(dev)).backward(); opt.step()
            model.eval(); ps = []
            with torch.no_grad():
                for s in range(0, len(yval), 128):
                    lo = model(input_ids=eva["input_ids"][s:s + 128].to(dev), attention_mask=eva["attention_mask"][s:s + 128].to(dev)).logits
                    ps.append(lo.argmax(1).cpu().numpy())
            vf1 = f1_score(yval, np.concatenate(ps), average="macro", labels=list(range(K)), zero_division=0)
            if vf1 > bestf1: bestf1 = vf1; best = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        train_s = time.time() - t0
        if best: model.load_state_dict(best); model.to(dev)

        def probs(texts):
            model.eval(); out = []
            with torch.no_grad():
                for s in range(0, len(texts), 128):
                    e = enc(tok, texts[s:s + 128]); lo = model(input_ids=e["input_ids"].to(dev), attention_mask=e["attention_mask"].to(dev)).logits
                    out.append(torch.softmax(lo, 1).cpu().numpy())
            return np.concatenate(out)
        P_cal, P_te = probs(cal["text"].tolist()), probs(te["text"].tolist()); yp = P_te.argmax(1)
        plf1 = {lg: float(f1_score(y_te[lang_te == lg], yp[lang_te == lg], average="macro", labels=list(range(K)), zero_division=0))
                for lg in sorted(set(lang_te)) if (lang_te == lg).sum() >= 10}
        cpu = model.to("cpu"); cpu.eval(); e1 = enc(tok, te["text"].tolist()[:1])
        with torch.no_grad():
            for _ in range(5): cpu(input_ids=e1["input_ids"], attention_mask=e1["attention_mask"])
            xs = []
            for _ in range(30):
                t = time.perf_counter(); cpu(input_ids=e1["input_ids"], attention_mask=e1["attention_mask"]); xs.append((time.perf_counter() - t) * 1000)
        R["models"][MODEL] = {"test_acc": float(accuracy_score(y_te, yp)), "test_macro_f1": float(f1_score(y_te, yp, average="macro")),
                              "per_lang_f1": plf1, "n_params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
                              "train_seconds": round(train_s, 1), "cpu_latency_ms_p50": round(float(np.percentile(xs, 50)), 2),
                              "P_cal": [[round(v, 6) for v in row] for row in P_cal.tolist()],
                              "P_te": [[round(v, 6) for v in row] for row in P_te.tolist()]}
        del model, etr; gc.collect(); torch.cuda.empty_cache() if dev == "cuda" else None
    print("PROBS_DONE")
    return R
