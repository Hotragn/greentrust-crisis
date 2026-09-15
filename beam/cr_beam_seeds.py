"""Optional work 1: multi-seed transformer training to firm up the accuracy
comparison (the one run-dependent number). Trains mBERT + XLM-R under 4 seeds each
on the identical deterministic split, returning per-seed macro-F1, per-example test
predictions (for paired bootstrap vs the student), and Mondrian infra counts.
Returns compact summaries (labels not probs) to keep the payload small.
Run via run_cr_beam_seeds.py."""
from beam import function, Image

image = (Image(python_version="python3.11")
         .add_python_packages(["torch", "transformers", "scikit-learn", "pandas",
                               "pyarrow", "numpy", "huggingface_hub", "sentencepiece"]))

SEEDS = [0, 1, 2, 3]


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
    from src.config import DEFAULT_CONFIG as CFG
    from src.data import build_unified_dataframe, stratified_split
    from src.features import build_teacher_vectorizer, build_student_vectorizer, select_features_by_mi
    from src.models import TeacherModel, StudentModel

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    K, MAXLEN, EPOCHS, BS, A, INF = 5, 96, 3, 32, 0.10, 3
    df = build_unified_dataframe(); tr, cal, te = stratified_split(df)
    y_tr, y_cal, y_te = tr["y_crisis_type"].values, cal["y_crisis_type"].values, te["y_crisis_type"].values

    def qlevel(n): return min(np.ceil((1 - A) * (n + 1)) / n, 1.0) if n > 0 else 1.0
    def qhat(s):
        s = np.asarray(s); return float(np.quantile(s, qlevel(len(s)), method="higher")) if len(s) else 1.0
    def conformal_counts(P_cal, P_te):
        tm = np.full(K, 1 - qhat(1 - P_cal[np.arange(len(y_cal)), y_cal]))
        tc = np.ones(K)
        for k in range(K):
            m = y_cal == k; tc[k] = 1 - qhat(1 - P_cal[m, k]) if m.any() else 1.0
        out = {}
        for nm, thr in [("marginal", tm), ("class_conditional", tc)]:
            S = [np.where(P_te[i] >= thr)[0] for i in range(len(P_te))]
            idx = np.where(y_te == INF)[0]
            out[nm] = {"infra_covered": int(sum(INF in S[i] for i in idx)), "infra_n": int(len(idx)),
                       "overall_cov": round(float(np.mean([y_te[i] in S[i] for i in range(len(y_te))])), 4),
                       "avg_set": round(float(np.mean([len(s) for s in S])), 3)}
        return out

    def enc(tok, texts): return tok(list(texts), truncation=True, padding="max_length", max_length=MAXLEN, return_tensors="pt")
    R = {"device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
         "epochs": EPOCHS, "seeds": SEEDS, "y_te": y_te.tolist(), "models": {}}

    for MODEL in ["bert-base-multilingual-cased", "xlm-roberta-base"]:
        R["models"][MODEL] = {}
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
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
            R["models"][MODEL][str(seed)] = {
                "test_macro_f1": float(f1_score(y_te, yp, average="macro")),
                "test_acc": float(accuracy_score(y_te, yp)),
                "train_seconds": round(train_s, 1),
                "yp_test": yp.astype(int).tolist(),
                "conformal": conformal_counts(P_cal, P_te)}
            print(f"{MODEL} seed={seed} f1={R['models'][MODEL][str(seed)]['test_macro_f1']:.4f}")
            del model, etr; gc.collect()
            if dev == "cuda": torch.cuda.empty_cache()

    # linear student on the same host/split (for paired comparison)
    tv = build_teacher_vectorizer(CFG.teacher_max_features, CFG.teacher_ngram_range, CFG.teacher_min_df)
    Xtr_t = tv.fit_transform(tr["text"].tolist()); teacher = TeacherModel(C=CFG.teacher_C, random_state=42).fit(Xtr_t, y_tr)
    tlog = teacher.decision_function(Xtr_t)
    sv = build_student_vectorizer(CFG.student_max_features, CFG.student_ngram_range, CFG.student_min_df)
    Xtr_sf = sv.fit_transform(tr["text"].tolist()); top = select_features_by_mi(Xtr_sf, y_tr, top_k=CFG.student_max_features)
    Xtr_s, Xte_s = Xtr_sf[:, top], sv.transform(te["text"].tolist())[:, top]
    stu = StudentModel(C=CFG.student_C, temperature=CFG.distill_temperature, alpha=CFG.distill_alpha,
                       epochs=CFG.distill_epochs, random_state=42); stu.fit_distilled(Xtr_s, y_tr, tlog)
    yps = stu.predict(Xte_s)
    R["student"] = {"yp_test": np.asarray(yps).astype(int).tolist(),
                    "test_macro_f1": float(f1_score(y_te, yps, average="macro"))}
    print("SEEDS_DONE")
    return R
