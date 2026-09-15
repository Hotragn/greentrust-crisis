"""Camera-ready Beam GPU job: SOTA multilingual baselines (mBERT, XLM-R) with
Mondrian conformal, SAME-HOST energy (transformer vs linear student timed in one
container), and a real-RAPL attempt. Returns compact summaries (covered counts,
so Clopper-Pearson CIs are computed locally). Run via run_cr_beam.py (.remote())."""
from beam import function, Image

image = (Image(python_version="python3.11")
         .add_python_packages(["torch", "transformers", "scikit-learn", "pandas",
                               "pyarrow", "numpy", "huggingface_hub", "sentencepiece"]))


@function(gpu="RTX4090", cpu=8, memory="24Gi", image=image, timeout=3000)
def run():
    import sys, time, json, os, gc, copy, urllib.request
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
    K, MAXLEN, EPOCHS, BS = 5, 96, 3, 32
    CLASS = ["not_related", "weather_disaster", "aid_request", "infrastructure", "other_related"]
    df = build_unified_dataframe(); tr, cal, te = stratified_split(df)
    y_tr, y_cal, y_te = tr["y_crisis_type"].values, cal["y_crisis_type"].values, te["y_crisis_type"].values
    lang_cal, lang_te = cal["lang"].values.astype(str), te["lang"].values.astype(str)

    def qlevel(n, a=0.1): return min(np.ceil((1 - a) * (n + 1)) / n, 1.0) if n > 0 else 1.0
    def qh(s, a=0.1):
        s = np.asarray(s); return float(np.quantile(s, qlevel(len(s), a), method="higher")) if len(s) else 1.0
    def counts(S):
        pc = {CLASS[k]: {"n": int((y_te == k).sum()), "covered": int(sum(k in S[i] for i in np.where(y_te == k)[0]))} for k in range(K)}
        pl = {}
        for lg in sorted(set(lang_te)):
            idx = np.where(lang_te == lg)[0]
            if len(idx) < 10: continue
            pl[lg] = {"n": int(len(idx)), "covered": int(sum(y_te[i] in S[i] for i in idx))}
        sizes = np.array([len(s) for s in S])
        return {"overall_covered": int(sum(y_te[i] in S[i] for i in range(len(y_te)))), "overall_n": int(len(y_te)),
                "avg_set_size": round(float(sizes.mean()), 3), "per_class": pc, "per_lang": pl}
    def conformal_all(P_cal, P_te):
        qg = qh(1 - P_cal[np.arange(len(y_cal)), y_cal]); Sm = [np.where(P_te[i] >= 1 - qg)[0] for i in range(len(P_te))]
        thr = np.ones(K)
        for k in range(K):
            m = y_cal == k; thr[k] = 1 - qh(1 - P_cal[m, k]) if m.any() else 1.0
        Sc = [np.where(P_te[i] >= thr)[0] for i in range(len(P_te))]
        ql = {lg: qh(1 - P_cal[lang_cal == lg, y_cal[lang_cal == lg]]) for lg in set(lang_cal) if (lang_cal == lg).sum() >= 30}
        Sl = [np.where(P_te[i] >= 1 - ql.get(lang_te[i], qg))[0] for i in range(len(P_te))]
        return {"marginal": counts(Sm), "class_conditional": counts(Sc), "language_conditional": counts(Sl)}

    def enc(tok, texts): return tok(list(texts), truncation=True, padding="max_length", max_length=MAXLEN, return_tensors="pt")
    R = {"device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "epochs": EPOCHS, "models": {}}

    for MODEL in ["bert-base-multilingual-cased", "xlm-roberta-base"]:
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
                              "conformal": conformal_all(P_cal, P_te)}
        del model, etr; gc.collect(); torch.cuda.empty_cache() if dev == "cuda" else None

    # same-host linear student
    tv = build_teacher_vectorizer(CFG.teacher_max_features, CFG.teacher_ngram_range, CFG.teacher_min_df)
    Xtr_t = tv.fit_transform(tr["text"].tolist()); teacher = TeacherModel(C=CFG.teacher_C, random_state=42).fit(Xtr_t, y_tr); tlog = teacher.decision_function(Xtr_t)
    sv = build_student_vectorizer(CFG.student_max_features, CFG.student_ngram_range, CFG.student_min_df)
    Xtr_sf = sv.fit_transform(tr["text"].tolist()); top = select_features_by_mi(Xtr_sf, y_tr, top_k=CFG.student_max_features)
    Xtr_s, Xte_s = Xtr_sf[:, top], sv.transform(te["text"].tolist())[:, top]
    stu = StudentModel(C=CFG.student_C, temperature=CFG.distill_temperature, alpha=CFG.distill_alpha, epochs=CFG.distill_epochs, random_state=42); stu.fit_distilled(Xtr_s, y_tr, tlog)
    x1 = Xte_s[:1]
    for _ in range(5): stu.predict_proba(x1)
    xs = [(lambda t: ((stu.predict_proba(x1)), (time.perf_counter() - t) * 1000)[1])(time.perf_counter()) for _ in range(300)]
    R["linear_student_same_host"] = {"cpu_latency_ms_p50": round(float(np.percentile(xs, 50)), 4),
                                     "test_macro_f1": float(f1_score(y_te, stu.predict(Xte_s), average="macro"))}
    rapl = "/sys/class/powercap/intel-rapl:0/energy_uj"; ri = {"available": os.path.exists(rapl)}
    if os.path.exists(rapl):
        try:
            a = int(open(rapl).read()); t = time.perf_counter()
            for _ in range(500): stu.predict_proba(x1)
            b = int(open(rapl).read()); ri.update({"linear_uJ_per_inf": (b - a) / 500.0, "burst_s": round(time.perf_counter() - t, 3)})
        except Exception as ex: ri["error"] = str(ex)
    R["rapl"] = ri
    R["energy_note"] = "energy_mJ = cpu_latency_ms_p50 * 65 (same-host TDP model, 65W); RAPL microjoules used if available"
    print("RESULT_JSON_START"); print(json.dumps(R)); print("RESULT_JSON_END")
    return R
