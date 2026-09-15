"""Camera-ready part 5: PAIRED significance tests (the correct way to compare
conformal methods on the same test items). Per-method Clopper-Pearson CIs answer
"is this method's coverage near target"; they are NOT the right tool to compare two
methods on shared examples. Here we use McNemar's exact test and paired bootstraps.
Writes cr_significance.json. Pure numpy/scipy, loads cache.npz."""
import json, io
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

RES = Path(__file__).resolve().parents[1] / "results"
d = np.load(RES / "cache.npz", allow_pickle=True)
P_cal, y_cal, lang_cal = d["P_cal"], d["y_cal"], d["lang_cal"].astype(str)
P_te, y_te, lang_te = d["P_te"], d["y_te"], d["lang_te"].astype(str)
K, A, INF = 5, 0.10, 3
CONCEPT = {0: "NotRelated", 1: "HazardEvent", 2: "ActionableNeed", 3: "ActionableNeed", 4: "RelatedOther"}

def qlevel(n): return min(np.ceil((1 - A) * (n + 1)) / n, 1.0) if n > 0 else 1.0
def qhat(s):
    s = np.asarray(s); return float(np.quantile(s, qlevel(len(s)), method="higher")) if len(s) else 1.0
def marg(): return np.full(K, 1 - qhat(1 - P_cal[np.arange(len(y_cal)), y_cal]))
def cc():
    t = np.ones(K)
    for k in range(K):
        m = y_cal == k; t[k] = 1 - qhat(1 - P_cal[m, k]) if m.any() else 1.0
    return t
def conceptq():
    cq = {}
    for c in set(CONCEPT.values()):
        ks = [k for k in range(K) if CONCEPT[k] == c]; m = np.isin(y_cal, ks)
        cq[c] = qhat(1 - P_cal[m, y_cal[m]]) if m.any() else 1.0
    return cq
def hier(tau=50.0):
    cq = conceptq(); t = np.ones(K)
    for k in range(K):
        m = y_cal == k; nk = int(m.sum()); qk = qhat(1 - P_cal[m, k]) if m.any() else 1.0
        lk = nk / (nk + tau); t[k] = 1 - (lk * qk + (1 - lk) * cq[CONCEPT[k]])
    return t
def langc():
    ql = {lg: qhat(1 - P_cal[lang_cal == lg, y_cal[lang_cal == lg]])
          for lg in set(lang_cal) if (lang_cal == lg).sum() >= 30}
    qg = qhat(1 - P_cal[np.arange(len(y_cal)), y_cal])
    return [set(np.where(P_te[i] >= 1 - ql.get(lang_te[i], qg))[0].tolist()) for i in range(len(P_te))]
def sets(thr): return [set(np.where(P_te[i] >= thr)[0].tolist()) for i in range(len(P_te))]
def covered(S): return np.array([y_te[i] in S[i] for i in range(len(y_te))])
def sizes(S): return np.array([len(s) for s in S])

Sm, Sc, Sh, Sl = sets(marg()), sets(cc()), sets(hier()), langc()
cm, cc_, ch, cl = covered(Sm), covered(Sc), covered(Sh), covered(Sl)

def mcnemar(cvA, cvB, mask):
    a, b = cvA[mask].astype(bool), cvB[mask].astype(bool)
    b_only, c_only = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(min(b_only, c_only), b_only + c_only, 0.5).pvalue if (b_only + c_only) else 1.0
    return {"n": int(mask.sum()), "A_only": b_only, "B_only": c_only, "p": round(float(p), 5),
            "significant": bool(p < 0.05)}

def paired_boot(cvA, cvB, mask, reps=10000, seed=0):
    idx = np.where(mask)[0]; rng = np.random.RandomState(seed); dif = []
    for _ in range(reps):
        r = rng.choice(idx, len(idx), replace=True); dif.append(cvB[r].mean() - cvA[r].mean())
    dif = np.array(dif)
    return {"delta": round(float(cvB[idx].mean() - cvA[idx].mean()), 3),
            "ci95": [round(float(np.percentile(dif, 2.5)), 3), round(float(np.percentile(dif, 97.5)), 3)],
            "frac_positive": round(float((dif > 0).mean()), 4)}

infm, frm = y_te == INF, lang_te == "fr"
out = {"note": ("Per-method Clopper-Pearson CIs (in cr_conformal.json) can OVERLAP even when a "
                "paired test shows a significant difference; McNemar/paired-bootstrap are the correct "
                "comparison on shared test items."),
       "coverage_comparisons": {
           "infra_marginal_vs_classcond": mcnemar(cm, cc_, infm),
           "infra_marginal_vs_ours": mcnemar(cm, ch, infm),
           "infra_ours_vs_classcond": mcnemar(ch, cc_, infm),
           "french_marginal_vs_langcond": mcnemar(cm, cl, frm)},
       "infra_delta_classcond_minus_marginal": paired_boot(cm, cc_, infm),
       "infra_delta_ours_minus_marginal": paired_boot(cm, ch, infm)}

# set-size advantage of ours vs class-conditional (well powered: all n=5250)
sz_c, sz_h = sizes(Sc), sizes(Sh); n = len(sz_c); rng = np.random.RandomState(0); dif = []
for _ in range(10000):
    r = rng.randint(0, n, n); dif.append(sz_c[r].mean() - sz_h[r].mean())
dif = np.array(dif)
out["setsize_ours_vs_classcond"] = {
    "classcond_mean": round(float(sz_c.mean()), 3), "ours_mean": round(float(sz_h.mean()), 3),
    "delta_cc_minus_ours": round(float(sz_c.mean() - sz_h.mean()), 3),
    "ci95": [round(float(np.percentile(dif, 2.5)), 3), round(float(np.percentile(dif, 97.5)), 3)],
    "frac_positive": round(float((dif > 0).mean()), 4),
    "ours_smaller_n": int((sz_c - sz_h > 0).sum()), "ours_larger_n": int((sz_c - sz_h < 0).sum()),
    "equal_n": int((sz_c - sz_h == 0).sum()), "total": int(n)}

# cross-backbone infra flips from baselines.json (bounded McNemar; marginal covered is small)
B = json.load(open(RES / "baselines.json", encoding="utf-8"))
xb = {"linear_student": mcnemar(cm, cc_, infm)}
for mdl, disp in [("bert-base-multilingual-cased", "mBERT"), ("xlm-roberta-base", "XLM-R")]:
    c = B["models"][mdl]["conformal"]
    km = c["marginal"]["per_class"]["infrastructure"]["covered"]
    kc = c["class_conditional"]["per_class"]["infrastructure"]["covered"]
    nn = c["marginal"]["per_class"]["infrastructure"]["n"]
    b_only, c_only = max(0, km - kc), max(0, kc - km)
    p = binomtest(min(b_only, c_only), b_only + c_only, 0.5).pvalue if (b_only + c_only) else 1.0
    xb[disp] = {"n": nn, "marginal_covered": km, "classcond_covered": kc,
                "net_flips_ge": c_only, "p_bound": round(float(p), 5), "significant": bool(p < 0.05)}
out["cross_backbone_infra_flips"] = xb

io.open(RES / "cr_significance.json", "w", encoding="utf-8").write(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
