"""Camera-ready part 7: run the FULL conformal suite (incl. the new concept-smoothed
method + tau-sweep) on the transformer backbones, and PAIRED macro-F1 tests
(student vs mBERT/XLM-R) on the same test examples. Answers: does the novelty
generalise beyond the linear student, and is the accuracy gap significant?
Needs transformer_probs.npz (from run_cr_beam_probs.py). Writes cr_transformer_conformal.json."""
import json, io
from pathlib import Path
import numpy as np
from scipy.stats import beta
from sklearn.metrics import f1_score

RES = Path(__file__).resolve().parents[1] / "results"
T = np.load(RES / "transformer_probs.npz", allow_pickle=True)
C = np.load(RES / "cache.npz", allow_pickle=True)
y_cal, y_te = T["y_cal"], T["y_te"]
lang_cal, lang_te = T["lang_cal"].astype(str), T["lang_te"].astype(str)
assert np.array_equal(y_te, C["y_te"]), "split mismatch: transformer y_te != cache y_te"
K, A, INF = 5, 0.10, 3
CLASS = ["not_related", "weather_disaster", "aid_request", "infrastructure", "other_related"]
CONCEPT = {0: "NotRelated", 1: "HazardEvent", 2: "ActionableNeed", 3: "ActionableNeed", 4: "RelatedOther"}

def cp(kc, n, a=0.05):
    if n == 0: return (float("nan"), float("nan"))
    lo = 0.0 if kc == 0 else float(beta.ppf(a/2, kc, n-kc+1)); hi = 1.0 if kc == n else float(beta.ppf(1-a/2, kc+1, n-kc))
    return round(lo, 3), round(hi, 3)
def qlevel(n): return min(np.ceil((1-A)*(n+1))/n, 1.0) if n > 0 else 1.0
def qhat(s):
    s = np.asarray(s); return float(np.quantile(s, qlevel(len(s)), method="higher")) if len(s) else 1.0

def marg(Pc, yc): return np.full(K, 1-qhat(1-Pc[np.arange(len(yc)), yc]))
def classc(Pc, yc):
    t = np.ones(K)
    for k in range(K):
        m = yc == k; t[k] = 1-qhat(1-Pc[m, k]) if m.any() else 1.0
    return t
def conceptq(Pc, yc):
    cq = {}
    for c in set(CONCEPT.values()):
        ks = [k for k in range(K) if CONCEPT[k] == c]; m = np.isin(yc, ks)
        cq[c] = qhat(1-Pc[m, yc[m]]) if m.any() else 1.0
    return cq
def hier(Pc, yc, tau=10.0):
    cq = conceptq(Pc, yc); t = np.ones(K)
    for k in range(K):
        m = yc == k; nk = int(m.sum()); qk = qhat(1-Pc[m, k]) if m.any() else 1.0
        lk = nk/(nk+tau); t[k] = 1-(lk*qk+(1-lk)*cq[CONCEPT[k]])
    return t
def sets(thr, Pt): return [np.where(Pt[i] >= thr)[0] for i in range(len(Pt))]
def infra(S):
    idx = np.where(y_te == INF)[0]; kk = int(sum(INF in S[i] for i in idx))
    return round(kk/len(idx), 3), list(cp(kk, len(idx))), kk, len(idx)
def overall(S):
    kk = int(sum(y_te[i] in S[i] for i in range(len(y_te)))); return round(kk/len(y_te), 3)
def avgset(S): return round(float(np.mean([len(s) for s in S])), 3)
def lang_cov(S, lg):
    idx = np.where(lang_te == lg)[0]; kk = int(sum(y_te[i] in S[i] for i in idx))
    return round(kk/len(idx), 3), list(cp(kk, len(idx))), len(idx)

def boot_f1_diff(yp_a, yp_b, reps=3000, seed=0):
    # paired bootstrap of macro-F1(b) - macro-F1(a) on the same test set
    rng = np.random.RandomState(seed); n = len(y_te); dif = []
    for _ in range(reps):
        r = rng.randint(0, n, n)
        fa = f1_score(y_te[r], yp_a[r], average="macro", labels=list(range(K)), zero_division=0)
        fb = f1_score(y_te[r], yp_b[r], average="macro", labels=list(range(K)), zero_division=0)
        dif.append(fb-fa)
    dif = np.array(dif)
    return {"delta_mean": round(float(dif.mean()), 4),
            "ci95": [round(float(np.percentile(dif, 2.5)), 4), round(float(np.percentile(dif, 97.5)), 4)],
            "frac_transformer_better": round(float((dif > 0).mean()), 4)}

yp_student = C["P_te"].argmax(1)
out = {"alpha": A, "note": "Concept-smoothed uses tau=10 (Pareto region matching class-cond coverage at smaller sets)."}
for tag, disp in [("mbert", "mBERT"), ("xlmr", "XLM-R")]:
    Pc, Pt = T[f"{tag}_P_cal"], T[f"{tag}_P_te"]
    Sm, Sc, Sh = sets(marg(Pc, yc=y_cal), Pt), sets(classc(Pc, y_cal), Pt), sets(hier(Pc, y_cal, 10.0), Pt)
    im, ic, ih = infra(Sm), infra(Sc), infra(Sh)
    # tau sweep on transformer
    sweep = []
    for tau in [0, 5, 10, 20, 50, 100, 1e9]:
        thr = classc(Pc, y_cal) if tau == 0 else hier(Pc, y_cal, tau if tau > 0 else 1e-9)
        S = sets(thr, Pt); iv, _, _, _ = infra(S)
        sweep.append({"tau": ("inf" if tau >= 1e8 else tau), "infra": iv, "overall": overall(S), "set": avgset(S)})
    yp = Pt.argmax(1)
    out[disp] = {
        "macro_f1": round(float(f1_score(y_te, yp, average="macro")), 4),
        "infra_marginal": {"cov": im[0], "ci95": im[1], "k_n": [im[2], im[3]]},
        "infra_classcond": {"cov": ic[0], "ci95": ic[1], "k_n": [ic[2], ic[3]]},
        "infra_ours_tau10": {"cov": ih[0], "ci95": ih[1], "k_n": [ih[2], ih[3]]},
        "set_marginal": avgset(Sm), "set_classcond": avgset(Sc), "set_ours_tau10": avgset(Sh),
        "overall_marginal": overall(Sm), "overall_classcond": overall(Sc), "overall_ours_tau10": overall(Sh),
        "french": {"marginal": lang_cov(Sm, "fr"), "lang_cond_na": "see linear; transformer uses class/marginal here"},
        "tau_sweep": sweep,
        "paired_f1_vs_student": boot_f1_diff(yp_student, yp)}

io.open(RES / "cr_transformer_conformal.json", "w", encoding="utf-8").write(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
