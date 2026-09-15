"""Camera-ready part 2: all conformal methods with confidence intervals + the new
ontology-hierarchical (concept-smoothed) method + Ding-style clustered baseline.
Loads cached per-example probabilities from part 1. Pure numpy/scipy."""
import json, io
from pathlib import Path
import numpy as np
from scipy.stats import beta
from sklearn.cluster import KMeans

RES = Path(__file__).resolve().parents[1] / "results"
FIG = Path(__file__).resolve().parents[1] / "figures"; FIG.mkdir(exist_ok=True)
d = np.load(RES / "cache.npz", allow_pickle=True)
P_cal, y_cal, lang_cal = d["P_cal"], d["y_cal"], d["lang_cal"].astype(str)
P_te, y_te, lang_te = d["P_te"], d["y_te"], d["lang_te"].astype(str)
K = 5; ALPHA = 0.10; TARGET = 1 - ALPHA
CLASS = ["not_related", "weather_disaster", "aid_request", "infrastructure", "other_related"]
CONCEPT = {0: "NotRelated", 1: "HazardEvent", 2: "ActionableNeed", 3: "ActionableNeed", 4: "RelatedOther"}


def cp(kc, n, a=0.05):
    if n == 0: return (float("nan"), float("nan"))
    lo = 0.0 if kc == 0 else float(beta.ppf(a / 2, kc, n - kc + 1))
    hi = 1.0 if kc == n else float(beta.ppf(1 - a / 2, kc + 1, n - kc))
    return lo, hi


def qlevel(n, a=ALPHA): return min(np.ceil((1 - a) * (n + 1)) / n, 1.0) if n > 0 else 1.0


def qhat(scores, a=ALPHA):
    scores = np.asarray(scores)
    return float(np.quantile(scores, qlevel(len(scores), a), method="higher")) if len(scores) else 1.0


def marginal():
    q = qhat(1 - P_cal[np.arange(len(y_cal)), y_cal]); return np.full(K, 1 - q)


def classcond():
    thr = np.ones(K)
    for k in range(K):
        m = y_cal == k; thr[k] = 1 - qhat(1 - P_cal[m, k]) if m.any() else 1.0
    return thr


def concept_q():
    cq = {}
    for c in set(CONCEPT.values()):
        ks = [k for k in range(K) if CONCEPT[k] == c]; m = np.isin(y_cal, ks)
        cq[c] = qhat(1 - P_cal[m, y_cal[m]]) if m.any() else 1.0
    return cq


def conceptcond():
    cq = concept_q(); return np.array([1 - cq[CONCEPT[k]] for k in range(K)])


def hierarchical(tau=50.0):
    cq = concept_q(); thr = np.ones(K); lam = {}
    for k in range(K):
        m = y_cal == k; nk = int(m.sum())
        qk = qhat(1 - P_cal[m, k]) if m.any() else 1.0
        lk = nk / (nk + tau); lam[CLASS[k]] = round(lk, 3)
        qsh = lk * qk + (1 - lk) * cq[CONCEPT[k]]; thr[k] = 1 - qsh
    return thr, lam


def ding(nclust=2):
    levels = np.linspace(0.5, 0.95, 10); emb = np.zeros((K, len(levels)))
    for k in range(K):
        m = y_cal == k; s = 1 - P_cal[m, k] if m.any() else np.array([1.0])
        emb[k] = [np.quantile(s, l, method="higher") for l in levels]
    lab = KMeans(n_clusters=nclust, n_init=10, random_state=42).fit_predict(emb)
    thr = np.ones(K)
    for c in range(nclust):
        ks = [k for k in range(K) if lab[k] == c]; m = np.isin(y_cal, ks)
        q = qhat(1 - P_cal[m, y_cal[m]]) if m.any() else 1.0
        for k in ks: thr[k] = 1 - q
    return thr, {CLASS[k]: int(lab[k]) for k in range(K)}


def sets(thr): return [np.where(P_te[i] >= thr)[0] for i in range(len(P_te))]


def langcond():
    ql = {}
    for lg in set(lang_cal):
        m = lang_cal == lg
        if m.sum() >= 30: ql[lg] = qhat(1 - P_cal[m, y_cal[m]])
    qg = qhat(1 - P_cal[np.arange(len(y_cal)), y_cal])
    return [np.where(P_te[i] >= 1 - ql.get(lang_te[i], qg))[0] for i in range(len(P_te))], ql


def evaluate(S):
    sizes = np.array([len(s) for s in S])
    kc = int(sum(y_te[i] in S[i] for i in range(len(y_te)))); lo, hi = cp(kc, len(y_te))
    per_class = {}
    for k in range(K):
        idx = np.where(y_te == k)[0]; kk = int(sum(k in S[i] for i in idx)); l, h = cp(kk, len(idx))
        per_class[CLASS[k]] = {"n": int(len(idx)), "coverage": round(kk / len(idx), 3) if len(idx) else None,
                               "ci95": [round(l, 3), round(h, 3)]}
    per_lang = {}
    for lg in sorted(set(lang_te)):
        idx = np.where(lang_te == lg)[0]
        if len(idx) < 10: continue
        kk = int(sum(y_te[i] in S[i] for i in idx)); l, h = cp(kk, len(idx))
        per_lang[lg] = {"n": int(len(idx)), "coverage": round(kk / len(idx), 3), "ci95": [round(l, 3), round(h, 3)]}
    rng = np.random.RandomState(0)
    bs = [sizes[rng.randint(0, len(sizes), len(sizes))].mean() for _ in range(2000)]
    return {"overall_coverage": round(float(kc / len(y_te)), 3), "overall_ci95": [round(lo, 3), round(hi, 3)],
            "avg_set_size": round(float(sizes.mean()), 3),
            "set_size_ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
            "singleton_rate": round(float((sizes == 1).mean()), 3), "empty_rate": round(float((sizes == 0).mean()), 3),
            "per_class": per_class, "per_lang": per_lang}


S_lang, ql = langcond()
R = {"alpha": ALPHA, "target": TARGET,
     "calib_per_class": {CLASS[k]: int((y_cal == k).sum()) for k in range(K)},
     "marginal": evaluate(sets(marginal())),
     "class_conditional": evaluate(sets(classcond())),
     "language_conditional": evaluate(S_lang),
     "concept_conditional": evaluate(sets(conceptcond()))}
R["language_conditional"]["lang_qhat"] = {k: round(v, 4) for k, v in ql.items()}
thr_h, lam = hierarchical(); Rh = evaluate(sets(thr_h)); Rh["shrinkage_lambda"] = lam; Rh["tau"] = 50.0
R["ontology_hierarchical_ours"] = Rh
thr_d, clust = ding(); Rd = evaluate(sets(thr_d)); Rd["clusters"] = clust
R["ding_clustered"] = Rd


def qhat_aps(a=ALPHA):
    order = np.argsort(-P_cal, axis=1); n = len(y_cal); s = np.empty(n)
    for i in range(n):
        cum = np.cumsum(P_cal[i][order[i]]); rank = np.where(order[i] == y_cal[i])[0][0]; s[i] = cum[rank]
    return float(np.quantile(s, qlevel(n, a), method="higher"))


def aps_sets():
    q = qhat_aps(); order = np.argsort(-P_te, axis=1); out = []
    for i in range(len(P_te)):
        cum = np.cumsum(P_te[i][order[i]]); k = int(np.searchsorted(cum, q)) + 1; out.append(order[i][:min(k, K)])
    return out


Saps = aps_sets(); sizes_aps = np.array([len(s) for s in Saps])
cov_aps = float(np.mean([y_te[i] in Saps[i] for i in range(len(y_te))]))
R["method_note_R2_9"] = {"score_overall_coverage": R["marginal"]["overall_coverage"],
    "aps_overall_coverage": round(cov_aps, 3), "aps_avg_set": round(float(sizes_aps.mean()), 3),
    "explanation": ("APS is conservative on a 5-class label space (its cumulative-mass rule overshoots), "
                    "so APS over-covers (~0.96) versus the score rule (~0.91, near the 0.90 target). "
                    "Table 4 used APS; reporting the score rule (or noting APS's known conservativeness) removes the puzzle.")}
io.open(RES / "cr_conformal.json", "w", encoding="utf-8").write(json.dumps(R, indent=2))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
methods = [("marginal", "#94A3B8", "Marginal"),
           ("class_conditional", "#3AAFA9", "Class-conditional"),
           ("ontology_hierarchical_ours", "#7C3AED", "Ontology-hierarchical (ours)")]
x = np.arange(K); w = 0.26
fig, ax = plt.subplots(figsize=(9, 4.6))
for j, (mname, col, lbl) in enumerate(methods):
    cov = [R[mname]["per_class"][c]["coverage"] for c in CLASS]
    lo = [cov[i] - R[mname]["per_class"][CLASS[i]]["ci95"][0] for i in range(K)]
    hi = [R[mname]["per_class"][CLASS[i]]["ci95"][1] - cov[i] for i in range(K)]
    ax.bar(x + (j - 1) * w, cov, w, yerr=[lo, hi], capsize=3, label=lbl, color=col, error_kw={"lw": 1})
ax.axhline(TARGET, ls="--", color="k", lw=1, label=f"target {TARGET:.2f}")
ax.set_xticks(x); ax.set_xticklabels([f"{c}\n(n={R['marginal']['per_class'][c]['n']})" for c in CLASS], fontsize=8)
ax.set_ylabel("empirical coverage (95% Clopper-Pearson CI)"); ax.set_ylim(0, 1.08)
ax.set_title("Per-class conformal coverage with confidence intervals"); ax.legend(fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(FIG / "mondrian_coverage_ci.png", dpi=220, bbox_inches="tight"); plt.close(fig)

# per-language coverage figure: marginal vs language-conditional (with CIs)
langs = sorted(R["marginal"]["per_lang"].keys())
xl = np.arange(len(langs)); w2 = 0.36
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for j, (mname, col, lbl) in enumerate([("marginal", "#94A3B8", "Marginal"), ("language_conditional", "#4C6EF5", "Language-conditional")]):
    cov = [R[mname]["per_lang"][lg]["coverage"] for lg in langs]
    lo = [cov[i] - R[mname]["per_lang"][langs[i]]["ci95"][0] for i in range(len(langs))]
    hi = [R[mname]["per_lang"][langs[i]]["ci95"][1] - cov[i] for i in range(len(langs))]
    ax.bar(xl + (j - 0.5) * w2, cov, w2, yerr=[lo, hi], capsize=3, label=lbl, color=col, error_kw={"lw": 1})
ax.axhline(TARGET, ls="--", color="k", lw=1, label=f"target {TARGET:.2f}")
ax.set_xticks(xl); ax.set_xticklabels([f"{lg}\n(n={R['marginal']['per_lang'][lg]['n']})" for lg in langs])
ax.set_ylabel("empirical coverage (95% CI)"); ax.set_ylim(0, 1.08)
ax.set_title("Per-language conformal coverage with confidence intervals"); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "lang_coverage_ci.png", dpi=220, bbox_inches="tight"); plt.close(fig)

print("saved cr_conformal.json + mondrian_coverage_ci.png + lang_coverage_ci.png")
print("language-conditional per-lang:", {lg: (R["language_conditional"]["per_lang"][lg]["coverage"], R["language_conditional"]["per_lang"][lg]["ci95"]) for lg in langs})
for m in ["marginal", "class_conditional", "ontology_hierarchical_ours", "ding_clustered", "concept_conditional"]:
    infra = R[m]["per_class"]["infrastructure"]
    print(f"{m:30s} overall={R[m]['overall_coverage']} set={R[m]['avg_set_size']} infra_cov={infra['coverage']} CI{infra['ci95']}")
print("R2.9 APS vs score coverage:", R["method_note_R2_9"]["aps_overall_coverage"], "vs", R["method_note_R2_9"]["score_overall_coverage"])
print("hierarchical shrinkage lambda:", lam)
print("ding clusters:", clust)
