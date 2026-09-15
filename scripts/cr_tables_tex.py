"""Emit exact LaTeX table bodies for the camera-ready, straight from the result
files, so nothing is transcribed by hand."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta

RES = Path(__file__).resolve().parents[1] / "results"
K, A, INF = 5, 0.10, 3
CLASS = ["not\\_related", "weather\\_disaster", "aid\\_request", "infrastructure", "other\\_related"]
PLAIN = ["not_related", "weather_disaster", "aid_request", "infrastructure", "other_related"]
CONCEPT = {0: "N", 1: "H", 2: "AN", 3: "AN", 4: "R"}

def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else float(beta.ppf(a/2, k, n-k+1))
    hi = 1.0 if k == n else float(beta.ppf(1-a/2, k+1, n-k))
    return lo, hi
def ci(k, n):
    lo, hi = cp(k, n); return f"[{lo:.2f},{hi:.2f}]"
def qlevel(n): return min(np.ceil((1-A)*(n+1))/n, 1.0) if n > 0 else 1.0
def qhat(s):
    s = np.asarray(s); return float(np.quantile(s, qlevel(len(s)), method="higher")) if len(s) else 1.0

C = np.load(RES/"cache.npz", allow_pickle=True)
P_cal, y_cal, P_te, y_te = C["P_cal"], C["y_cal"], C["P_te"], C["y_te"]
lang_cal, lang_te = C["lang_cal"].astype(str), C["lang_te"].astype(str)

def marg(): return np.full(K, 1-qhat(1-P_cal[np.arange(len(y_cal)), y_cal]))
def classc():
    t = np.ones(K)
    for k in range(K):
        m = y_cal == k; t[k] = 1-qhat(1-P_cal[m, k]) if m.any() else 1.0
    return t
def conceptq():
    cq = {}
    for c in set(CONCEPT.values()):
        ks = [k for k in range(K) if CONCEPT[k] == c]; m = np.isin(y_cal, ks)
        cq[c] = qhat(1-P_cal[m, y_cal[m]]) if m.any() else 1.0
    return cq
def hier(tau):
    cq = conceptq(); t = np.ones(K)
    for k in range(K):
        m = y_cal == k; nk = int(m.sum()); qk = qhat(1-P_cal[m, k]) if m.any() else 1.0
        lk = nk/(nk+tau); t[k] = 1-(lk*qk+(1-lk)*cq[CONCEPT[k]])
    return t
def langc():
    ql = {lg: qhat(1-P_cal[lang_cal == lg, y_cal[lang_cal == lg]])
          for lg in set(lang_cal) if (lang_cal == lg).sum() >= 30}
    qg = qhat(1-P_cal[np.arange(len(y_cal)), y_cal])
    return [set(np.where(P_te[i] >= 1-ql.get(lang_te[i], qg))[0].tolist()) for i in range(len(P_te))]
def sets(thr): return [set(np.where(P_te[i] >= thr)[0].tolist()) for i in range(len(P_te))]
def stats(S):
    cov_k = {}
    for k in range(K):
        idx = np.where(y_te == k)[0]; kk = int(sum(k in S[i] for i in idx))
        cov_k[k] = (kk, len(idx))
    cov_l = {}
    for lg in ["en", "fr", "ht"]:
        idx = np.where(lang_te == lg)[0]; kk = int(sum(y_te[i] in S[i] for i in idx))
        cov_l[lg] = (kk, len(idx))
    ov = float(np.mean([y_te[i] in S[i] for i in range(len(y_te))]))
    sz = float(np.mean([len(s) for s in S]))
    return cov_k, cov_l, ov, sz

Sm, Sc, Sh, Sl = sets(marg()), sets(classc()), sets(hier(10.0)), langc()
km, lm, om, zm = stats(Sm); kc, lc, oc, zc = stats(Sc)
kh, lh, oh, zh = stats(Sh); kl, ll, ol, zl = stats(Sl)

print("="*78); print("TABLE: group-conditional coverage (paste body)"); print("="*78)
print(f"Overall coverage        & {om:.3f} & {oc:.3f} & {oh:.3f} & {ol:.3f} \\\\")
print(f"Mean set size           & {zm:.2f}  & {zc:.2f}  & {zh:.2f}  & {zl:.2f} \\\\")
print("\\midrule")
print("\\multicolumn{5}{l}{\\textit{Per-class coverage, 95\\% CI in brackets}}\\\\")
for k in range(K):
    b = "\\textbf" if k == INF else ""
    a1, n1 = km[k]; a2, _ = kc[k]; a3, _ = kh[k]
    row = (f"\\quad {CLASS[k]} ($n{{=}}{n1}$) & {a1/n1:.3f}\\,{ci(a1,n1)} & {a2/n1:.3f}\\,{ci(a2,n1)} "
           f"& {a3/n1:.3f}\\,{ci(a3,n1)} & --- \\\\")
    if k == INF:
        row = (f"\\quad \\textbf{{{CLASS[k]}}} ($n{{=}}{n1}$) & \\textbf{{{a1/n1:.3f}}}\\,{ci(a1,n1)} "
               f"& \\textbf{{{a2/n1:.3f}}}\\,{ci(a2,n1)} & \\textbf{{{a3/n1:.3f}}}\\,{ci(a3,n1)} & --- \\\\")
    print(row)
print("\\midrule")
print("\\multicolumn{5}{l}{\\textit{Per-language coverage}}\\\\")
NAMES = {"en": "English (en)", "fr": "French (fr)", "ht": "Haitian Creole (ht)"}
for lg in ["en", "fr", "ht"]:
    a1, n1 = lm[lg]; a4, _ = ll[lg]
    if lg == "fr":
        print(f"\\quad \\textbf{{{NAMES[lg]}}} ($n{{=}}{n1}$) & \\textbf{{{a1/n1:.3f}}}\\,{ci(a1,n1)} & --- & --- & \\textbf{{{a4/n1:.3f}}}\\,{ci(a4,n1)} \\\\")
    else:
        print(f"\\quad {NAMES[lg]} ($n{{=}}{n1}$) & {a1/n1:.3f}\\,{ci(a1,n1)} & --- & --- & {a4/n1:.3f}\\,{ci(a4,n1)} \\\\")

print()
print("="*78); print("TABLE: multi-seed baselines"); print("="*78)
SD = json.load(open(RES/"cr_seeds.json", encoding="utf-8"))
print(f"student macro-F1 = {SD['student_macro_f1']:.4f}")
for disp, m in SD["models"].items():
    f = m["macro_f1_mean"]; sd = m["macro_f1_sd"]
    print(f"{disp:8s} & {f:.3f}$\\,\\pm\\,${sd:.3f} & {m['macro_f1_min']:.3f}--{m['macro_f1_max']:.3f} "
          f"& $+{m['delta_vs_student_mean']:.3f}$ & {m['seeds_with_significant_advantage']}/4 \\\\")

print()
print("="*78); print("TABLE: ablation (accuracy panel)"); print("="*78)
B = json.load(open(RES/"cr_base_metrics.json", encoding="utf-8"))
for nm, key in [("Teacher (TF-IDF LR)", "teacher"), ("Student, hard labels", "student_hard"),
                ("Student, distilled ($T{=}6$)", "student_distilled")]:
    d = B[key]; lo, hi = d["macro_f1_ci95"]
    print(f"{nm} & {d['macro_f1']:.3f} & [{lo:.3f},{hi:.3f}] \\\\")

print()
print("="*78); print("TABLE: temperature (score rule + APS)"); print("="*78)
T = json.load(open(RES/"cr_temperature.json", encoding="utf-8"))["sweep"]
print("$T$ & " + " & ".join(str(int(r["T"])) for r in T) + " \\\\")
print("macro-F1 & " + " & ".join(f"{r['macro_f1']:.3f}" for r in T) + " \\\\")
print("coverage (score) & " + " & ".join(f"{r['score_coverage']:.3f}" for r in T) + " \\\\")
print("set size (score) & " + " & ".join(f"{r['score_set']:.2f}" for r in T) + " \\\\")
print("coverage (APS) & " + " & ".join(f"{r['aps_coverage']:.3f}" for r in T) + " \\\\")
print("set size (APS) & " + " & ".join(f"{r['aps_set']:.2f}" for r in T) + " \\\\")

print()
print("="*78); print("TABLE: tau frontier"); print("="*78)
R = json.load(open(RES/"cr_robustness.json", encoding="utf-8"))
for s in R["tau_sweep"]:
    t = "$\\infty$" if s["tau"] == "inf" else str(s["tau"])
    print(f"{t} & {s['lambda_infra']:.2f} & {s['infra_cov']:.3f} & {s['overall_cov']:.3f} & {s['avg_set']:.2f} \\\\")
rr = R["repeated_resplits"]
print(f"[resplits] cc>marg {rr['frac_classcond_gt_marginal']:.3f}, ours>marg {rr['frac_ours_gt_marginal']:.3f}")

print()
print("="*78); print("Ding + significance + separation"); print("="*78)
CF = json.load(open(RES/"cr_conformal.json", encoding="utf-8"))
dg = CF["ding_clustered"]
print("Ding: infra", dg["per_class"]["infrastructure"]["coverage"], "overall", dg["overall_coverage"], "set", dg["avg_set_size"])
SG = json.load(open(RES/"cr_significance.json", encoding="utf-8"))
print("McNemar infra marg->cc:", SG["coverage_comparisons"]["infra_marginal_vs_classcond"])
print("McNemar fr marg->lang:", SG["coverage_comparisons"]["french_marginal_vs_langcond"])
SV = json.load(open(RES/"cr_seed_variability.json", encoding="utf-8"))
print("separation:", SV["marginal_range"], SV["classcond_range"], "overlap:", SV["methods_overlap"],
      SV["classcond_at_or_above_17of19"])
TV = json.load(open(RES/"cr_tau_cv.json", encoding="utf-8"))
print("tau CV rule:", TV["selection_rule"]); print("tau*:", TV["tau_star"]); print("test at tau*:", TV["test_at_tau_star"]); print("test class-cond:", TV["test_class_conditional"]); print("ruleA:", TV["rule_A_aggressive"])
