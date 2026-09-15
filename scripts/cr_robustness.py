"""Camera-ready part 6: robustness improvements that address the n=19 weakness.
(A) Repeated stratified calib/test resplits (model fixed) -> the marginal->Mondrian
    infra effect is stable, not a one-split fluke.
(B) tau-sweep for the concept-smoothed method -> a tunable coverage<->set-size
    frontier instead of a single tau=50 point.
Uses only cached out-of-train probabilities. Writes cr_robustness.json + pareto fig."""
import json, io
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

RES = Path(__file__).resolve().parents[1] / "results"; FIG = Path(__file__).resolve().parents[1] / "figures"
d = np.load(RES / "cache.npz", allow_pickle=True)
# pool all out-of-train examples (calib + test come from the same fixed model)
P = np.vstack([d["P_cal"], d["P_te"]]); y = np.concatenate([d["y_cal"], d["y_te"]])
K, A, INF = 5, 0.10, 3
CONCEPT = {0: "NotRelated", 1: "HazardEvent", 2: "ActionableNeed", 3: "ActionableNeed", 4: "RelatedOther"}
TEST_FRAC = 5250 / len(y)

def qlevel(n): return min(np.ceil((1 - A) * (n + 1)) / n, 1.0) if n > 0 else 1.0
def qhat(s):
    s = np.asarray(s); return float(np.quantile(s, qlevel(len(s)), method="higher")) if len(s) else 1.0

def thresholds(Pc, yc, tau=50.0):
    # marginal
    tm = np.full(K, 1 - qhat(1 - Pc[np.arange(len(yc)), yc]))
    # class-conditional
    tc = np.ones(K)
    for k in range(K):
        m = yc == k; tc[k] = 1 - qhat(1 - Pc[m, k]) if m.any() else 1.0
    # concept pooled
    cq = {}
    for c in set(CONCEPT.values()):
        ks = [k for k in range(K) if CONCEPT[k] == c]; m = np.isin(yc, ks)
        cq[c] = qhat(1 - Pc[m, yc[m]]) if m.any() else 1.0
    th = np.ones(K)
    for k in range(K):
        m = yc == k; nk = int(m.sum()); qk = qhat(1 - Pc[m, k]) if m.any() else 1.0
        lk = nk / (nk + tau); th[k] = 1 - (lk * qk + (1 - lk) * cq[CONCEPT[k]])
    return tm, tc, th

def infra_cov(thr, Pt, yt):
    idx = np.where(yt == INF)[0]
    if len(idx) == 0: return np.nan, 0
    cov = np.mean([INF in np.where(Pt[i] >= thr)[0] for i in idx]); return float(cov), len(idx)

# ---------- (A) repeated stratified resplits ----------
KREP = 300; rng = np.random.RandomState(0)
mm, cc, hh, dpos_cc, dpos_h, infra_ns = [], [], [], 0, 0, []
for rep in range(KREP):
    ci, ti = train_test_split(np.arange(len(y)), test_size=TEST_FRAC, stratify=y, random_state=rep)
    Pc, yc, Pt, yt = P[ci], y[ci], P[ti], y[ti]
    tm, tc, th = thresholds(Pc, yc)
    cm, _ = infra_cov(tm, Pt, yt); cc_, ninf = infra_cov(tc, Pt, yt); ch, _ = infra_cov(th, Pt, yt)
    mm.append(cm); cc.append(cc_); hh.append(ch); infra_ns.append(ninf)
    dpos_cc += (cc_ > cm); dpos_h += (ch > cm)
mm, cc, hh = np.array(mm), np.array(cc), np.array(hh)
def summ(a): return {"mean": round(float(a.mean()), 3), "sd": round(float(a.std()), 3),
                     "p05": round(float(np.percentile(a, 5)), 3), "p95": round(float(np.percentile(a, 95)), 3)}
repeated = {"n_splits": KREP, "test_frac": round(TEST_FRAC, 3), "infra_test_n_mean": round(float(np.mean(infra_ns)), 1),
            "marginal_infra": summ(mm), "class_cond_infra": summ(cc), "ours_infra": summ(hh),
            "frac_classcond_gt_marginal": round(dpos_cc / KREP, 3),
            "frac_ours_gt_marginal": round(dpos_h / KREP, 3),
            "mean_delta_classcond_minus_marginal": round(float((cc - mm).mean()), 3),
            "mean_delta_ours_minus_marginal": round(float((hh - mm).mean()), 3)}

# ---------- (B) tau-sweep on the original split ----------
P_cal, y_cal, P_te, y_te = d["P_cal"], d["y_cal"], d["P_te"], d["y_te"]
taus = [0.0, 5, 10, 20, 50, 100, 200, 500, 1e9]
sweep = []
for tau in taus:
    _, tc, th = thresholds(P_cal, y_cal, tau=tau if tau > 0 else 1e-9)
    thr = tc if tau == 0 else th
    S = [np.where(P_te[i] >= thr)[0] for i in range(len(P_te))]
    inf_idx = np.where(y_te == INF)[0]
    infra = float(np.mean([INF in S[i] for i in inf_idx]))
    overall = float(np.mean([y_te[i] in S[i] for i in range(len(y_te))]))
    setsz = float(np.mean([len(s) for s in S]))
    lam_inf = (24 / (24 + tau)) if tau < 1e8 else 0.0
    sweep.append({"tau": ("inf" if tau >= 1e8 else (0 if tau == 0 else tau)),
                  "lambda_infra": round(lam_inf, 3), "infra_cov": round(infra, 3),
                  "overall_cov": round(overall, 3), "avg_set": round(setsz, 3)})

out = {"repeated_resplits": repeated, "tau_sweep": sweep,
       "note": ("Repeated resplits keep the trained model fixed and re-partition the out-of-train pool; "
                "they characterise sampling variability of the coverage estimate (not a new guarantee). "
                "tau=0 is class-conditional, tau=inf is concept-conditional; tau=50 is the reported operating point.")}
io.open(RES / "cr_robustness.json", "w", encoding="utf-8").write(json.dumps(out, indent=2))

# Pareto figure: infra coverage vs avg set size across tau
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.4, 4.4))
xs = [s["avg_set"] for s in sweep]; ys = [s["infra_cov"] for s in sweep]
ax.plot(xs, ys, "o-", color="#7C3AED")
for s in sweep:
    lbl = f"tau={s['tau']}"; ax.annotate(lbl, (s["avg_set"], s["infra_cov"]), fontsize=7,
                                          xytext=(4, -2), textcoords="offset points")
ax.axhline(0.90, ls="--", color="k", lw=1, label="target 0.90")
ax.set_xlabel("average set size"); ax.set_ylabel("infrastructure coverage (n=19)")
ax.set_title("Concept-smoothed frontier: tau trades set size for rare-class coverage\n(tau=0 class-cond, tau=inf concept-cond)")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "tau_pareto.png", dpi=220, bbox_inches="tight")

print("REPEATED-SPLIT STABILITY (%d splits, mean infra test n=%.1f):" % (KREP, repeated["infra_test_n_mean"]))
print("  marginal infra   :", repeated["marginal_infra"])
print("  class-cond infra :", repeated["class_cond_infra"])
print("  ours infra       :", repeated["ours_infra"])
print("  class-cond > marginal in %.1f%% of splits (mean Δ=%.3f)" % (100*repeated["frac_classcond_gt_marginal"], repeated["mean_delta_classcond_minus_marginal"]))
print("  ours       > marginal in %.1f%% of splits (mean Δ=%.3f)" % (100*repeated["frac_ours_gt_marginal"], repeated["mean_delta_ours_minus_marginal"]))
print("\nTAU SWEEP (original split):")
for s in sweep: print(f"  tau={str(s['tau']):>4} lam_inf={s['lambda_infra']:.3f}  infra={s['infra_cov']}  overall={s['overall_cov']}  set={s['avg_set']}")
print("saved cr_robustness.json + tau_pareto.png")
