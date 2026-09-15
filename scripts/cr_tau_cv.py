"""Optional work 2: choose tau by cross-validation on the CALIBRATION set only,
so the operating point is never tuned on test data.

Protocol: repeated stratified K-fold *within* the calibration set. For each fold we
fit the concept-smoothed thresholds on K-1 parts and evaluate on the held-out part.
Selection rule: pick the LARGEST tau (= smallest sets) whose mean held-out
infrastructure coverage still meets the 1-alpha target. Only after tau* is fixed do
we touch the test set, once, to report it. Writes cr_tau_cv.json + tau_cv.png."""
import json, io
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold

RES = Path(__file__).resolve().parents[1] / "results"; FIG = Path(__file__).resolve().parents[1] / "figures"
d = np.load(RES / "cache.npz", allow_pickle=True)
P_cal, y_cal, P_te, y_te = d["P_cal"], d["y_cal"], d["P_te"], d["y_te"]
K, A, INF, TARGET = 5, 0.10, 3, 0.90
CLASS = ["not_related", "weather_disaster", "aid_request", "infrastructure", "other_related"]
CONCEPT = {0: "NotRelated", 1: "HazardEvent", 2: "ActionableNeed", 3: "ActionableNeed", 4: "RelatedOther"}
TAUS = [0, 2, 5, 10, 15, 20, 30, 50, 100, 200, 500, 1e9]

def qlevel(n): return min(np.ceil((1 - A) * (n + 1)) / n, 1.0) if n > 0 else 1.0
def qhat(s):
    s = np.asarray(s); return float(np.quantile(s, qlevel(len(s)), method="higher")) if len(s) else 1.0

def fit_thresholds(Pc, yc, tau):
    """Concept-smoothed thresholds. tau=0 -> class-conditional; tau->inf -> concept-conditional."""
    cq = {}
    for c in set(CONCEPT.values()):
        ks = [k for k in range(K) if CONCEPT[k] == c]; m = np.isin(yc, ks)
        cq[c] = qhat(1 - Pc[m, yc[m]]) if m.any() else 1.0
    thr = np.ones(K)
    for k in range(K):
        m = yc == k; nk = int(m.sum())
        qk = qhat(1 - Pc[m, k]) if m.any() else 1.0
        lk = 1.0 if tau == 0 else nk / (nk + tau)
        thr[k] = 1 - (lk * qk + (1 - lk) * cq[CONCEPT[k]])
    return thr

def evaluate(thr, Pv, yv):
    S = [np.where(Pv[i] >= thr)[0] for i in range(len(Pv))]
    idx = np.where(yv == INF)[0]
    infra = float(np.mean([INF in S[i] for i in idx])) if len(idx) else np.nan
    overall = float(np.mean([yv[i] in S[i] for i in range(len(yv))]))
    return infra, overall, float(np.mean([len(s) for s in S]))

# ---------- cross-validation inside the calibration set ----------
REPEATS, FOLDS = 20, 5
acc = {t: {"infra": [], "overall": [], "set": []} for t in TAUS}
for rep in range(REPEATS):
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=rep)
    for tr_i, va_i in skf.split(P_cal, y_cal):
        Pt_, yt_, Pv_, yv_ = P_cal[tr_i], y_cal[tr_i], P_cal[va_i], y_cal[va_i]
        for t in TAUS:
            inf, ov, sz = evaluate(fit_thresholds(Pt_, yt_, t), Pv_, yv_)
            if not np.isnan(inf): acc[t]["infra"].append(inf)
            acc[t]["overall"].append(ov); acc[t]["set"].append(sz)

cv = []
for t in TAUS:
    cv.append({"tau": ("inf" if t >= 1e8 else t),
               "cv_infra_cov": round(float(np.mean(acc[t]["infra"])), 4),
               "cv_infra_sd": round(float(np.std(acc[t]["infra"])), 4),
               "cv_overall_cov": round(float(np.mean(acc[t]["overall"])), 4),
               "cv_avg_set": round(float(np.mean(acc[t]["set"])), 4)})

# Two PRE-SPECIFIED selection rules, both reported (no post-hoc rule picking).
# (A) aggressive: largest tau whose mean CV infra coverage >= target.
# (B) one-standard-error rule (standard CV convention, Breiman et al.): largest tau
#     whose CV infra coverage is at least one SE ABOVE target -- i.e. require a
#     margin so the choice is robust to CV noise.
NFOLD = REPEATS * FOLDS
for c in cv:
    c["cv_infra_se"] = round(float(c["cv_infra_sd"] / np.sqrt(NFOLD)), 4)

def pick(pred, label):
    el = [c for c in cv if pred(c)]
    if not el: return None, label + " (none eligible)"
    return max(el, key=lambda c: (float("inf") if c["tau"] == "inf" else c["tau"])), label

best_A, rule_A = pick(lambda c: c["cv_infra_cov"] >= TARGET,
                      f"(A) aggressive: largest tau with CV infra coverage >= {TARGET}")
best_B, rule_B = pick(lambda c: c["cv_infra_cov"] - c["cv_infra_se"] >= TARGET,
                      f"(B) one-SE rule: largest tau with CV infra coverage - 1 SE >= {TARGET}")
if best_B is None: best_B, rule_B = best_A, rule_B + " -> fell back to (A)"
best, rule = best_B, rule_B  # recommend the conservative rule
tau_star = 0 if best["tau"] == 0 else (1e9 if best["tau"] == "inf" else best["tau"])

# ---------- ONE look at test, after tau* is fixed ----------
thr_star = fit_thresholds(P_cal, y_cal, tau_star)
inf_t, ov_t, sz_t = evaluate(thr_star, P_te, y_te)
thr_cc = fit_thresholds(P_cal, y_cal, 0)
inf_c, ov_c, sz_c = evaluate(thr_cc, P_te, y_te)
# paired set-size test vs class-conditional
sz_star_v = np.array([int((P_te[i] >= thr_star).sum()) for i in range(len(P_te))])
sz_cc_v = np.array([int((P_te[i] >= thr_cc).sum()) for i in range(len(P_te))])
rng = np.random.RandomState(0); n = len(sz_cc_v); dif = []
for _ in range(10000):
    r = rng.randint(0, n, n); dif.append(sz_cc_v[r].mean() - sz_star_v[r].mean())
dif = np.array(dif)

def test_at(t):
    th = fit_thresholds(P_cal, y_cal, 0 if t == 0 else (1e9 if t == "inf" else t))
    i, o, s = evaluate(th, P_te, y_te); return {"tau": t, "infra_cov": round(i, 3), "overall_cov": round(o, 3), "avg_set": round(s, 3)}

out = {"protocol": (f"{REPEATS}x{FOLDS}-fold stratified CV inside the calibration set "
                    f"({len(y_cal)} msgs); thresholds fit on K-1 folds, evaluated on the held-out fold. "
                    "Test set untouched until tau* was fixed. TWO pre-specified rules reported."),
       "rule_A_aggressive": {"rule": rule_A, "tau": best_A["tau"], "test": test_at(best_A["tau"])},
       "rule_B_one_se_recommended": {"rule": rule_B, "tau": best_B["tau"], "test": test_at(best_B["tau"])},
       "selection_rule": rule, "tau_star": best["tau"], "cv_curve": cv,
       "test_at_tau_star": {"infra_cov": round(inf_t, 3), "overall_cov": round(ov_t, 3), "avg_set": round(sz_t, 3)},
       "test_class_conditional": {"infra_cov": round(inf_c, 3), "overall_cov": round(ov_c, 3), "avg_set": round(sz_c, 3)},
       "setsize_vs_classcond": {"delta": round(float(sz_cc_v.mean() - sz_star_v.mean()), 3),
                                "ci95": [round(float(np.percentile(dif, 2.5)), 3), round(float(np.percentile(dif, 97.5)), 3)],
                                "frac_positive": round(float((dif > 0).mean()), 4),
                                "ours_larger_on": int((sz_cc_v - sz_star_v < 0).sum())}}
io.open(RES / "cr_tau_cv.json", "w", encoding="utf-8").write(json.dumps(out, indent=2))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
xs = [i for i, _ in enumerate(cv)]; lbl = [str(c["tau"]) for c in cv]
fig, ax1 = plt.subplots(figsize=(7.4, 4.4))
ax1.errorbar(xs, [c["cv_infra_cov"] for c in cv], yerr=[c["cv_infra_sd"] for c in cv],
             fmt="o-", color="#7C3AED", capsize=3, label="CV infrastructure coverage")
ax1.axhline(TARGET, ls="--", color="k", lw=1, label="target 0.90")
ax1.axvline(xs[cv.index(best)], ls=":", color="#DC2626", lw=1.5)
ax1.text(xs[cv.index(best)] + 0.12, 0.33, f"selected $\\tau$={best['tau']}", color="#B91C1C", fontsize=9)
ax1.set_xticks(xs); ax1.set_xticklabels(lbl); ax1.set_xlabel(r"$\tau$")
ax1.set_ylabel("held-out coverage (calibration CV)"); ax1.set_ylim(0, 1.05)
ax2 = ax1.twinx(); ax2.plot(xs, [c["cv_avg_set"] for c in cv], "s--", color="#0F766E", label="CV avg set size")
ax2.set_ylabel("avg set size", color="#0F766E")
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="center left")
ax1.set_title(r"Selecting $\tau$ by cross-validation on calibration only (test untouched)", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "tau_cv.png", dpi=220, bbox_inches="tight")

print(out["protocol"]); print("rule:", rule)
print(f"{'tau':>6} {'CV infra':>9} {'sd':>7} {'CV overall':>11} {'CV set':>8}")
for c in cv: print(f"{str(c['tau']):>6} {c['cv_infra_cov']:>9.4f} {c['cv_infra_sd']:>7.4f} {c['cv_overall_cov']:>11.4f} {c['cv_avg_set']:>8.4f}")
print("\nSELECTED tau* =", best["tau"])
print(f"  TEST @tau*: infra={inf_t:.3f} overall={ov_t:.3f} set={sz_t:.3f}")
print(f"  TEST class-cond: infra={inf_c:.3f} overall={ov_c:.3f} set={sz_c:.3f}")
print(f"  set-size delta (cc - ours) = {out['setsize_vs_classcond']['delta']} CI{out['setsize_vs_classcond']['ci95']} "
      f"ours larger on {out['setsize_vs_classcond']['ours_larger_on']} msgs")
print("saved cr_tau_cv.json + tau_cv.png")
