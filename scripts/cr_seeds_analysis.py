"""Optional work 2 (analysis): multi-seed transformer results.
Answers: (a) how variable is transformer accuracy across seeds, (b) is the
transformer's advantage over the distilled student CONSISTENTLY significant under a
paired bootstrap, and (c) does the marginal-vs-class-conditional rare-class
separation hold across every seed. Reads seeds_raw.json -> cr_seeds.json + figure."""
import json, io
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import beta

RES = Path(__file__).resolve().parents[1] / "results"; FIG = Path(__file__).resolve().parents[1] / "figures"
R = json.load(open(RES / "seeds_raw.json", encoding="utf-8"))
y_te = np.array(R["y_te"]); K = 5
yp_stu = np.array(R["student"]["yp_test"]); f1_stu = R["student"]["test_macro_f1"]
DISP = {"bert-base-multilingual-cased": "mBERT", "xlm-roberta-base": "XLM-R"}

def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else float(beta.ppf(a/2, k, n-k+1)); hi = 1.0 if k == n else float(beta.ppf(1-a/2, k+1, n-k))
    return round(lo, 3), round(hi, 3)

def paired_f1(yp_a, yp_b, reps=2000, seed=0):
    """bootstrap of macro-F1(b) - macro-F1(a) on the same test items"""
    rng = np.random.RandomState(seed); n = len(y_te); dif = np.empty(reps)
    for i in range(reps):
        r = rng.randint(0, n, n)
        dif[i] = (f1_score(y_te[r], yp_b[r], average="macro", labels=list(range(K)), zero_division=0)
                  - f1_score(y_te[r], yp_a[r], average="macro", labels=list(range(K)), zero_division=0))
    lo, hi = float(np.percentile(dif, 2.5)), float(np.percentile(dif, 97.5))
    return {"delta": round(float(dif.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)],
            "significant": bool(lo > 0), "frac_positive": round(float((dif > 0).mean()), 4)}

out = {"student_macro_f1": round(f1_stu, 4), "seeds": R["seeds"], "models": {}}
for mdl, per in R["models"].items():
    disp = DISP.get(mdl, mdl); seeds = sorted(per.keys(), key=int)
    f1s = np.array([per[s]["test_macro_f1"] for s in seeds])
    rows = []
    for s in seeds:
        d = per[s]; yp = np.array(d["yp_test"])
        cm = d["conformal"]["marginal"]; cc = d["conformal"]["class_conditional"]
        rows.append({"seed": int(s), "macro_f1": round(d["test_macro_f1"], 4),
                     "acc": round(d["test_acc"], 4),
                     "paired_vs_student": paired_f1(yp_stu, yp, seed=int(s)),
                     "infra_marginal": f"{cm['infra_covered']}/{cm['infra_n']}",
                     "infra_marginal_cov": round(cm["infra_covered"]/cm["infra_n"], 3),
                     "infra_classcond": f"{cc['infra_covered']}/{cc['infra_n']}",
                     "infra_classcond_cov": round(cc["infra_covered"]/cc["infra_n"], 3),
                     "overall_marginal": cm["overall_cov"], "overall_classcond": cc["overall_cov"],
                     "set_marginal": cm["avg_set"], "set_classcond": cc["avg_set"]})
    nsig = sum(r["paired_vs_student"]["significant"] for r in rows)
    out["models"][disp] = {
        "macro_f1_mean": round(float(f1s.mean()), 4), "macro_f1_sd": round(float(f1s.std(ddof=1)), 4),
        "macro_f1_min": round(float(f1s.min()), 4), "macro_f1_max": round(float(f1s.max()), 4),
        "delta_vs_student_mean": round(float(f1s.mean() - f1_stu), 4),
        "seeds_with_significant_advantage": f"{nsig}/{len(rows)}",
        "infra_marginal_range": [min(r["infra_marginal_cov"] for r in rows), max(r["infra_marginal_cov"] for r in rows)],
        "infra_classcond_range": [min(r["infra_classcond_cov"] for r in rows), max(r["infra_classcond_cov"] for r in rows)],
        "per_seed": rows}

allm = [r["infra_marginal_cov"] for m in out["models"].values() for r in m["per_seed"]]
allc = [r["infra_classcond_cov"] for m in out["models"].values() for r in m["per_seed"]]
out["separation_across_all_seeds"] = {
    "marginal_max": max(allm), "classcond_min": min(allc), "n_runs": len(allm),
    "separated": bool(max(allm) < 0.90 <= min(allc))}
io.open(RES/"cr_seeds.json", "w", encoding="utf-8").write(json.dumps(out, indent=2))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7.6, 4.4))
for i, (disp, m) in enumerate(out["models"].items()):
    f1s = [r["macro_f1"] for r in m["per_seed"]]; xs = np.full(len(f1s), i) + np.linspace(-.09, .09, len(f1s))
    ax.scatter(xs, f1s, s=46, color=["#E8590C", "#4C6EF5"][i], zorder=3, label=f"{disp} (per seed)")
    ax.hlines(m["macro_f1_mean"], i-0.18, i+0.18, color="k", lw=2, zorder=4)
ax.axhline(f1_stu, ls="--", color="#2F9E44", lw=1.6, label=f"distilled student ({f1_stu:.3f})")
ax.set_xticks(range(len(out["models"]))); ax.set_xticklabels(list(out["models"].keys()))
ax.set_ylabel("test macro-F1"); ax.legend(fontsize=8)
ax.set_title("Transformer accuracy across training seeds vs the distilled student", fontsize=11)
fig.tight_layout(); fig.savefig(FIG/"seed_variance.png", dpi=220, bbox_inches="tight")

print("student macro-F1:", out["student_macro_f1"])
for disp, m in out["models"].items():
    print(f"\n{disp}: F1 {m['macro_f1_mean']}±{m['macro_f1_sd']} (min {m['macro_f1_min']}, max {m['macro_f1_max']}) "
          f"| Δ vs student {m['delta_vs_student_mean']:+.4f} | significant in {m['seeds_with_significant_advantage']} seeds")
    for r in m["per_seed"]:
        p = r["paired_vs_student"]
        print(f"   seed {r['seed']}: f1={r['macro_f1']} Δ={p['delta']:+.4f} CI{p['ci95']} "
              f"{'SIG' if p['significant'] else 'n.s.'} | infra {r['infra_marginal']}->{r['infra_classcond']}")
print("\nSeparation across all seeded runs:", out["separation_across_all_seeds"])
print("saved cr_seeds.json + seed_variance.png")
