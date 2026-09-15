"""Camera-ready figures, authored at the size they are printed at.

The earlier set was drawn 5.0-5.8 in wide and then placed at 0.66-0.72
\\textwidth (3.2-3.5 in), so everything was scaled down by 0.27-0.66 and the
lettering landed between 2.2 pt and 5.3 pt. Springer's floor is 6 pt.

Every figure here is drawn at 4.82 in wide, which is exactly llncs
\\textwidth (347.12 pt), and every figure is placed at width=\\textwidth. Scale
is therefore 1.0 and the 7-8 pt lettering prints at 7-8 pt.

Figure 1 is also redrawn from scratch. The old architecture.pdf came from the
pre-submission generator and still showed 50K teacher features, T=4 with
alpha=0.7, and five languages including Urdu. The paper says 28,356 features,
T=6, alpha=0.3, and four languages.

Writes into ../figures_camera/ as vector PDF.
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

TW_IN = 347.12354 / 72.0          # llncs \textwidth in inches

matplotlib.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "pdf.fonttype": 42, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})

RES = Path(__file__).resolve().parents[1] / "results"
OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)

K, A, TARGET = 5, 0.10, 0.90
CLASS_PLAIN = ["not_related", "weather_disaster", "aid_request",
               "infrastructure", "other_related"]
CONCEPT = {0: "N", 1: "H", 2: "AN", 3: "AN", 4: "R"}


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.012)
    plt.close(fig)


# =====================================================================
# Figure 1: pipeline architecture, redrawn to match the paper
# =====================================================================
# data units are 0.1 in, so the canvas is 48.2 x 20.0 units
fig, ax = plt.subplots(figsize=(TW_IN, 1.89))
ax.set_xlim(0, 48.2)
ax.set_ylim(0, 20.0)
ax.axis("off")
ax.set_position([0, 0, 1, 1])

FS = 7.0


def box(x, y, w, h, text, fill="white", dashed=False, bold=False, fs=FS):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.5",
        linewidth=1.0 if dashed else 0.8, edgecolor="black", facecolor=fill,
        linestyle=(0, (2.4, 1.6)) if dashed else "-", mutation_aspect=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.18)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7,
        linewidth=0.8, color="black", shrinkA=0, shrinkB=0))


W, GAP, H = 14.73, 2.0, 5.6
TOP_Y, BOT_Y = 13.4, 1.6
cols = [0.0, W + GAP, 2 * (W + GAP)]

box(cols[0], TOP_Y, W, H, "Multilingual crisis\nmessages (en, ht, fr, es)")
box(cols[1], TOP_Y, W, H, "TF-IDF features\n(1–2 grams)")
box(cols[2], TOP_Y, W, H, "Teacher: logistic\nregression,\n28,356 features")
arrow(cols[0] + W, TOP_Y + H / 2, cols[1], TOP_Y + H / 2)
arrow(cols[1] + W, TOP_Y + H / 2, cols[2], TOP_Y + H / 2)

box(cols[0], BOT_Y, W, H, "Student: logistic\nregression, top 5,000\nmutual-information\nfeatures")
box(cols[1], BOT_Y, W, H, "Ontology-grounded,\ngroup-conditional\nconformal layer\n(this work)",
    fill="0.93", dashed=True)
box(cols[2], BOT_Y, W, H, "Prediction set,\nnot a single label")
arrow(cols[0] + W, BOT_Y + H / 2, cols[1], BOT_Y + H / 2)
arrow(cols[1] + W, BOT_Y + H / 2, cols[2], BOT_Y + H / 2)

# distillation elbow: teacher bottom -> across -> student top
tx, sx, my = cols[2] + W / 2, cols[0] + W / 2, 10.05
ax.plot([tx, tx, sx], [TOP_Y, my, my], color="black", lw=0.8, solid_joinstyle="miter")
arrow(sx, my, sx, BOT_Y + H)
ax.text((tx + sx) / 2, my + 0.7, "soft labels, $T{=}6$, $\\alpha{=}0.3$",
        ha="center", va="bottom", fontsize=FS, style="italic")
ax.text(47.9, 0.05, "every stage runs on one CPU core",
        ha="right", va="bottom", fontsize=FS, style="italic")
save(fig, "architecture.pdf")


# =====================================================================
# Figure 2: concept hierarchy
# =====================================================================
fig, ax = plt.subplots(figsize=(TW_IN, 1.39))
ax.set_xlim(-0.3, 48.5)
ax.set_ylim(-0.5, 15.2)
ax.axis("off")
ax.set_position([0, 0, 1, 1])

BH = 4.0
LEAF_Y, CONC_Y, ROOT_Y = 0.3, 5.5, 10.7


def node(cx, y, w, text, fill="white", bold=False):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, y), w, BH, boxstyle="round,pad=0.0,rounding_size=0.5",
        linewidth=0.8, edgecolor="black", facecolor=fill))
    ax.text(cx, y + BH / 2, text, ha="center", va="center", fontsize=FS,
            fontweight="bold" if bold else "normal")


def link(x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], color="black", lw=0.7, zorder=0)


# leaf row: widths scaled to the label, packed left to right with 0.8 gaps
leaf_names = ["not_related", "weather_disaster", "aid_request",
              "infrastructure", "other_related"]
gap = 0.8
usable = 48.2 - gap * (len(leaf_names) - 1)
per_char = usable / sum(len(n) for n in leaf_names)
leaf_cx, xcur = {}, 0.0
for n in leaf_names:
    w = len(n) * per_char
    leaf_cx[n] = (xcur + w / 2, w)
    xcur += w + gap
for n in leaf_names:
    cx, w = leaf_cx[n]
    node(cx, LEAF_Y, w, n)

ROOT = 24.1
concepts = [("NotRelated", 8.2, ["not_related"]),
            ("HazardEvent", 8.8, ["weather_disaster"]),
            ("ActionableNeed", 10.7, ["aid_request", "infrastructure"]),
            ("RelatedOther", 9.4, ["other_related"])]
for name, w, kids in concepts:
    cx = sum(leaf_cx[k][0] for k in kids) / len(kids)
    cx = min(max(cx, w / 2), 48.2 - w / 2)
    node(cx, CONC_Y, w, name, fill="0.94")
    link(ROOT, ROOT_Y, cx, CONC_Y + BH)
    for k in kids:
        link(cx, CONC_Y, leaf_cx[k][0], LEAF_Y + BH)
node(ROOT, ROOT_Y, 11.6, "Crisis message", fill="0.86", bold=True)
save(fig, "kg_hierarchy.pdf")


# =====================================================================
# conformal machinery, shared by figures 3 and 4
# =====================================================================
def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def qlevel(n):
    return min(np.ceil((1 - A) * (n + 1)) / n, 1.0) if n > 0 else 1.0


def qhat(s):
    s = np.asarray(s)
    return float(np.quantile(s, qlevel(len(s)), method="higher")) if len(s) else 1.0


C = np.load(RES / "cache.npz", allow_pickle=True)
P_cal, y_cal, P_te, y_te = C["P_cal"], C["y_cal"], C["P_te"], C["y_te"]


def marg():
    return np.full(K, 1 - qhat(1 - P_cal[np.arange(len(y_cal)), y_cal]))


def classc():
    t = np.ones(K)
    for k in range(K):
        m = y_cal == k
        t[k] = 1 - qhat(1 - P_cal[m, k]) if m.any() else 1.0
    return t


def conceptq():
    cq = {}
    for c in set(CONCEPT.values()):
        ks = [k for k in range(K) if CONCEPT[k] == c]
        m = np.isin(y_cal, ks)
        cq[c] = qhat(1 - P_cal[m, y_cal[m]]) if m.any() else 1.0
    return cq


def hier(tau):
    cq = conceptq()
    t = np.ones(K)
    for k in range(K):
        m = y_cal == k
        nk = int(m.sum())
        qk = qhat(1 - P_cal[m, k]) if m.any() else 1.0
        lk = nk / (nk + tau)
        t[k] = 1 - (lk * qk + (1 - lk) * cq[CONCEPT[k]])
    return t


def percls(thr):
    S = [set(np.where(P_te[i] >= thr)[0].tolist()) for i in range(len(P_te))]
    out = []
    for k in range(K):
        idx = np.where(y_te == k)[0]
        kk = int(sum(k in S[i] for i in idx))
        lo, hi = cp(kk, len(idx))
        out.append((kk / len(idx), lo, hi, len(idx)))
    return out, float(np.mean([len(s) for s in S]))


# =====================================================================
# Figure 3: per-class coverage
# =====================================================================
pm, szm = percls(marg())
pc, szc = percls(classc())
ph, szh = percls(hier(10.0))
x = np.arange(K)
w = 0.26
fig, ax = plt.subplots(figsize=(TW_IN, 1.84))
specs = [(pm, "white", "//////", f"marginal (mean set {szm:.2f})"),
         (pc, "0.62", "", f"class-conditional ({szc:.2f})"),
         (ph, "0.30", "xxxxxx", f"concept-smoothed, ours ({szh:.2f})")]
for j, (dat, fc, hatch, lbl) in enumerate(specs):
    cov = [d[0] for d in dat]
    lo = [max(0, cov[i] - dat[i][1]) for i in range(K)]
    hi = [max(0, dat[i][2] - cov[i]) for i in range(K)]
    ax.bar(x + (j - 1) * w, cov, w, yerr=[lo, hi], capsize=1.8, label=lbl,
           facecolor=fc, edgecolor="black", linewidth=0.7, hatch=hatch,
           error_kw={"lw": 0.7, "ecolor": "black"})
ax.axhline(TARGET, ls="--", color="black", lw=0.9)
ax.set_xticks(x)
SHOW = {"weather_disaster": "weather_\ndisaster"}   # too wide for one line at 7 pt
ax.set_xticklabels([f"{SHOW.get(c, c)}\n(n={pm[i][3]})"
                    for i, c in enumerate(CLASS_PLAIN)])
ax.set_ylabel("coverage (95% CI)")
ax.set_ylim(0, 1.12)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
          handlelength=1.5, handletextpad=0.5, columnspacing=1.2, borderpad=0.2)
fig.tight_layout(pad=0.15)
save(fig, "coverage_perclass.pdf")


# =====================================================================
# Figure 4: tau frontier
# =====================================================================
R = json.load(open(RES / "cr_robustness.json", encoding="utf-8"))
sw = R["tau_sweep"]
fig, ax = plt.subplots(figsize=(TW_IN, 1.74))
xs = [s["avg_set"] for s in sw]
ys = [s["infra_cov"] for s in sw]
ax.plot(xs, ys, "-", color="black", lw=0.9, zorder=2)
ax.scatter(xs, ys, s=22, marker="o", facecolor="white", edgecolor="black",
           linewidth=0.8, zorder=3)
# 200, 500 and infinity land on the same point, so they share one label
LAB = {0: (r"$\tau=0$", (0, -9), "center"), 5: (r"$5$", (0, -9), "center"),
       10: (r"$10$", (0, -9), "center"), 20: (r"$20$", (4, -2), "left"),
       50: (r"$50$", (4, -2), "left"), 100: (r"$100$", (4, -2), "left"),
       200: (r"$\tau=200,\,500,\,\infty$", (-3, -9), "left")}
for s in sw:
    if s["tau"] not in LAB:
        continue
    lab, off, ha = LAB[s["tau"]]
    ax.annotate(lab, (s["avg_set"], s["infra_cov"]), fontsize=7,
                xytext=off, textcoords="offset points", ha=ha)
ax.axhline(TARGET, ls="--", color="black", lw=0.9)
ax.set_xlim(2.545, 3.062)
ax.set_ylim(0.40, 0.96)
ax.text(2.552, TARGET - 0.018, "target 0.90", fontsize=7, ha="left", va="top")
ax.set_xlabel("mean prediction-set size")
ax.set_ylabel("infrastructure coverage\n($n=19$)")
fig.tight_layout(pad=0.15)
save(fig, "tau_frontier.pdf")


# =====================================================================
# Figure 5: distillation temperature
# =====================================================================
T = json.load(open(RES / "cr_temperature.json", encoding="utf-8"))["sweep"]
Ts = [r["T"] for r in T]
fig, ax1 = plt.subplots(figsize=(TW_IN, 1.69))
ax1.plot(Ts, [r["aps_set"] for r in T], "-o", color="black", lw=0.9, ms=3.2,
         mfc="white", mec="black", label="APS set size")
ax1.plot(Ts, [r["score_set"] for r in T], "--s", color="black", lw=0.9, ms=3.2,
         mfc="black", mec="black", label="score set size")
ax1.set_xlabel("distillation temperature $T$")
ax1.set_ylabel("mean set size")
ax1.set_xticks(Ts)
ax2 = ax1.twinx()
ax2.plot(Ts, [r["macro_f1"] for r in T], ":^", color="0.45", lw=0.9, ms=3.2,
         mfc="0.45", mec="0.45", label="macro-F1")
ax2.set_ylabel("macro-F1", color="0.35")
ax2.tick_params(axis="y", colors="0.35")
ax2.set_yticks([0.455, 0.460, 0.465])
ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="center right", frameon=True, framealpha=1,
           edgecolor="black", handlelength=2.0, borderpad=0.35,
           labelspacing=0.35).get_frame().set_linewidth(0.6)
fig.tight_layout(pad=0.15)
save(fig, "temperature.pdf")


# =====================================================================
# Figure 6: accuracy against energy
# =====================================================================
pts = [("teacher (host A)", 77.4, 0.409, "o", "white"),
       ("student, hard labels (host A)", 1.85, 0.454, "^", "0.55"),
       ("student, distilled", 0.845, 0.467, "s", "black"),
       ("mBERT", 155182.0, 0.485, "D", "white"),
       ("XLM-R", 148310.0, 0.485, "v", "0.55")]
fig, ax = plt.subplots(figsize=(TW_IN, 1.69))
for name, e, f, mk, fc in pts:
    ax.scatter(e, f, s=32, marker=mk, facecolor=fc, edgecolor="black",
               linewidth=0.8, zorder=3, label=name)
ax.set_xscale("log")
# plain decade labels: matplotlib's default $10^{n}$ renders the exponent at 70%
# of the tick size, which would put it under Springer's 6 pt floor
ax.set_xticks([1, 10, 100, 1000, 10000, 100000])
ax.set_xticklabels(["1", "10", "100", "1,000", "10,000", "100,000"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("energy per message (mJ, log scale, TDP model)")
ax.set_ylabel("macro-F1")
ax.grid(True, which="major", axis="both", color="0.85", lw=0.5, zorder=0)
ax.legend(loc="lower right", frameon=True, framealpha=1, edgecolor="black",
          handletextpad=0.3, borderpad=0.35,
          labelspacing=0.3).get_frame().set_linewidth(0.6)
ax.set_ylim(0.395, 0.50)
fig.tight_layout(pad=0.15)
save(fig, "accuracy_energy.pdf")


# =====================================================================
# Figure 7: 14-run separation
# =====================================================================
V = json.load(open(RES / "cr_seed_variability.json", encoding="utf-8"))
runs = V["runs"]
ARCH = ["Distilled linear", "DistilBERT", "mBERT", "XLM-R"]
fig, ax = plt.subplots(figsize=(TW_IN, 1.79))
marg_v = np.array([r["marginal_cov"] for r in runs])
cc_v = np.array([r["classcond_cov"] for r in runs])
for ai, a in enumerate(ARCH):
    idx = [i for i, r in enumerate(runs) if r["architecture"] == a]
    jm = np.linspace(-0.11, 0.11, len(idx)) if len(idx) > 1 else np.array([0.0])
    ax.scatter(np.full(len(idx), ai - 0.18) + jm, marg_v[idx], s=22, marker="o",
               facecolor="white", edgecolor="black", linewidth=0.8, zorder=3,
               label="marginal" if ai == 0 else None)
    ax.scatter(np.full(len(idx), ai + 0.18) + jm, cc_v[idx], s=22, marker="s",
               facecolor="black", edgecolor="black", linewidth=0.8, zorder=3,
               label="class-conditional" if ai == 0 else None)
ax.axhspan(marg_v.max(), cc_v.min(), facecolor="0.88", edgecolor="none", zorder=1)
ax.axhline(TARGET, ls="--", color="black", lw=0.9, zorder=2)
ax.text(-0.50, TARGET + 0.025, "target 0.90", fontsize=7, ha="left")
ax.text(3.46, (marg_v.max() + cc_v.min()) / 2, "no overlap", fontsize=7,
        ha="right", va="center")
ax.set_xticks(range(len(ARCH)))
ax.set_xticklabels(ARCH)
ax.set_xlim(-0.55, 3.55)
ax.set_ylabel("infrastructure coverage\n($n=19$)")
ax.set_ylim(-0.06, 1.14)
ax.legend(loc="lower left", frameon=True, framealpha=1, edgecolor="black",
          handletextpad=0.3, borderpad=0.35,
          labelspacing=0.3).get_frame().set_linewidth(0.6)
fig.tight_layout(pad=0.15)
save(fig, "separation_14runs.pdf")


print("wrote camera figures (vector PDF, authored at printed size):")
for p in sorted(OUT.glob("*.pdf")):
    print(f"   {p.name:26s} {p.stat().st_size/1024:6.1f} kB")
