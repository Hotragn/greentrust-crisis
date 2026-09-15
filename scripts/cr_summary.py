"""Camera-ready summary: transformer-conformal Clopper-Pearson CIs, cross-backbone
infrastructure-coverage table+figure, and a carefully-computed energy/CO2 table
(same-host TDP model; all arithmetic done in-code to avoid unit errors)."""
import json, io
from pathlib import Path
import numpy as np
from scipy.stats import beta

CR = Path(__file__).resolve().parents[1]
RES = CR / "results"; FIG = CR / "figures"
B = json.load(open(RES / "baselines.json", encoding="utf-8"))
CONF = json.load(open(RES / "cr_conformal.json", encoding="utf-8"))
try:
    DB = json.load(open(CR / "results" / "transformer_gpu.json", encoding="utf-8"))
except Exception:
    DB = None

def cp(k, n, a=0.05):
    if n == 0: return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(beta.ppf(a/2, k, n-k+1))
    hi = 1.0 if k == n else float(beta.ppf(1-a/2, k+1, n-k))
    return round(lo, 3), round(hi, 3)

# ---- cross-backbone infrastructure coverage (marginal -> class-conditional) ----
rows = []
# linear student from cr_conformal (per-class counts reconstructed from coverage*n)
li_m = CONF["marginal"]["per_class"]["infrastructure"]; li_c = CONF["class_conditional"]["per_class"]["infrastructure"]
n_infra = li_m["n"]
rows.append(("Distilled linear (ours)", round(li_m["coverage"], 3), li_m["ci95"], round(li_c["coverage"], 3), li_c["ci95"]))
if DB and "mondrian" in DB:
    dm = DB["mondrian"]["marginal"]["per_class"]["infrastructure"]; dc = DB["mondrian"]["class_conditional"]["per_class"]["infrastructure"]
    km = round(dm * n_infra) if isinstance(dm, float) else dm;
    # DB stored coverage floats
    covm = dm if isinstance(dm, float) else None
for name, key in [("DistilBERT", None)]:
    pass
def from_counts(pc):
    n = pc["n"]; k = pc["covered"]; return round(k/n, 3), list(cp(k, n)), k, n
for mdl, disp in [("bert-base-multilingual-cased", "mBERT (178M)"), ("xlm-roberta-base", "XLM-R (278M)")]:
    c = B["models"][mdl]["conformal"]
    m = c["marginal"]["per_class"]["infrastructure"]; cc = c["class_conditional"]["per_class"]["infrastructure"]
    covm, cim, km, nm = from_counts(m); covc, cic, kc, nc = from_counts(cc)
    rows.append((disp, covm, cim, covc, cic))
# DistilBERT from the GPU run (coverage floats; reconstruct counts on n=19)
if DB and "mondrian" in DB:
    dm = DB["mondrian"]["marginal"]["per_class"]["infrastructure"]
    dc = DB["mondrian"]["class_conditional"]["per_class"]["infrastructure"]
    km = round(dm*n_infra); kc = round(dc*n_infra)
    rows.insert(1, ("DistilBERT (135M)", round(dm,3), list(cp(km,n_infra)), round(dc,3), list(cp(kc,n_infra))))

cross = [{"backbone": r[0], "marginal_infra_cov": r[1], "marginal_ci95": r[2],
          "classcond_infra_cov": r[3], "classcond_ci95": r[4]} for r in rows]

# ---- energy / CO2 (same-host TDP model) ----
TDP_CPU = 65.0; TDP_GPU = 450.0; GRID = 0.4  # kg CO2 / kWh
lin_lat = B["linear_student_same_host"]["cpu_latency_ms_p50"]
def energy_block(name, lat_ms, train_s=None, f1=None):
    e_mJ = lat_ms * TDP_CPU
    kwh_day_1M = e_mJ/1000*1e6/3.6e6            # 1e6 msgs/day
    co2_yr_1M = kwh_day_1M*365*GRID
    d = {"cpu_latency_ms": round(lat_ms,4), "energy_mJ_per_msg": round(e_mJ,4),
         "kwh_per_day_at_1M": round(kwh_day_1M,5), "co2_kg_per_year_at_1M": round(co2_yr_1M,2)}
    if train_s is not None:
        d["train_seconds_gpu"] = train_s; d["train_energy_kwh"] = round(train_s*TDP_GPU/3.6e6,4)
        d["train_co2_kg"] = round(train_s*TDP_GPU/3.6e6*GRID,4)
    if f1 is not None: d["macro_f1"] = round(f1,4)
    return d
energy = {"assumptions": f"CPU TDP {TDP_CPU}W, GPU TDP {TDP_GPU}W, grid {GRID} kgCO2/kWh; same-host CPU latency; 1e6 msgs/day scenario. RAPL not exposed on host so energy is modelled, but latency is same-host so the RATIO is hardware-independent.",
          "distilled_linear_ours": energy_block("linear", lin_lat, f1=B["linear_student_same_host"]["test_macro_f1"]),
          "mBERT": energy_block("mBERT", B["models"]["bert-base-multilingual-cased"]["cpu_latency_ms_p50"], B["models"]["bert-base-multilingual-cased"]["train_seconds"], B["models"]["bert-base-multilingual-cased"]["test_macro_f1"]),
          "XLM-R": energy_block("XLM-R", B["models"]["xlm-roberta-base"]["cpu_latency_ms_p50"], B["models"]["xlm-roberta-base"]["train_seconds"], B["models"]["xlm-roberta-base"]["test_macro_f1"])}
energy["ratio_mBERT_over_linear"] = round(energy["mBERT"]["energy_mJ_per_msg"]/energy["distilled_linear_ours"]["energy_mJ_per_msg"], 0)

summary = {"cross_backbone_infrastructure": cross, "energy_co2": energy,
           "baseline_accuracy": {"distilled_linear_ours": round(B["linear_student_same_host"]["test_macro_f1"],4),
                                 "mBERT": round(B["models"]["bert-base-multilingual-cased"]["test_macro_f1"],4),
                                 "XLM-R": round(B["models"]["xlm-roberta-base"]["test_macro_f1"],4)}}
io.open(RES/"cr_summary.json","w",encoding="utf-8").write(json.dumps(summary,indent=2))

# ---- cross-backbone figure: infra coverage marginal vs class-cond ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
names=[r["backbone"] for r in cross]; x=np.arange(len(names)); w=0.38
fig,ax=plt.subplots(figsize=(8.5,4.4))
mcov=[r["marginal_infra_cov"] for r in cross]; ccov=[r["classcond_infra_cov"] for r in cross]
mlo=[mcov[i]-cross[i]["marginal_ci95"][0] for i in range(len(cross))]; mhi=[cross[i]["marginal_ci95"][1]-mcov[i] for i in range(len(cross))]
clo=[ccov[i]-cross[i]["classcond_ci95"][0] for i in range(len(cross))]; chi=[cross[i]["classcond_ci95"][1]-ccov[i] for i in range(len(cross))]
ax.bar(x-w/2,mcov,w,yerr=[mlo,mhi],capsize=3,label="marginal",color="#94A3B8",error_kw={"lw":1})
ax.bar(x+w/2,ccov,w,yerr=[clo,chi],capsize=3,label="class-conditional",color="#3AAFA9",error_kw={"lw":1})
ax.axhline(0.90,ls="--",color="k",lw=1,label="target 0.90")
ax.set_xticks(x); ax.set_xticklabels(names,rotation=15,ha="right",fontsize=8)
ax.set_ylabel("infrastructure coverage (95% CI, n=19)"); ax.set_ylim(0,1.1)
ax.set_title("Marginal conformal fails the rare class on every backbone; Mondrian fixes it")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG/"cross_backbone_infra.png",dpi=220,bbox_inches="tight")
print("saved cr_summary.json + cross_backbone_infra.png")
print("\nCross-backbone infrastructure coverage (marginal -> class-conditional):")
for r in cross: print(f"  {r['backbone']:26s} {r['marginal_infra_cov']} {r['marginal_ci95']} -> {r['classcond_infra_cov']} {r['classcond_ci95']}")
print("\nEnergy/CO2:")
for k in ["distilled_linear_ours","mBERT","XLM-R"]:
    e=energy[k]; print(f"  {k:22s} {e['energy_mJ_per_msg']} mJ/msg  f1={e.get('macro_f1')}  co2/yr@1M={e['co2_kg_per_year_at_1M']}kg  train_kwh={e.get('train_energy_kwh')}")
print("  energy ratio mBERT/linear:", energy["ratio_mBERT_over_linear"])
