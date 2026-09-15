"""Same-host inference latency for the three linear models.

The camera-ready energy run (cr_beam_baselines.py) only timed the distilled
student and the two transformers. The teacher and hard-label student rows of the
energy table were carried over from the pre-submission run, and they do not
follow the 65 W TDP model the table's own caption states. This re-times all
three on one host with the protocol used for the other rows: p50 of 300
single-row predict_proba calls on an already-vectorised message, after five
warm-up calls.

Writes ../results/cr_latency_linear.json
"""
import sys, io, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DEFAULT_CONFIG as CFG
from src.data import build_unified_dataframe, stratified_split
from src.features import (build_teacher_vectorizer, build_student_vectorizer,
                          select_features_by_mi)
from src.models import TeacherModel, StudentModel

OUT = Path(__file__).resolve().parents[1] / "results"
TDP_CPU, GRID = 65.0, 0.4
REPS, WARM = 300, 5


def p50_ms(fn, x):
    for _ in range(WARM):
        fn(x)
    xs = []
    for _ in range(REPS):
        t = time.perf_counter()
        fn(x)
        xs.append((time.perf_counter() - t) * 1000)
    return float(np.percentile(xs, 50))


print("[1] data + splits")
df = build_unified_dataframe()
tr, cal, te = stratified_split(df)
y_tr = tr["y_crisis_type"].values

print("[2] teacher")
tv = build_teacher_vectorizer(CFG.teacher_max_features, CFG.teacher_ngram_range,
                              CFG.teacher_min_df)
Xtr_t = tv.fit_transform(tr["text"].tolist())
teacher = TeacherModel(C=CFG.teacher_C, random_state=42).fit(Xtr_t, y_tr)
tlog_tr = teacher.decision_function(Xtr_t)
Xte_t = tv.transform(te["text"].tolist())

print("[3] student features")
sv = build_student_vectorizer(CFG.student_max_features, CFG.student_ngram_range,
                              CFG.student_min_df)
Xtr_sf = sv.fit_transform(tr["text"].tolist())
top = select_features_by_mi(Xtr_sf, y_tr, top_k=CFG.student_max_features)
Xtr_s = Xtr_sf[:, top]
Xte_s = sv.transform(te["text"].tolist())[:, top]

print("[4] students")
s_hard = StudentModel(C=CFG.student_C, temperature=CFG.distill_temperature,
                      alpha=0.0, epochs=CFG.distill_epochs, random_state=42)
s_hard.fit_hard(Xtr_s, y_tr)
s_dist = StudentModel(C=CFG.student_C, temperature=CFG.distill_temperature,
                      alpha=CFG.distill_alpha, epochs=CFG.distill_epochs,
                      random_state=42)
s_dist.fit_distilled(Xtr_s, y_tr, tlog_tr)

print("[5] timing")
rows = {
    "teacher": (teacher.predict_proba, Xte_t[:1], int(Xtr_t.shape[1])),
    "student_hard": (s_hard.predict_proba, Xte_s[:1], int(Xtr_s.shape[1])),
    "student_distilled": (s_dist.predict_proba, Xte_s[:1], int(Xtr_s.shape[1])),
}
res = {"protocol": (f"p50 of {REPS} single-row predict_proba calls after {WARM} "
                    f"warm-ups, one CPU core, same host as the transformer "
                    f"timings; energy = {TDP_CPU} W x latency"),
       "tdp_w": TDP_CPU, "grid_kg_co2_per_kwh": GRID, "models": {}}
for name, (fn, x, nfeat) in rows.items():
    lat = p50_ms(fn, x)
    e_mJ = lat * TDP_CPU
    kwh_day_1M = e_mJ / 1000 * 1e6 / 3.6e6
    res["models"][name] = {
        "n_features": nfeat,
        "cpu_latency_ms_p50": round(lat, 4),
        "energy_mJ_per_msg": round(e_mJ, 4),
        "co2_kg_per_year_at_1M": round(kwh_day_1M * 365 * GRID, 3),
    }
    print(f"   {name:20s} {nfeat:6d} feat  {lat:9.4f} ms  {e_mJ:9.4f} mJ")

io.open(OUT / "cr_latency_linear.json", "w", encoding="utf-8").write(
    json.dumps(res, indent=2))
print("wrote", OUT / "cr_latency_linear.json")
