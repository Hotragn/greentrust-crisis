"""Camera-ready part 1: reproduce base pipeline, cache per-example probabilities
(calib+test) for teacher/hard/distilled, and report base metrics with bootstrap
95% CIs on macro-F1. Deterministic (seed 42). Reuses the existing src pipeline."""
import sys, json, io
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DEFAULT_CONFIG as CFG
from src.data import build_unified_dataframe, stratified_split
from src.features import build_teacher_vectorizer, build_student_vectorizer, select_features_by_mi
from src.models import TeacherModel, StudentModel

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(parents=True, exist_ok=True)
K = 5
CLASS_NAMES = ["not_related","weather_disaster","aid_request","infrastructure","other_related"]

def boot_f1(y, yp, B=2000, seed=0):
    rng = np.random.RandomState(seed); n=len(y); v=np.empty(B)
    for b in range(B):
        idx = rng.randint(0,n,n)
        v[b] = f1_score(y[idx], yp[idx], average="macro", labels=list(range(K)), zero_division=0)
    return float(np.percentile(v,2.5)), float(np.percentile(v,97.5))

print("[1] data + splits"); df = build_unified_dataframe(); tr,cal,te = stratified_split(df)
y_tr,y_cal,y_te = tr["y_crisis_type"].values, cal["y_crisis_type"].values, te["y_crisis_type"].values
lang_cal, lang_te = cal["lang"].values, te["lang"].values
print("   train/calib/test =",len(tr),len(cal),len(te))
# per-class calibration counts (key for the small-sample story)
calib_counts = {CLASS_NAMES[k]: int((y_cal==k).sum()) for k in range(K)}
test_counts  = {CLASS_NAMES[k]: int((y_te==k).sum()) for k in range(K)}
print("   calib per-class:",calib_counts); print("   test per-class:",test_counts)

print("[2] teacher"); tv=build_teacher_vectorizer(CFG.teacher_max_features,CFG.teacher_ngram_range,CFG.teacher_min_df)
Xtr_t=tv.fit_transform(tr["text"].tolist()); teacher=TeacherModel(C=CFG.teacher_C,random_state=42).fit(Xtr_t,y_tr)
tlog_tr=teacher.decision_function(Xtr_t); Xte_t=tv.transform(te["text"].tolist())
P_te_teacher=teacher.predict_proba(Xte_t)

print("[3] student features (MI 5k)")
sv=build_student_vectorizer(CFG.student_max_features,CFG.student_ngram_range,CFG.student_min_df)
Xtr_sf=sv.fit_transform(tr["text"].tolist()); top=select_features_by_mi(Xtr_sf,y_tr,top_k=CFG.student_max_features)
Xtr_s=Xtr_sf[:,top]; Xcal_s=sv.transform(cal["text"].tolist())[:,top]; Xte_s=sv.transform(te["text"].tolist())[:,top]

print("[4] students")
s_hard=StudentModel(C=CFG.student_C,temperature=CFG.distill_temperature,alpha=0.0,epochs=CFG.distill_epochs,random_state=42); s_hard.fit_hard(Xtr_s,y_tr)
s_dist=StudentModel(C=CFG.student_C,temperature=CFG.distill_temperature,alpha=CFG.distill_alpha,epochs=CFG.distill_epochs,random_state=42); s_dist.fit_distilled(Xtr_s,y_tr,tlog_tr)
P_cal=s_dist.predict_proba(Xcal_s); P_te=s_dist.predict_proba(Xte_s)
yp_dist=P_te.argmax(1); yp_hard=s_hard.predict(Xte_s); yp_teacher=P_te_teacher.argmax(1)

np.savez(OUT/"cache.npz", P_cal=P_cal, y_cal=y_cal, lang_cal=lang_cal.astype(str),
         P_te=P_te, y_te=y_te, lang_te=lang_te.astype(str),
         P_te_teacher=P_te_teacher, yp_hard=yp_hard)
def rep(name,yp):
    lo,hi=boot_f1(y_te,yp)
    return {"acc":float(accuracy_score(y_te,yp)),"macro_f1":float(f1_score(y_te,yp,average="macro")),
            "macro_f1_ci95":[round(lo,4),round(hi,4)]}
base={"splits":{"train":len(tr),"calib":len(cal),"test":len(te)},
      "calib_per_class":calib_counts,"test_per_class":test_counts,
      "teacher":rep("teacher",yp_teacher),"student_hard":rep("hard",yp_hard),"student_distilled":rep("dist",yp_dist)}
io.open(OUT/"cr_base_metrics.json","w",encoding="utf-8").write(json.dumps(base,indent=2))
print("[done] distilled:",base["student_distilled"]); print("cache + base metrics saved ->",OUT)
