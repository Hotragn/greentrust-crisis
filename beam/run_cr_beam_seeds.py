import json, os
from cr_beam_seeds import run
r = run.remote()
out = os.path.join(os.path.dirname(__file__), "..", "camera-ready", "results")
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "seeds_raw.json"), "w", encoding="utf-8") as f:
    json.dump(r, f)
print("REMOTE_DONE -> seeds_raw.json")
for mdl, per in r["models"].items():
    for s, d in per.items():
        print(f"  {mdl} seed={s} f1={d['test_macro_f1']:.4f} acc={d['test_acc']:.4f} "
              f"infra marg={d['conformal']['marginal']['infra_covered']}/19 "
              f"cc={d['conformal']['class_conditional']['infra_covered']}/19")
print("  student f1=", round(r["student"]["test_macro_f1"], 4))
