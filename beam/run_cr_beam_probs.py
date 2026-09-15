import json, os
import numpy as np
from cr_beam_probs import run
r = run.remote()
out = os.path.join(os.path.dirname(__file__), "..", "camera-ready", "results")
os.makedirs(out, exist_ok=True)
# save per-example probs as npz (aligned to the deterministic split)
kw = {"y_cal": np.array(r["y_cal"]), "y_te": np.array(r["y_te"]),
      "lang_cal": np.array(r["lang_cal"]), "lang_te": np.array(r["lang_te"])}
meta = {"device": r["device"], "gpu": r["gpu"], "epochs": r["epochs"], "models": {}}
for mdl, tag in [("bert-base-multilingual-cased", "mbert"), ("xlm-roberta-base", "xlmr")]:
    m = r["models"][mdl]
    kw[f"{tag}_P_cal"] = np.array(m["P_cal"]); kw[f"{tag}_P_te"] = np.array(m["P_te"])
    meta["models"][mdl] = {k: v for k, v in m.items() if k not in ("P_cal", "P_te")}
np.savez_compressed(os.path.join(out, "transformer_probs.npz"), **kw)
with open(os.path.join(out, "baselines_v2.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print("REMOTE_DONE -> transformer_probs.npz + baselines_v2.json")
print(json.dumps(meta, indent=2)[:1500])
