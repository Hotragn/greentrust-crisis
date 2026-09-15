import json, os
from cr_beam_baselines import run
r = run.remote()
out = os.path.join(os.path.dirname(__file__), "..", "camera-ready", "results")
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "baselines.json"), "w", encoding="utf-8") as f:
    json.dump(r, f, indent=2)
print("REMOTE_DONE -> baselines.json")
print(json.dumps(r, indent=2)[:2500])
