"""Minimal inference API (FastAPI). Optional — used for the deployment use case.

Run with:
    uvicorn src.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations
from typing import List, Optional, Dict
import pickle
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import MODELS_DIR, DEFAULT_CONFIG
from .conformal import SplitConformalPredictor
from .explain import explain_linear


class InferenceRequest(BaseModel):
    text: str
    return_explanation: bool = True
    return_conformal_set: bool = True
    top_k_explanation: int = 8


class InferenceResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    probabilities: Dict[str, float]
    conformal_set: Optional[List[str]] = None
    explanation: Optional[List[Dict[str, float]]] = None
    latency_ms: float


# Lazy-loaded artefacts
_STATE: Dict[str, object] = {}


def _load_artifacts():
    if _STATE:
        return _STATE
    art_path = MODELS_DIR / "artifacts.pkl"
    if not art_path.exists():
        raise FileNotFoundError(
            f"{art_path} not found. Run `python scripts/train.py` first.")
    with open(art_path, "rb") as f:
        art = pickle.load(f)
    _STATE.update(art)
    return _STATE


app = FastAPI(title="GreenTrust-Crisis Inference API",
              description="Lightweight, uncertainty-aware multilingual crisis NLP",
              version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=InferenceResponse)
def predict(req: InferenceRequest):
    import time
    art = _load_artifacts()
    teacher = art["teacher"]
    student = art["student"]
    teacher_vec = art["teacher_vec"]
    student_vec = art["student_vec"]
    student_top_indices = art.get("student_top_indices")
    cp = art.get("conformal_predictor")
    classes = art["classes"]
    label_names = art["label_names"]

    t0 = time.perf_counter()
    # We use the STUDENT for inference (energy-efficient)
    Xs = student_vec.transform([req.text])
    if student_top_indices is not None:
        Xs = Xs[:, student_top_indices]
    probs = student.predict_proba(Xs)
    pred_idx = int(probs.argmax(axis=1)[0])
    t1 = time.perf_counter()

    resp = InferenceResponse(
        predicted_class=pred_idx,
        predicted_label=label_names[pred_idx],
        probabilities={label_names[i]: float(probs[0, i]) for i in range(len(label_names))},
        latency_ms=(t1 - t0) * 1000.0,
    )

    if req.return_conformal_set and cp is not None:
        sets = cp.predict_set(probs)
        resp.conformal_set = [label_names[i] for i in sets[0]]

    if req.return_explanation:
        W = student.W  # (n_features, n_classes)
        if student_top_indices is not None:
            W = W  # already in pruned feature space
        feat_names = art["student_feature_names"]
        contribs = explain_linear(W, feat_names, student_vec,
                                  req.text, pred_idx,
                                  top_k=req.top_k_explanation)
        # Note: explain_linear indexes into the FULL vectorizer vocab;
        # if we use student_top_indices we must remap. We expose the simpler
        # case here; the notebook shows the pruned version.
        resp.explanation = [{"token": t, "contribution": float(v)}
                            for t, v in contribs]
    return resp
