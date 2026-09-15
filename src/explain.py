"""Token-level saliency: returns the top-k tokens contributing to the model's
predicted class for a single input text.

Two complementary methods:
    1. Coefficient-based  : for the linear student, contribution = coef * tfidf.
       Fast, deterministic, faithful to the model (LIME-like for linear models).
    2. Occlusion (LIME-style fallback): perturb tokens and measure change in
       predicted probability. Slow but model-agnostic.

The coefficient method is the default for the student; we expose occlusion for
completeness and as a sanity check.
"""
from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import scipy.sparse as sp


def explain_linear(coef_matrix: np.ndarray, feature_names: List[str],
                   vectorizer, text: str, class_index: int,
                   top_k: int = 10) -> List[Tuple[str, float]]:
    """Return top-k tokens with their signed contribution to `class_index`.

    `coef_matrix` : (n_features, n_classes) from the linear model
    `vectorizer`  : a fitted TfidfVectorizer (must support transform + vocab)
    """
    X = vectorizer.transform([text])
    if X.nnz == 0:
        return []
    coefs = coef_matrix[:, class_index]
    contributions = []
    for j in X.indices:
        contributions.append((feature_names[j], float(coefs[j] * X[0, j])))
    contributions.sort(key=lambda t: -abs(t[1]))
    return contributions[:top_k]


def explain_occlusion(predict_proba_fn, vectorizer, text: str,
                      class_index: int, top_k: int = 8,
                      n_samples: int = 50, random_state: int = 42
                      ) -> List[Tuple[str, float]]:
    """LIME-style occlusion importance: drop tokens randomly and measure change
    in P(class_index).
    """
    rng = np.random.default_rng(random_state)
    tokens = text.split()
    if not tokens:
        return []

    # Baseline
    base_p = predict_proba_fn(vectorizer.transform([text]))[0, class_index]

    # Importance per token: E[perturbed_p - base_p]
    importance = np.zeros(len(tokens), dtype=np.float64)
    for _ in range(n_samples):
        # mask out ~30% of tokens
        mask = rng.random(len(tokens)) > 0.30
        if mask.sum() == 0:
            continue
        perturbed = " ".join(t for t, m in zip(tokens, mask) if m)
        p = predict_proba_fn(vectorizer.transform([perturbed]))[0, class_index]
        importance += (base_p - p)  # if removing token drops p, importance > 0

    importance /= max(1, n_samples)
    ranked = sorted(enumerate(importance), key=lambda kv: -abs(kv[1]))[:top_k]
    return [(tokens[i], float(v)) for i, v in ranked]
