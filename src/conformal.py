"""Split-conformal prediction with three score variants:
    - "score" : 1 - softmax prob of true label (Vovk et al. 2005)
    - "aps"   : Adaptive Prediction Sets (Romano et al. 2020)
    - "raps"  : Regularized Adaptive Prediction Sets (Angelopoulos et al. 2021)

All methods guarantee marginal coverage 1 - alpha under exchangeability.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np


class SplitConformalPredictor:
    """Generic split-conformal wrapper. Wraps any classifier exposing
    `predict_proba` (and optionally `decision_function`)."""

    def __init__(self, method: str = "aps", alpha: float = 0.10,
                 k_reg: int = 5, lambda_reg: float = 0.001):
        assert method in ("score", "aps", "raps")
        self.method = method
        self.alpha = alpha
        self.k_reg = k_reg
        self.lambda_reg = lambda_reg
        self.quantile_: Optional[float] = None
        self.classes_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Score functions
    # ------------------------------------------------------------------
    def _score(self, probs: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Per-sample nonconformity score (smaller = more conforming)."""
        if self.method == "score":
            # 1 - p(y_true)
            return 1.0 - probs[np.arange(len(y)), y]

        # APS / RAPS — rank-based scores
        n, k = probs.shape
        order = np.argsort(-probs, axis=1)              # classes by descending prob
        ranks = np.empty_like(order)
        rows = np.arange(n)[:, None]
        ranks[rows, order] = np.arange(k)[None, :] + 1  # rank 1..k

        # Cumulative probability up to and including true class
        sorted_probs = np.take_along_axis(probs, order, axis=1)
        cum_probs = np.cumsum(sorted_probs, axis=1)
        true_cum = cum_probs[np.arange(n), ranks[np.arange(n), y] - 1]

        if self.method == "aps":
            return true_cum
        # RAPS — penalise large ranks
        penalty = self.lambda_reg * np.maximum(0, ranks[np.arange(n), y] - self.k_reg)
        return true_cum + penalty

    def fit(self, probs_calib: np.ndarray, y_calib: np.ndarray,
            classes: Optional[np.ndarray] = None) -> "SplitConformalPredictor":
        self.classes_ = classes if classes is not None else np.unique(y_calib)
        scores = self._score(probs_calib, y_calib)
        # Quantile with +1 correction (Vovk et al.)
        n = len(scores)
        q_level = np.ceil((1 - self.alpha) * (n + 1)) / n
        q_level = min(q_level, 1.0)
        self.quantile_ = float(np.quantile(scores, q_level, method="higher"))
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_set(self, probs: np.ndarray) -> List[np.ndarray]:
        """Return a list of arrays of class indices in the conformal set."""
        if self.quantile_ is None:
            raise RuntimeError("Call fit() before predict_set().")
        n, k = probs.shape
        order = np.argsort(-probs, axis=1)
        sorted_probs = np.take_along_axis(probs, order, axis=1)
        cum = np.cumsum(sorted_probs, axis=1)

        sets: List[np.ndarray] = []
        for i in range(n):
            cum_i = cum[i]
            if self.method == "score":
                # include any class with 1 - p <= q  =>  p >= 1 - q
                mask = probs[i] >= 1.0 - self.quantile_
                sets.append(np.where(mask)[0])
            else:
                # smallest prefix whose cumulative >= q
                idx = np.searchsorted(cum_i, self.quantile_)
                idx = min(idx + 1, k)
                sets.append(order[i, :idx])
        return sets

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def evaluate(self, probs_test: np.ndarray, y_test: np.ndarray) -> dict:
        sets = self.predict_set(probs_test)
        sizes = np.array([len(s) for s in sets], dtype=np.float64)
        covered = np.array([y_test[i] in sets[i] for i in range(len(y_test))])
        return {
            "empirical_coverage": float(covered.mean()),
            "target_coverage": float(1 - self.alpha),
            "avg_set_size": float(sizes.mean()),
            "median_set_size": float(np.median(sizes)),
            "singleton_accuracy": float(
                np.mean([y_test[i] == list(s)[0] for i, s in enumerate(sets) if len(s) == 1])
                if any(len(s) == 1 for s in sets) else 0.0
            ),
            "empty_rate": float(np.mean(sizes == 0)),
        }
