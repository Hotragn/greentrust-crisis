"""Teacher, Student, and Distillation models.

Teacher  : Multinomial Logistic Regression over high-dimensional TF-IDF
           (a defensible, deterministic, CPU-friendly "teacher").

Student  : A smaller Logistic Regression over MI-pruned features, trained with
           (i) hard labels (cross-entropy) and (ii) soft teacher logits (KL with
           temperature). This is the classic Hinton et al. distillation recipe
           adapted for CPU-friendly linear models.

Because both teacher and student are linear models over TF-IDF, the energy
footprint of inference is dominated by the feature dimensionality. This makes
the dimensionality reduction (50K → 5K) the primary source of energy savings,
while distillation preserves most of the teacher's accuracy.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Numerical stability helpers
_EPS = 1e-12


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _label_smoothing(onehot: np.ndarray, smoothing: float = 0.1) -> np.ndarray:
    return onehot * (1 - smoothing) + smoothing / onehot.shape[1]


def _to_onehot(y: np.ndarray, n_classes: int) -> np.ndarray:
    o = np.zeros((len(y), n_classes), dtype=np.float64)
    o[np.arange(len(y)), y] = 1.0
    return o


class TeacherModel:
    """Multinomial logistic regression teacher."""

    def __init__(self, C: float = 1.0, max_iter: int = 400, random_state: int = 42):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.clf: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: sp.csr_matrix, y: np.ndarray) -> "TeacherModel":
        self.classes_ = np.unique(y)
        # class weights to handle imbalance (e.g. 'not_related' is the minority)
        cw = compute_class_weight("balanced", classes=self.classes_, y=y)
        weights = dict(zip(self.classes_, cw))
        self.scaler = StandardScaler(with_mean=False)  # keep sparse
        Xs = self.scaler.fit_transform(X)
        self.clf = LogisticRegression(
            C=self.C, max_iter=self.max_iter, random_state=self.random_state,
            class_weight=weights, solver="lbfgs"
        )
        self.clf.fit(Xs, y)
        return self

    def decision_function(self, X: sp.csr_matrix) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.clf.decision_function(Xs)

    def predict_proba(self, X: sp.csr_matrix) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.clf.predict_proba(Xs)

    def predict(self, X: sp.csr_matrix) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


class StudentModel:
    """Distilled student: small logistic regression with custom training loop
    that mixes hard-label cross-entropy with soft-label KL divergence."""

    def __init__(self, C: float = 0.5, temperature: float = 4.0, alpha: float = 0.7,
                 epochs: int = 30, lr: float = 0.5, random_state: int = 42,
                 lr_decay: float = 0.98):
        self.C = C
        self.temperature = temperature
        self.alpha = alpha
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state
        self.lr_decay = lr_decay
        self.W: Optional[np.ndarray] = None   # (n_features, n_classes)
        self.b: Optional[np.ndarray] = None   # (n_classes,)
        self.classes_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Weight init
    # ------------------------------------------------------------------
    def _init_weights(self, n_features: int, n_classes: int):
        rng = np.random.default_rng(self.random_state)
        # Glorot-style init
        limit = np.sqrt(6.0 / (n_features + n_classes))
        self.W = rng.uniform(-limit, limit, size=(n_features, n_classes))
        self.b = np.zeros(n_classes, dtype=np.float64)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _logits(self, X: sp.csr_matrix) -> np.ndarray:
        return X @ self.W + self.b

    def predict_proba(self, X: sp.csr_matrix) -> np.ndarray:
        return _softmax(self._logits(X), temperature=1.0)

    def predict(self, X: sp.csr_matrix) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    # ------------------------------------------------------------------
    # Warm start from sklearn logistic regression (faster convergence)
    # ------------------------------------------------------------------
    def _warm_start(self, X: sp.csr_matrix, y: np.ndarray, y_idx: np.ndarray,
                    n_classes: int):
        """Initialise W from a quick sklearn LogisticRegression fit."""
        from sklearn.linear_model import LogisticRegression
        try:
            lr = LogisticRegression(C=self.C, max_iter=200, solver="lbfgs",
                                    random_state=self.random_state)
            lr.fit(X, y)
            # Reshape coef_ to (n_features, n_classes) — binary case has shape (1, n_features)
            if lr.coef_.shape[0] == 1:
                # Binary: expand to two classes
                self.W = np.vstack([-lr.coef_[0], lr.coef_[0]]).T  # (n_features, 2)
                self.b = np.array([-lr.intercept_[0], lr.intercept_[0]])
            else:
                self.W = lr.coef_.T.copy()       # (n_features, n_classes)
                self.b = lr.intercept_.copy()
        except Exception:
            # Fallback: random init
            self._init_weights(X.shape[1], n_classes)

    # ------------------------------------------------------------------
    # Training with distillation
    # ------------------------------------------------------------------
    def fit_distilled(self, X: sp.csr_matrix, y: np.ndarray,
                      teacher_logits: np.ndarray) -> "StudentModel":
        """Train with mix of hard labels (y) and soft teacher logits."""
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        y_idx = np.searchsorted(self.classes_, y)
        n_samples, n_features = X.shape

        # Warm-start from sklearn LR — gives a strong baseline to distill from
        self._warm_start(X, y, y_idx, n_classes)

        # Precompute soft targets (teacher softmax with temperature)
        soft_targets = _softmax(teacher_logits, temperature=self.temperature)
        hard_targets = _to_onehot(y_idx, n_classes)
        hard_targets = _label_smoothing(hard_targets, smoothing=0.05)

        # Class-balanced loss weights (helps with imbalanced crisis data)
        class_counts = np.bincount(y_idx, minlength=n_classes).astype(np.float64)
        class_weights = class_counts.sum() / (n_classes * np.maximum(1, class_counts))
        sample_weights = class_weights[y_idx]

        # Mini-batch SGD with LR decay
        rng = np.random.default_rng(self.random_state)
        batch_size = min(512, n_samples)
        lr = self.lr
        for epoch in range(self.epochs):
            perm = rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                idx = perm[start:start + batch_size]
                Xb = X[idx]
                yb_hard = hard_targets[idx]
                yb_soft = soft_targets[idx]
                wb = sample_weights[idx][:, None]

                logits = self._logits(Xb) / self.temperature
                probs = _softmax(logits, temperature=1.0)

                # Combined gradient with sample weighting
                grad_logits = (
                    self.alpha * (probs - yb_soft) / self.temperature
                    + (1 - self.alpha) * (probs - yb_hard)
                ) * wb / len(idx)

                grad_W = Xb.T @ grad_logits + (1.0 / self.C) * self.W / len(idx)
                grad_b = grad_logits.sum(axis=0)

                self.W -= lr * grad_W
                self.b -= lr * grad_b
            lr *= self.lr_decay  # decay LR
        return self

    def fit_hard(self, X: sp.csr_matrix, y: np.ndarray) -> "StudentModel":
        """Plain logistic regression (no distillation) — uses warm-start."""
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        y_idx = np.searchsorted(self.classes_, y)
        n_samples, n_features = X.shape
        self._warm_start(X, y, y_idx, n_classes)
        # No further SGD needed — the warm-start LR is already a strong baseline.
        return self

    def decision_function(self, X: sp.csr_matrix) -> np.ndarray:
        """Return raw logits — conformal prediction consumes these."""
        return self._logits(X)
