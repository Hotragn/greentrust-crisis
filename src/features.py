"""TF-IDF feature engineering + mutual-information-based feature selection
for the knowledge-distillation pipeline.

Teacher = high-dimensional TF-IDF (50K features, 1-2 grams).
Student = MI-selected subset of unigrams (5K features), distilled from teacher.
"""
from __future__ import annotations
from typing import Tuple, List, Optional
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif


def build_teacher_vectorizer(max_features: int = 50_000,
                             ngram_range: Tuple[int, int] = (1, 2),
                             min_df: int = 2) -> TfidfVectorizer:
    """High-dimensional TF-IDF vectorizer for the teacher model."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
        token_pattern=r"\b\w[\w\u00C0-\u024F]+\b",  # Latin extended (covers fr/es/ht)
    )


def build_student_vectorizer(max_features: int = 5_000,
                             ngram_range: Tuple[int, int] = (1, 1),
                             min_df: int = 3) -> TfidfVectorizer:
    """Compact TF-IDF vectorizer for the student model."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
        token_pattern=r"\b\w[\w\u00C0-\u024F]+\b",
    )


def select_features_by_mi(X_train: sp.csr_matrix, y_train: np.ndarray,
                          top_k: int,
                          random_state: int = 42) -> np.ndarray:
    """Return indices of the top-k features by mutual information with y."""
    # mutual_info_classif needs discrete labels and continuous features; TF-IDF works.
    mi = mutual_info_classif(X_train, y_train, random_state=random_state)
    top = np.argsort(mi)[::-1][:top_k]
    return np.sort(top)  # keep sorted for sparse slicing efficiency


class TfidfPipeline:
    """Wraps the teacher and student vectorizers + their fitted feature selections."""

    def __init__(self, teacher_vec: TfidfVectorizer, student_vec: TfidfVectorizer,
                 student_top_indices: Optional[np.ndarray] = None):
        self.teacher_vec = teacher_vec
        self.student_vec = student_vec
        self.student_top_indices = student_top_indices

    def transform_teacher(self, texts: List[str]) -> sp.csr_matrix:
        return self.teacher_vec.transform(texts)

    def transform_student(self, texts: List[str]) -> sp.csr_matrix:
        Xs_full = self.student_vec.transform(texts)
        if self.student_top_indices is not None:
            return Xs_full[:, self.student_top_indices]
        return Xs_full

    def fit_student(self, texts: List[str], top_k: Optional[int] = None) -> sp.csr_matrix:
        Xs_full = self.student_vec.fit_transform(texts)
        return Xs_full  # full vocabulary; selection indices are stored separately

    def student_feature_names(self) -> List[str]:
        names = self.student_vec.get_feature_names_out()
        if self.student_top_indices is not None:
            return [names[i] for i in self.student_top_indices]
        return list(names)
