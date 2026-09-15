"""Dataset loading, language detection, and multi-task label engineering."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from .config import (DATA_DIR, PRIMARY_LABEL, SECONDARY_LABEL,
                     WEATHER_LABELS, AID_LABELS, SUPPORTED_LANGS,
                     CRISIS_EVENTS, DEFAULT_CONFIG)


# -----------------------------------------------------------------------------
# Language detection (heuristic, dependency-free)
# -----------------------------------------------------------------------------
_LATIN_SCRIPT_RE = re.compile(r"[A-Za-zÀ-ÿ]")
_HAITIAN_HINTS = {"nap", "ap", "pou", "ki", "se", "yo", "mwen", "nou", "ou", "li",
                  "pa", "an", "nan", "yon", "sa", "ki", "ki", "gen", "anpil"}
_FRENCH_HINTS = {"le", "la", "les", "un", "une", "et", "de", "que", "pas", "pour",
                 "avec", "sans", "dans", "sur", "nous", "vous", "ils", "elle",
                 "il", "y", "une", "est", "ont", "ont", "se", "ce", "mais",
                 "donc", "or", "ni", "car", "qui", "où", "quand", "comment"}
_SPANISH_HINTS = {"el", "la", "los", "las", "un", "una", "que", "de", "no", "y",
                  "con", "por", "para", "está", "están", "como", "más",
                  "hay", "es", "en", "del", "las", "los", "una", "uno",
                  "pero", "cuando", "donde", "cómo", "qué", "porque", "se"}
_URDU_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")


def detect_language(text: str) -> str:
    """Return ISO-639-1 code: en / fr / ht / es / ur."""
    if not text or not isinstance(text, str):
        return "en"
    if _URDU_RE.search(text):
        return "ur"
    tokens = [t.lower() for t in re.findall(r"[A-Za-zÀ-ÿ]+", text)]
    if not tokens:
        return "en"
    tok_set = set(tokens)
    haitian = len(tok_set & _HAITIAN_HINTS)
    french = len(tok_set & _FRENCH_HINTS)
    spanish = len(tok_set & _SPANISH_HINTS)
    # Haitian Creole shares most French tokens; boost haitian when present
    if haitian >= 2 and haitian >= french - 1:
        return "ht"
    if french >= 2 and french > spanish:
        return "fr"
    if spanish >= 2 and spanish > french:
        return "es"
    return "en"


# -----------------------------------------------------------------------------
# Crisis-event attribution (heuristic mapping based on keyword / language)
# -----------------------------------------------------------------------------
_EVENT_KEYWORDS = {
    "earthquake_haiti_2010":   ["haiti", "port-au-prince", "port au prince", "haitian",
                                "seisme", "tremblement", "goudou goudou"],
    "earthquake_chile_2010":   ["chile", "chileno", "santiago", "concepcion", "chilena"],
    "floods_pakistan_2010":    ["pakistan", "pakistani", "sindh", "punjab", "indus",
                                "khyber", "balochistan", "flood"],
    "sandy_usa_2012":          ["sandy", "hurricane sandy", "new york", "new jersey",
                                "manhattan", "staten island"],
}


def attribute_event(row: pd.Series) -> str:
    """Attribute a message to one of the canonical crisis events (or 'other')."""
    text = f"{row.get('message','')} {row.get('original','')}".lower()
    # First by language hint
    lang = row.get("lang", "en")
    if lang == "ht":
        return "earthquake_haiti_2010"
    if lang == "ur":
        return "floods_pakistan_2010"
    # Then by keyword
    best, best_score = "other", 0
    for event, kws in _EVENT_KEYWORDS.items():
        score = sum(1 for k in kws if k in text)
        if score > best_score:
            best, best_score = event, score
    return best if best_score > 0 else "news_archive"


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def load_raw_splits() -> Dict[str, pd.DataFrame]:
    """Load the three raw parquet splits shipped with the dataset."""
    return {
        "train":      pd.read_parquet(DATA_DIR / "train.parquet"),
        "validation": pd.read_parquet(DATA_DIR / "validation.parquet"),
        "test":       pd.read_parquet(DATA_DIR / "test.parquet"),
    }


def build_unified_dataframe() -> pd.DataFrame:
    """Concatenate splits, detect language, attribute event, derive a 5-class
    severity-aware task label, and return one unified dataframe.
    """
    splits = load_raw_splits()
    df = pd.concat(splits.values(), ignore_index=True)

    # Detect language of the ORIGINAL message; fall back to English
    df["lang"] = df["original"].apply(
        lambda x: detect_language(x) if isinstance(x, str) and x.strip() else "en"
    )
    # Event attribution
    df["event"] = df.apply(attribute_event, axis=1)

    # ------------------------------------------------------------------
    # Primary binary task: related (1) vs not-related (0)
    # ------------------------------------------------------------------
    df["y_binary"] = df[PRIMARY_LABEL].astype(int)

    # ------------------------------------------------------------------
    # Secondary 5-class task: crisis type (more useful for the paper)
    #   0 = not_related
    #   1 = weather_disaster   (any of WEATHER_LABELS == 1)
    #   2 = aid_request         (request == 1 OR aid_related == 1)
    #   3 = infrastructure      (infrastructure_related == 1)
    #   4 = other_related       (related == 1 but none of the above)
    # ------------------------------------------------------------------
    def _crisis_type(row):
        if row[PRIMARY_LABEL] != 1:
            return 0
        if any(row[c] == 1 for c in WEATHER_LABELS):
            return 1
        if row.get("request", 0) == 1 or row.get("aid_related", 0) == 1:
            return 2
        if row.get("infrastructure_related", 0) == 1:
            return 3
        return 4
    df["y_crisis_type"] = df.apply(_crisis_type, axis=1)

    # Use the original message where available (preserves multilingual signal)
    df["text"] = df.apply(
        lambda r: r["original"] if isinstance(r["original"], str) and r["original"].strip()
                  else r["message"],
        axis=1
    )
    df["text_en"] = df["message"]  # always-English translation

    # Keep useful columns
    keep = ["text", "text_en", "lang", "event", "genre", "split",
            "y_binary", "y_crisis_type"] + WEATHER_LABELS + AID_LABELS
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    df["text_en"] = df["text_en"].astype(str)
    return df


def stratified_split(df: pd.DataFrame,
                     test_size: float = DEFAULT_CONFIG.test_size,
                     calib_size: float = DEFAULT_CONFIG.calibration_size,
                     random_state: int = DEFAULT_CONFIG.random_state
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Produce train / calibration / test splits stratified on (lang, y_crisis_type)."""
    df = df.copy()
    df["strat"] = df["lang"].astype(str) + "_" + df["y_crisis_type"].astype(str)
    # Drop strata with <2 members
    counts = df["strat"].value_counts()
    valid = counts[counts >= 2].index
    df_valid = df[df["strat"].isin(valid)].copy()

    train_df, test_df = train_test_split(
        df_valid, test_size=test_size, stratify=df_valid["strat"],
        random_state=random_state
    )
    # Further split train → train / calibration
    cal_frac_of_train = calib_size / (1.0 - test_size)
    train_df, calib_df = train_test_split(
        train_df, test_size=cal_frac_of_train, stratify=train_df["strat"],
        random_state=random_state
    )
    return (train_df.reset_index(drop=True),
            calib_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


def dataset_summary(df: pd.DataFrame) -> Dict[str, object]:
    """Compute summary statistics for the paper's dataset section."""
    return {
        "n_total": int(len(df)),
        "by_language": df["lang"].value_counts().to_dict(),
        "by_event": df["event"].value_counts().to_dict(),
        "by_genre": df["genre"].value_counts().to_dict(),
        "by_crisis_type": df["y_crisis_type"].value_counts().sort_index().to_dict(),
        "by_binary": df["y_binary"].value_counts().to_dict(),
        "avg_text_length_chars": float(df["text"].str.len().mean()),
        "avg_text_length_tokens": float(df["text"].str.split().str.len().mean()),
        "multilingual_pct": float((df["lang"] != "en").mean() * 100),
    }
