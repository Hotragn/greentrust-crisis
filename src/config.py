"""Central configuration for GreenTrust-Crisis experiments."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"
CONFIGS_DIR  = PROJECT_ROOT / "configs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Dataset definition: 36 disaster-related labels from Figure-Eight/Appen,
# shipped as `community-datasets/disaster_response_messages` on the HF Hub.
# -----------------------------------------------------------------------------
PRIMARY_LABEL = "related"                       # binary: 1 = disaster-related, 0 = not
SECONDARY_LABEL = "genre"                        # multi-class: direct / news / social
WEATHER_LABELS = ["floods", "storm", "fire", "earthquake", "cold", "other_weather"]
AID_LABELS = ["medical_help", "search_and_rescue", "water", "food", "shelter",
              "clothing", "money", "refugees", "death", "missing_people"]

# Multilingual mapping. The "original" field holds the non-English source
# text. We detect language heuristically; messages whose original is empty
# are treated as English.
SUPPORTED_LANGS = ["en", "fr", "ht", "ur", "es"]

# Crisis event taxonomy (derived from the dataset's source events).
CRISIS_EVENTS = {
    "earthquake_haiti_2010":   {"langs": ["ht", "fr", "en"], "label_hint": "earthquake"},
    "earthquake_chile_2010":   {"langs": ["es", "en"],        "label_hint": "earthquake"},
    "floods_pakistan_2010":    {"langs": ["ur", "en"],        "label_hint": "floods"},
    "sandy_usa_2012":          {"langs": ["en"],              "label_hint": "storm"},
    "news_archive":            {"langs": ["en"],              "label_hint": None},
}

# -----------------------------------------------------------------------------
# Hyperparameters (kept CPU-friendly — entire pipeline runs in < 10 minutes)
# -----------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    random_state: int = 42
    test_size: float = 0.20        # held-out test (after the official split)
    calibration_size: float = 0.25 # of training, for conformal calibration

    # TF-IDF (teacher) — high-dimensional, rich features
    teacher_max_features: int = 30_000
    teacher_ngram_range: tuple = (1, 2)
    teacher_min_df: int = 3

    # TF-IDF (student) — aggressively pruned via mutual information
    student_max_features: int = 5_000
    student_ngram_range: tuple = (1, 2)
    student_min_df: int = 3

    # Classifier
    teacher_C: float = 0.3       # stronger L2 regularization to prevent overfitting
    student_C: float = 0.5

    # Distillation
    distill_temperature: float = 6.0
    distill_alpha: float = 0.3    # 0.3 = 30% soft-label loss, 70% hard-label loss
    distill_epochs: int = 10

    # Conformal prediction
    conformal_alpha: float = 0.10 # 1 - alpha = 0.90 target coverage
    conformal_method: str = "aps" # "aps" | "raps" | "score"

    # Energy measurement
    energy_repeats: int = 200     # how many inferences to average for energy/latency

    # Output artefacts
    results_dir: Path = RESULTS_DIR
    models_dir: Path = MODELS_DIR

    def to_json(self, path: Path) -> None:
        d = self.__dict__.copy()
        d["results_dir"] = str(d["results_dir"])
        d["models_dir"] = str(d["models_dir"])
        path.write_text(json.dumps(d, indent=2))

DEFAULT_CONFIG = ExperimentConfig()
