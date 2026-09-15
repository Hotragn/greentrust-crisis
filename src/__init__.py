"""GreenTrust-Crisis: A Lightweight, Uncertainty-Aware Multilingual NLP Framework.

Submodules:
    data       : dataset loading, multilingual splitting, label hierarchies
    features   : TF-IDF feature engineering + mutual-information feature selection
    models     : teacher / student classifiers and the distillation wrapper
    conformal  : split-conformal prediction with APS / RAPS / score variants
    explain    : token-level saliency via coefficient attribution + LIME-style
    energy     : CPU energy and latency instrumentation
    api        : minimal FastAPI inference server (optional)
"""

__version__ = "1.0.0"
__author__ = "GreenTrust-Crisis Authors"
__license__ = "MIT"
