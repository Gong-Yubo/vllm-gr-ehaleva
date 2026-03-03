# Imported from: https://github.com/Kuaishou-OneRec/OpenOneRec
# Original commit: ae668a6a30bd71bde7a3f459fe0ac958182ce1d2
# Date imported: 2026‑01‑09
# License: Apache 2.0 License (inherited from source)

"""
Label Prediction Task Module

Classification task for predicting user engagement with video content.
Uses logprobs-based classification with AUC and wuAUC metrics.
"""

from . import utils
from .config import LABEL_PRED_CONFIG
from .evaluator import LabelPredEvaluator

__all__ = [
    "LABEL_PRED_CONFIG",
    "LabelPredEvaluator",
    "utils",
]
