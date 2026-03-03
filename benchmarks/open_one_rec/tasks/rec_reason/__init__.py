# Imported from: https://github.com/Kuaishou-OneRec/OpenOneRec
# Original commit: ae668a6a30bd71bde7a3f459fe0ac958182ce1d2
# Date imported: 2026‑01‑09
# License: Apache 2.0 License (inherited from source)

"""
Recommendation Reason Task Module
"""

from . import utils
from .config import REC_REASON_CONFIG
from .evaluator import RecoReasonEvaluator

__all__ = [
    "REC_REASON_CONFIG",
    "RecoReasonEvaluator",
    "utils",
]
