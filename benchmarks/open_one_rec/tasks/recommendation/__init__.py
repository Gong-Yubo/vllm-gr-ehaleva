# Imported from: https://github.com/Kuaishou-OneRec/OpenOneRec
# Original commit: ae668a6a30bd71bde7a3f459fe0ac958182ce1d2
# Date imported: 2026‑01‑09
# License: Apache 2.0 License (inherited from source)

"""
Recommendation Task Module

Universal module for all recommendation tasks including:
- label_cond: Predict next video given specified consumption behavior
- video: Next video prediction
- product: Predict next clicked product
- ad: Predict next clicked advertisement
"""

from . import utils
from .config import (
    AD_CONFIG,
    INTERACTIVE_CONFIG,
    LABEL_COND_CONFIG,
    PRODUCT_CONFIG,
    RECOMMENDATION_EVALUATION_CONFIG,
    RECOMMENDATION_GENERATION_CONFIG,
    RECOMMENDATION_PROMPT_CONFIG,
    RECOMMENDATION_TASK_CONFIGS,
    VIDEO_CONFIG,
)
from .evaluator import RecommendationEvaluator

__all__ = [
    # Configs
    "LABEL_COND_CONFIG",
    "VIDEO_CONFIG",
    "PRODUCT_CONFIG",
    "AD_CONFIG",
    "INTERACTIVE_CONFIG",
    "RECOMMENDATION_PROMPT_CONFIG",
    "RECOMMENDATION_TASK_CONFIGS",
    "RECOMMENDATION_GENERATION_CONFIG",
    "RECOMMENDATION_EVALUATION_CONFIG",
    # Classes
    "RecommendationEvaluator",
    # Utils module
    "utils",
]
