# Imported from: https://github.com/Kuaishou-OneRec/OpenOneRec
# Original commit: ae668a6a30bd71bde7a3f459fe0ac958182ce1d2
# Date imported: 2026‑01‑09
# License: Apache 2.0 License (inherited from source)
"""
Base Loader for all task data loaders

Provides common functionality for data loading, sampling, and file path resolution.
"""

import json
import logging
import os
from abc import ABC
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Dict, Optional, Type

import pandas as pd

from .tasks.item_understand.config import ITEM_UNDERSTAND_CONFIG
from .tasks.item_understand.evaluator import ItemUnderstandEvaluator
from .tasks.label_pred import LabelPredEvaluator
from .tasks.label_pred.config import LABEL_PRED_CONFIG
from .tasks.rec_reason.config import REC_REASON_CONFIG
from .tasks.rec_reason.evaluator import RecoReasonEvaluator
from .tasks.recommendation.config import (
    AD_CONFIG,
    INTERACTIVE_CONFIG,
    LABEL_COND_CONFIG,
    PRODUCT_CONFIG,
    VIDEO_CONFIG,
)
from .tasks.recommendation.evaluator import RecommendationEvaluator

logger = logging.getLogger(__name__)


@dataclass
class TaskRegistration:
    """Task registration information"""

    name: str
    config: Dict[str, Any]
    evaluator_class: Type
    category: str  # "general", "recommendation", "caption"


# ========================================
# Unified Task Registry
# ========================================

TASK_REGISTRY: Dict[str, TaskRegistration] = {
    "label_cond": TaskRegistration(
        name="label_cond",
        config=LABEL_COND_CONFIG,
        evaluator_class=RecommendationEvaluator,
        category="recommendation",
    ),
    "video": TaskRegistration(
        name="video",
        config=VIDEO_CONFIG,
        evaluator_class=RecommendationEvaluator,
        category="recommendation",
    ),
    "product": TaskRegistration(
        name="product",
        config=PRODUCT_CONFIG,
        evaluator_class=RecommendationEvaluator,
        category="recommendation",
    ),
    "ad": TaskRegistration(
        name="ad",
        config=AD_CONFIG,
        evaluator_class=RecommendationEvaluator,
        category="recommendation",
    ),
    "interactive": TaskRegistration(
        name="interactive",
        config=INTERACTIVE_CONFIG,
        evaluator_class=RecommendationEvaluator,
        category="recommendation",
    ),
    "label_pred": TaskRegistration(
        name="label_pred",
        config=LABEL_PRED_CONFIG,
        evaluator_class=LabelPredEvaluator,
        category="recommendation",
    ),
    "item_understand": TaskRegistration(
        name="item_understand",
        config=ITEM_UNDERSTAND_CONFIG,
        evaluator_class=ItemUnderstandEvaluator,
        category="caption",
    ),
    "rec_reason": TaskRegistration(
        name="rec_reason",
        config=REC_REASON_CONFIG,
        evaluator_class=RecoReasonEvaluator,
        category="caption",
    ),
}


# ========================================
# Factory Functions
# ========================================


def get_loader(
    task_name: str,
    data_dir: str,
    tokenizer: Optional[Any] = None,
    enable_thinking: Optional[bool] = None,
    custom_input_len: Optional[int] = None,
):
    """
    Get loader instance for a task

    Replaces loader_factory.get_loader()

    Args:
        task_name: Name of the task
        benchmark_version: Version of the benchmark (used for task selection, not passed to loader)
        data_dir: Data directory path
        tokenizer: Tokenizer instance (optional, required for message-based formats)
        enable_thinking: Enable thinking mode (optional, overrides task config if set)
        custom_input_len: Set the input length to a predefined length

    Returns:
        Loader instance

    Raises:
        ValueError: If task_name is not registered
    """
    if task_name not in TASK_REGISTRY:
        available_tasks = ", ".join(TASK_REGISTRY.keys())
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {available_tasks}")

    reg = TASK_REGISTRY[task_name]

    # Create loader instance with aligned parameters
    return BaseLoader(
        task_config=reg.config,
        data_dir=data_dir,
        tokenizer=tokenizer,
        enable_thinking=enable_thinking,
        custom_input_len=custom_input_len,
    )


class DataLoaderWrapper:
    """Wrapper for unified data loading interface"""

    def __init__(
        self,
        model_path: str,
        benchmark_version: str,
        data_dir: str,
        enable_thinking: Optional[bool] = None,
        custom_input_len: Optional[int] = None,
    ):
        self.model_path = model_path
        self._tokenizer = self._create_tokenizer(model_path) if model_path else None

        if custom_input_len is not None:
            custom_input_len = int(custom_input_len)
            if custom_input_len <= 0:
                raise ValueError("custom_input_len must be a positive integer")

        self.benchmark_version = benchmark_version
        self.data_dir = data_dir
        self.enable_thinking = enable_thinking
        self.custom_input_len = custom_input_len
        self._loader_cache = {}

    def _create_tokenizer(self, model_path: str):
        """Create tokenizer from model path"""
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            logger.info(f"Tokenizer loaded from: {model_path}")
            return tokenizer
        except (OSError, ValueError, ImportError) as e:
            raise RuntimeError(f"Failed to load tokenizer from {model_path}: {e}") from e

    def load_data(
        self,
        task_name: str,
        split: str = "test",
        sample_size: Optional[Any] = None,
    ):
        """Load data using new loader system"""
        if task_name not in self._loader_cache:
            self._loader_cache[task_name] = get_loader(
                task_name=task_name,
                data_dir=self.data_dir,
                tokenizer=self._tokenizer,
                enable_thinking=self.enable_thinking,
                custom_input_len=self.custom_input_len,
            )

        loader = self._loader_cache[task_name]
        return loader.load_data(split=split, sample_size=sample_size)


class BaseLoader(ABC):
    """Base class for all task data loaders"""

    def __init__(
        self,
        task_config: Dict[str, Any],
        data_dir: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        enable_thinking: Optional[bool] = None,
        custom_input_len: Optional[int] = None,
    ):
        """Initialize base loader"""
        if custom_input_len is not None:
            custom_input_len = int(custom_input_len)
            if custom_input_len <= 0:
                raise ValueError("custom_input_len must be a positive integer")

        self.task_config = task_config
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking
        self.custom_input_len = custom_input_len
        self.task_name = task_config.get("name", "unknown")

        # Validate tokenizer is provided for messages-based format
        if self.tokenizer is None:
            raise ValueError(
                f"{self.task_name} requires tokenizer for messages-based format. "
                f"Please provide model_path when initializing Benchmark.\n"
                f"Example: Benchmark(task_types=['{self.task_name}'], model_path='your-model-path')"
            )

    def load_data(
        self, split: str = "test", sample_size: Optional[Any] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Load data for the task in messages-based format

        Args:
            split: Dataset split (default "test")
            sample_size: Override sample size (can be int, "full", or None to use task config)

        Returns:
            Dictionary mapping sample_id to sample data:
            {
                sample_id: {
                    "prompt": "formatted prompt from apply_chat_template",
                    "ground_truth": "answer",
                    "metadata": {
                        "row_index": idx,
                        "messages": [...]
                    }
                }
            }
        """
        # Determine effective sample size
        if sample_size is not None:
            if sample_size == "full":
                effective_sample_size = self.task_config.get("size")
            else:
                effective_sample_size = int(sample_size)
        else:
            effective_sample_size = self.task_config.get("sample_size")

        full_size = self.task_config.get("size")

        # Try to load cached sample dataframe
        df = None
        if (
            effective_sample_size is not None
            and full_size is not None
            and effective_sample_size < full_size
        ):
            df = self._load_sample_dataframe(split, effective_sample_size)

        # If no cache, load and sample original data
        if df is None:
            df = self._load_dataframe(split)

            # Perform sampling if needed
            if effective_sample_size is not None and effective_sample_size < len(df):
                df = self._sample_data(df, effective_sample_size)

                # Save sampled data
                if full_size is not None and effective_sample_size < full_size:
                    self._save_sample_data(df, split, effective_sample_size)

        if "messages" not in df.columns:
            raise ValueError(
                f"{self.task_name} requires 'messages' column in data file. "
                f"Found columns: {list(df.columns)}\n"
                f"Please ensure your data is in messages-based format."
            )

        if "metadata" not in df.columns:
            raise ValueError(
                f"{self.task_name} requires 'metadata' column in data file. "
                f"Found columns: {list(df.columns)}\n"
                f"Please ensure your data is in messages-based format."
            )

        logger.info(f"Processing {self.task_name} data in messages-based format")

        result = self._process_dataframe(df)

        return result

    @staticmethod
    def _is_empty_value(value) -> bool:
        """Check if a value is None, NaN, or empty"""
        if value is None:
            return True

        if isinstance(value, float):
            try:
                return pd.isna(value)
            except (ValueError, TypeError):
                return False

        if isinstance(value, str):
            return len(value.strip()) == 0

        try:
            if hasattr(value, "__len__"):
                return len(value) == 0
        except (ValueError, TypeError):
            pass

        return False

    @staticmethod
    def _convert_messages_format(messages: list) -> list:
        """
        Convert message format.

        {"role": "user", "content": [{"type": "text", "text": "..."}]}
        ->
        {"role": "user", "content": "..."}
        """
        converted = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                # Extract text from content list
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                converted.append({"role": msg.get("role"), "content": "".join(text_parts)})
            else:
                # Already in old format
                converted.append(msg)
        return converted

    def _load_custom_chat_template(self):
        """Load custom chat template based on configuration"""
        if not self.tokenizer:
            return

        prompt_config = self.task_config.get("prompt_config", {})
        custom_template = prompt_config.get("custom_chat_template")

        template_path = os.path.join(os.path.dirname(__file__), custom_template)

        if not os.path.exists(template_path):
            raise FileNotFoundError(f"✗ Custom chat template not found: {template_path}")

        with open(template_path, "r", encoding="utf-8") as f:
            self.tokenizer.chat_template = f.read()
        logger.info(f"✓ Loaded custom chat template: {custom_template}")

    def _get_data_file_path(self, split: str) -> str:
        """Get data file path for the given split"""
        if self.data_dir:
            base_dir = self.data_dir
        else:
            base_dir = "./data"

        filename = f"{self.task_name}_{split}.parquet"

        possible_paths = [
            os.path.join(base_dir, self.task_name, filename),
        ]

        for file_path in possible_paths:
            if os.path.exists(file_path):
                return file_path

        return possible_paths[0]

    def _get_sample_data_file_path(self, split: str, sample_size: int) -> str:
        """Get sample data file path"""
        if self.data_dir:
            base_dir = self.data_dir
        else:
            base_dir = "./data"

        possible_paths = [
            os.path.join(
                base_dir,
                self.task_name,
                f"{self.task_name}_{split}_sample_{sample_size}.parquet",
            ),
            os.path.join(
                base_dir,
                f"{self.task_name}_{split}_sample_{sample_size}.parquet",
            ),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return possible_paths[0]

    def _load_dataframe(self, split: str) -> pd.DataFrame:
        """Load DataFrame from data file"""
        data_file = self._get_data_file_path(split)

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        logger.info(f"Loading data file: {data_file}")

        if data_file.endswith(".parquet"):
            df = pd.read_parquet(data_file)
        else:
            raise ValueError(f"Unsupported file format: {data_file}")

        return df

    def _sample_data(self, df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Reproducibly sample data (take head) from DataFrame"""
        if sample_size >= len(df):
            return df

        logger.info(f"Sampling {sample_size} samples (total: {len(df)})")
        return df.sample(sample_size, random_state=42)

    def _save_sample_data(self, df: pd.DataFrame, split: str, sample_size: int):
        """Save sample data in parquet format"""
        sample_file = self._get_sample_data_file_path(split, sample_size)

        sample_dir = os.path.dirname(sample_file)
        if sample_dir:
            os.makedirs(sample_dir, exist_ok=True)

        df.to_parquet(sample_file, index=False)
        logger.info(f"Sample data saved to: {sample_file}")

    def _load_sample_dataframe(self, split: str, sample_size: int) -> Optional[pd.DataFrame]:
        """Load sample dataframe from cache if exists"""
        sample_file = self._get_sample_data_file_path(split, sample_size)

        if not os.path.exists(sample_file):
            return None

        logger.info(f"Loading sample data from cache: {sample_file}")

        df = pd.read_parquet(sample_file)
        return df

    def _process_dataframe(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Process DataFrame and convert to model input format"""
        self._load_custom_chat_template()

        result = {}

        prompt_config = self.task_config.get("prompt_config", {})
        # Command-line parameter has higher priority than config
        if self.enable_thinking is not None:
            enable_thinking = self.enable_thinking
        else:
            enable_thinking = prompt_config.get("enable_thinking", False)

        logger.info(f"Auto Thinking: {'✓ Enabled' if enable_thinking else '✗ Disabled'}")

        for idx, row in df.iterrows():
            sample_id = str(idx)

            messages = row.get("messages")
            if self._is_empty_value(messages):
                logger.info(f"Sample {sample_id}: messages is empty, skipping")
                continue

            if isinstance(messages, str):
                try:
                    messages = json.loads(messages)
                except JSONDecodeError:
                    logger.info(f"Sample {sample_id}: failed to parse messages, skipping")
                    continue

            messages = self._convert_messages_format(messages)

            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except Exception as e:
                logger.info(f"Sample {sample_id}: failed to apply chat template: {e}, skipping")
                continue

            if self.custom_input_len is not None:
                formatted_prompt = self._enforce_prompt_token_length(formatted_prompt, sample_id)

            metadata_raw = row.get("metadata")
            if self._is_empty_value(metadata_raw):
                logger.info(f"Sample {sample_id}: metadata is empty, skipping")
                continue

            if isinstance(metadata_raw, str):
                try:
                    metadata_dict = json.loads(metadata_raw)
                except JSONDecodeError:
                    logger.info(f"Sample {sample_id}: failed to parse metadata, skipping")
                    continue
            elif isinstance(metadata_raw, dict):
                metadata_dict = metadata_raw
            else:
                logger.info(f"Sample {sample_id}: invalid metadata format, skipping")
                continue

            answer = metadata_dict.get("answer")
            if self._is_empty_value(answer):
                logger.info(f"Sample {sample_id}: answer is empty in metadata, skipping")
                continue

            ground_truth_str = str(answer).strip()

            result_item = {
                "prompt": formatted_prompt,
                "ground_truth": ground_truth_str,
                "metadata": self._make_metadata_serializable(idx, metadata_dict),
            }

            result[sample_id] = result_item

        logger.info(f"Loaded {len(result)} samples for {self.task_name}")

        return result

    def _tokenize_prompt(self, prompt: str) -> list[int]:
        """Tokenize prompt text for length normalization."""
        tokenized = self.tokenizer(prompt, add_special_tokens=False)
        token_ids = getattr(tokenized, "input_ids", None)
        if token_ids is None and isinstance(tokenized, dict):
            token_ids = tokenized.get("input_ids", [])
        return list(token_ids or [])

    def _enforce_prompt_token_length(self, prompt: str, sample_id: str) -> str:
        """Trim or repeat prompt tokens to match custom_input_len exactly."""
        if self.custom_input_len is None:
            return prompt

        target_len = int(self.custom_input_len)
        if target_len <= 0:
            raise ValueError("custom_input_len must be a positive integer")

        prompt_token_ids = self._tokenize_prompt(prompt)
        current_len = len(prompt_token_ids)

        if current_len == target_len:
            return prompt

        if current_len == 0:
            logger.warning(
                "Sample %s: prompt tokenized to empty input; cannot enforce custom_input_len=%s",
                sample_id,
                target_len,
            )
            return prompt

        if current_len < target_len:
            repeat_count = (target_len + current_len - 1) // current_len
            adjusted_token_ids = (prompt_token_ids * repeat_count)[:target_len]
            action = "extended"
        else:
            adjusted_token_ids = prompt_token_ids[:target_len]
            action = "trimmed"

        adjusted_prompt = self.tokenizer.decode(
            adjusted_token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        final_len = len(self._tokenize_prompt(adjusted_prompt))
        if final_len != target_len:
            logger.warning(
                "Sample %s: requested custom_input_len=%s, got %s after %s",
                sample_id,
                target_len,
                final_len,
                action,
            )
        else:
            logger.debug(
                "Sample %s: prompt %s to %s tokens",
                sample_id,
                action,
                target_len,
            )

        return adjusted_prompt

    def _make_metadata_serializable(
        self,
        idx: Any,
        metadata_dict: dict,
    ) -> dict:
        """Convert metadata to JSON-serializable format"""
        del metadata_dict["answer"]

        metadata = {
            "row_index": int(idx) if hasattr(idx, "__int__") else str(idx),
            **metadata_dict,
        }

        return metadata
