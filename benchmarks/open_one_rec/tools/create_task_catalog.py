# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Create a reduced SID catalog JSON from OpenOneRec task parquet data."""

import argparse
import json
import logging
import os
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import regex as re

LOGGER = logging.getLogger(__name__)

SID_BEGIN_TOKEN = "<|sid_begin|>"
SID_END_TOKEN = "<|sid_end|>"
SID_BLOCK_PATTERN = re.compile(r"<\|sid_begin\|>(.*?)<\|sid_end\|>", re.DOTALL)
SID_TOKEN_PATTERN = re.compile(r"<s_[^>\s]+>")


def _is_empty_value(value: Any) -> bool:
    """Check if a value is None, NaN, or empty."""
    if value is None:
        return True

    if isinstance(value, float):
        try:
            return pd.isna(value)
        except (TypeError, ValueError):
            return False

    if isinstance(value, str):
        return len(value.strip()) == 0

    try:
        if hasattr(value, "__len__"):
            return len(value) == 0
    except (TypeError, ValueError):
        pass

    return False


def _maybe_parse_json(value: Any) -> Any:
    """Parse JSON if input is a JSON-like string, otherwise return original value."""
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text or text[0] not in "[{":
        return value

    try:
        return json.loads(text)
    except JSONDecodeError:
        return value


def _extract_sid_sequences(answer: Any) -> list[list[str]]:
    """Extract all SID sequences in reduced_catalog format from answer payload."""
    answer = _maybe_parse_json(answer)
    if _is_empty_value(answer):
        return []

    if isinstance(answer, (list, tuple, set)):
        sequences: list[list[str]] = []
        for item in answer:
            sequences.extend(_extract_sid_sequences(item))
        return sequences

    if isinstance(answer, dict):
        for key in ("answer", "answers", "sid", "sids", "result", "results"):
            if key in answer:
                return _extract_sid_sequences(answer[key])
        return []

    text = str(answer)
    sequences: list[list[str]] = []

    for sid_block in SID_BLOCK_PATTERN.findall(text):
        sid_tokens = SID_TOKEN_PATTERN.findall(sid_block)
        if sid_tokens:
            sequences.append([SID_BEGIN_TOKEN, *sid_tokens, SID_END_TOKEN])

    if sequences:
        return sequences

    sid_tokens = SID_TOKEN_PATTERN.findall(text)
    if sid_tokens:
        return [[SID_BEGIN_TOKEN, *sid_tokens, SID_END_TOKEN]]

    return []


def _extract_answer_from_row(row: pd.Series, columns: pd.Index) -> Any:
    """Extract answer field from metadata first, then fallback to top-level answer column."""
    answer = None

    if "metadata" in columns:
        metadata_raw = row.get("metadata")
        metadata_raw = _maybe_parse_json(metadata_raw)
        if isinstance(metadata_raw, dict):
            answer = metadata_raw.get("answer")

    if _is_empty_value(answer) and "answer" in columns:
        answer = row.get("answer")

    return answer


def _resolve_data_file(task_name: str, split: str, data_dir: str) -> Path:
    """Resolve parquet path for a task split."""
    file_name = f"{task_name}_{split}.parquet"
    candidates = [
        Path(data_dir) / task_name / file_name,
        Path(data_dir) / file_name,
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Data file not found for task '{task_name}' and split '{split}'. "
        f"Checked paths: {[str(path) for path in candidates]}"
    )


def create_task_sid_catalog(
    task_name: str,
    split: str = "test",
    data_dir: str = "data",
    output_path: Optional[str] = None,
    deduplicate: bool = True,
) -> tuple[Path, int]:
    """Load all correct SID answers of one task and write catalog JSON."""
    data_file = _resolve_data_file(task_name=task_name, split=split, data_dir=data_dir)
    data_frame = pd.read_parquet(data_file)

    catalog: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    for _, row in data_frame.iterrows():
        answer = _extract_answer_from_row(row=row, columns=data_frame.columns)
        for sid_sequence in _extract_sid_sequences(answer):
            sequence_key = tuple(sid_sequence)

            if deduplicate and sequence_key in seen:
                continue

            if deduplicate:
                seen.add(sequence_key)
            catalog.append(sid_sequence)

    output = Path(output_path) if output_path else Path(f"{task_name}_{split}_sid_catalog.json")
    output_parent = output.parent
    if str(output_parent):
        os.makedirs(output_parent, exist_ok=True)

    with output.open("w", encoding="utf-8") as file:
        json.dump(catalog, file, ensure_ascii=False, indent=2)

    return output, len(catalog)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a reduced SID catalog JSON for one OpenOneRec task from task parquet data."
        )
    )
    parser.add_argument(
        "--task-type",
        "-t",
        required=True,
        help="Task type, e.g. interactive, video, label_cond, product, ad.",
    )
    parser.add_argument(
        "--dataset-path",
        "-d",
        default="data",
        help="Data root directory containing task parquet files. Default: data.",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        default=None,
        help=(
            "Output JSON path. Default: <task_name>_<split>_sid_catalog.json "
            "in current working directory."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level. Default: INFO.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    try:
        output, count = create_task_sid_catalog(
            task_name=args.task_type,
            split="test",
            data_dir=args.dataset_path,
            output_path=args.output_path,
            deduplicate=True,
        )
    except (FileNotFoundError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1

    LOGGER.info("Saved %d SID sequences to %s", count, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
