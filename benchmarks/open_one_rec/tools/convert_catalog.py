# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Convert OpenOneRec sid2pid mapping into vllm-gr catalog format."""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
BASE = 8192


def _build_sid_tokens(key: int) -> list[str]:
    """Decode integer SID key into tokenized SID sequence."""
    val_a = key // (BASE * BASE)
    val_b = (key % (BASE * BASE)) // BASE
    val_c = key % BASE

    return [
        "<|sid_begin|>",
        f"<s_a_{val_a}>",
        f"<s_b_{val_b}>",
        f"<s_c_{val_c}>",
        "<|sid_end|>",
    ]


def process_json(input_file_path: str, output_file_path: str) -> int:
    """Convert sid2pid JSON keys to catalog format and write output."""
    input_path = Path(input_file_path)
    output_path = Path(output_file_path)

    try:
        with input_path.open("r", encoding="utf-8") as file:
            data: Any = json.load(file)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Input file not found: {input_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Failed to decode JSON from: {input_path}") from error

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {input_path}, got {type(data).__name__} instead")

    result_list: list[list[str]] = []

    for key_str in data:
        try:
            key = int(key_str)
        except (TypeError, ValueError):
            LOGGER.warning("Skipping non-integer key: %s", key_str)
            continue

        result_list.append(_build_sid_tokens(key))

    output_dir = output_path.parent
    if str(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result_list, file, indent=2, ensure_ascii=False)

    return len(result_list)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert data/sid2pid.json to vllm-gr catalog format."
    )
    parser.add_argument(
        "--dataset-path",
        "-d",
        default="data",
        help="Dataset root path containing sid2pid.json. Default: data.",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        default="catalog.json",
        help="Output catalog JSON path. Default: catalog.json.",
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

    input_path = Path(args.dataset_path) / "sid2pid.json"
    try:
        count = process_json(str(input_path), args.output_path)
    except (FileNotFoundError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1

    LOGGER.info("Saved %d catalog entries to %s", count, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
