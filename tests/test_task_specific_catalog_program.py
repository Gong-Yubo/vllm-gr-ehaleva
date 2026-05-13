# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.open_one_rec.tools import (
    create_task_catalog as create_task_catalog_module,
)


def test_create_task_specific_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = "interactive"
    parquet_path = tmp_path / task / f"{task}_test.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.touch()

    df = pd.DataFrame(
        {
            "metadata": [
                json.dumps({"answer": "<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>"}),
                json.dumps(
                    {
                        "answer": [
                            "<|sid_begin|><s_a_4><s_b_5><s_c_6><|sid_end|>",
                            "<|sid_begin|><s_a_4><s_b_5><s_c_6><|sid_end|>",
                        ]
                    }
                ),
                json.dumps({"answer": "no_sid_here"}),
            ]
        }
    )

    def mock_read_parquet(path: Path) -> pd.DataFrame:
        assert Path(path) == parquet_path
        return df

    monkeypatch.setattr(create_task_catalog_module.pd, "read_parquet", mock_read_parquet)

    output_path = tmp_path / "interactive_catalog.json"
    output, count = create_task_catalog_module.create_task_sid_catalog(
        task_name=task,
        split="test",
        data_dir=str(tmp_path),
        output_path=str(output_path),
        deduplicate=True,
    )

    catalog = json.loads(Path(output).read_text(encoding="utf-8"))
    expected = [
        ["<|sid_begin|>", "<s_a_1>", "<s_b_2>", "<s_c_3>", "<|sid_end|>"],
        ["<|sid_begin|>", "<s_a_4>", "<s_b_5>", "<s_c_6>", "<|sid_end|>"],
    ]

    assert output == output_path
    assert count == 2
    assert catalog == expected
