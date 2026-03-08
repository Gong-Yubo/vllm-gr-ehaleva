import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest

from vllm_gr.entrypoints.openai.serving_models import Catalog


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()

    def convert_tokens_to_ids(seq):
        # Mock behavior: convert string tokens to integers
        # Assuming seq is a list of strings like ["1", "2"]
        if isinstance(seq, list):
            return [int(x) for x in seq]
        # Fallback if seq is a single string
        return int(seq)

    tokenizer.convert_tokens_to_ids.side_effect = convert_tokens_to_ids
    return tokenizer


def test_catalog_functionality(mock_tokenizer):
    # Define catalog data
    # 3 sequences:
    # [1, 2, 3, 4, 5]
    # [1, 2, 3, 6, 7]
    # [8, 9, 10, 11, 12]
    catalog_data = [
        ["1", "2", "3", "4", "5"],
        ["1", "2", "3", "6", "7"],
        ["8", "9", "10", "11", "12"],
    ]

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        json.dump(catalog_data, f)
        catalog_path = f.name

    try:
        catalog = Catalog(mock_tokenizer)
        catalog.load(catalog_path)

        # Case 1: Empty prefix -> returns start tokens of all sequences
        # Expected: {1, 8}
        assert set(catalog.valid([])) == {1, 8}

        # Case 2: Prefix [1, 2, 3] -> returns {4, 6}
        assert set(catalog.valid([1, 2, 3])) == {4, 6}

        # Case 3: Prefix [1, 2, 3, 4] -> returns {5}
        assert set(catalog.valid([1, 2, 3, 4])) == {5}

        # Case 4: Full sequence [1, 2, 3, 4, 5] -> returns []
        assert catalog.valid([1, 2, 3, 4, 5]) == set()

        # Case 5: Invalid start token
        assert catalog.valid([99]) == set()
        # Case 6: Valid start, invalid continuation
        assert catalog.valid([1, 99]) == set()

    finally:
        if os.path.exists(catalog_path):
            os.remove(catalog_path)
