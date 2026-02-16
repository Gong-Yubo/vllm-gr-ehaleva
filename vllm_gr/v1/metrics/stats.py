# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.metrics.stats import RequestStateStats as BaseRequestStateStats
from dataclasses import dataclass


@dataclass
class RequestStateStats(BaseRequestStateStats):
    """Stats that need to be tracked across delta updates."""

    # Beam search overhead
    beam_search_overhead: float = 0.0
    beam_search_decode_time: float = 0.0
