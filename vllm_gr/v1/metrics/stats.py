# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from vllm.v1.metrics.stats import RequestStateStats as BaseRequestStateStats


@dataclass
class RequestStateStats(BaseRequestStateStats):
    """Stats that need to be tracked across delta updates."""

    # Beam search overhead
    beam_search_overhead: float = 0.0
    beam_search_decode_time: float = 0.0
    prefill_time: float = 0.0
    decode_time: float = 0.0
    num_generation_tokens: int = 0
