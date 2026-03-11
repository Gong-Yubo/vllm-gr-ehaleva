# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared types for vllm_gr engine patches.

Defines BeamForkRequest.
"""

from __future__ import annotations

from collections.abc import Mapping

import msgspec
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams

# ---------------------------------------------------------------------------
# BeamForkRequest — lightweight beam fork (ADD_BATCH + BEAM_FORK path)
# ---------------------------------------------------------------------------


class BeamForkRequest(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """Lightweight beam search fork: clone parent state + append 1 token.

    Parallel arrays: parent_ids[i] -> child_ids[i] with token_ids[i] appended.
    A parent can appear multiple times (one beam forking into multiple children).
    """

    parent_ids: list[str]
    child_ids: list[str]
    token_ids: list[int]
    abort_ids: list[str]
    sampling_params: SamplingParams
    eos_token_id: int | None = None
    client_index: int = 0
    current_wave: int = 0
    priority: int = 0
    lora_request: LoRARequest | None = None
    cache_salt: str | None = None
    trace_headers: Mapping[str, str] | None = None
