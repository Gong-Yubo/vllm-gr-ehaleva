# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patches for AsyncMPClient / AsyncLLM to support
ADD_BATCH and BEAM_FORK."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger

from vllm_gr.v1.engine.async_llm import (
    _add_requests_batch_fn,
    beam_step_update_fn,
    mega_request_step_update_fn,
    prepare_request_fn,
    register_beam_output_fn,
)
from vllm_gr.v1.engine.core_client import (
    add_requests_async,
    beam_step_update_async,
    mega_request_step_update_async,
)

logger = init_logger(__name__)

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# apply_batch_fork_patches — wire ADD_BATCH / BEAM_FORK into client classes
# ---------------------------------------------------------------------------


def apply_batch_fork_patches():
    """Monkey-patch AsyncMPClient and AsyncLLM with ADD_BATCH/BEAM_FORK
    methods. Must be called after the enum members exist (i.e. after
    _add_enum_member has been called for ADD_BATCH and BEAM_FORK in the
    parent process)."""
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.core_client import AsyncMPClient

    from vllm_gr.v1.engine.engine_core_patch import _add_enum_member

    # Ensure enum members exist in parent process
    _add_enum_member("ADD_BATCH", b"\x05")
    _add_enum_member("BEAM_STEP_UPDATE", b"\x07")
    _add_enum_member("MEGA_REQUEST_STEP_UPDATE", b"\x08")

    # Patch AsyncMPClient
    AsyncMPClient.add_requests_async = add_requests_async
    AsyncMPClient.beam_step_update_async = beam_step_update_async
    AsyncMPClient.mega_request_step_update_async = mega_request_step_update_async

    # Patch AsyncLLM
    AsyncLLM.prepare_request = prepare_request_fn
    AsyncLLM._add_requests_batch = _add_requests_batch_fn
    AsyncLLM.register_beam_output = register_beam_output_fn
    AsyncLLM.beam_step_update = beam_step_update_fn
    AsyncLLM.mega_request_step_update = mega_request_step_update_fn



