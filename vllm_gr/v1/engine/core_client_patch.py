# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patches for AsyncMPClient / AsyncLLM to support
ADD_BATCH and BEAM_FORK."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger

from vllm_gr.v1.engine.async_llm import (
    _add_requests_batch_fn,
    beam_fork_fn,
    prepare_request_fn,
    register_beam_output_fn,
)
from vllm_gr.v1.engine.core_client import add_requests_async, beam_fork_async

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
    from vllm.v1.engine.output_processor import OutputProcessor

    from vllm_gr.v1.engine.engine_core_patch import _add_enum_member

    # Ensure enum members exist in parent process
    _add_enum_member("ADD_BATCH", b"\x05")
    _add_enum_member("BEAM_FORK", b"\x06")

    # Patch AsyncMPClient
    AsyncMPClient.add_requests_async = add_requests_async
    AsyncMPClient.beam_fork_async = beam_fork_async

    # Patch AsyncLLM
    AsyncLLM.prepare_request = prepare_request_fn
    AsyncLLM._add_requests_batch = _add_requests_batch_fn
    AsyncLLM.register_beam_output = register_beam_output_fn
    AsyncLLM.beam_fork = beam_fork_fn

    _original_process_outputs = OutputProcessor.process_outputs

    def patched_process_outputs(self, engine_core_outputs, engine_core_timestamp=None, iteration_stats=None):
        import time
        if not hasattr(self, "_total_process_outputs_time"):
            self._total_process_outputs_time = 0.0
        start_time = time.perf_counter()
        res = _original_process_outputs(self, engine_core_outputs, engine_core_timestamp, iteration_stats)
        cur_time = time.perf_counter() - start_time
        self._total_process_outputs_time += cur_time
        logger.info("OutputProcessor.process_outputs took %.2f ms (Total: %.2f ms) [Frontend]", cur_time * 1000, self._total_process_outputs_time * 1000)
        return res

    OutputProcessor.process_outputs = patched_process_outputs
