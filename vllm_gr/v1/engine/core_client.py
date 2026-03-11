# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.engine import EngineCoreRequestType

from vllm_gr.v1.engine.types import BeamForkRequest

# ---------------------------------------------------------------------------
# AsyncMPClient.add_requests_async  (new method — ADD_BATCH path)
# ---------------------------------------------------------------------------


async def add_requests_async(self, requests, force_batch=False):
    """Send multiple requests to EngineCore.

    If single and not forced, delegate to add_request_async.
    Otherwise send as ADD_BATCH.
    """
    if not requests:
        return
    if len(requests) == 1 and not force_batch:
        return await self.add_request_async(requests[0])
    for request in requests:
        request.client_index = self.client_index
    await self._send_input(EngineCoreRequestType.ADD_BATCH, requests)
    self._ensure_output_queue_task()


# ---------------------------------------------------------------------------
# AsyncMPClient.beam_fork_async  (new method — BEAM_FORK path)
# ---------------------------------------------------------------------------


async def beam_fork_async(self, fork_request: BeamForkRequest) -> None:
    """Send BEAM_FORK to EngineCore."""
    fork_request.client_index = self.client_index
    await self._send_input(EngineCoreRequestType.BEAM_FORK, fork_request)
    self._ensure_output_queue_task()
