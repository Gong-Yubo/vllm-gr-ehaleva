# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import msgspec
from vllm.v1.engine import EngineCoreRequestType

from vllm_gr.v1.engine.types import MegaRequestStepUpdate

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

    # DP path: ensure stats task is alive (lazy init for DPAsyncMPClient).
    if hasattr(self, "_ensure_stats_update_task"):
        self._ensure_stats_update_task()

    for request in requests:
        request.client_index = self.client_index
        if hasattr(self, "current_wave"):
            request.current_wave = self.current_wave

    # Route to the correct engine based on data_parallel_rank (DP case).
    # get_core_engine_for_request uses request.data_parallel_rank when set,
    # otherwise falls back to load-balancing. It also registers the first
    # request in reqs_in_flight; we register the rest manually.
    if hasattr(self, "get_core_engine_for_request"):
        chosen_engine = self.get_core_engine_for_request(requests[0])
        if hasattr(self, "reqs_in_flight"):
            for request in requests[1:]:
                self.reqs_in_flight[request.request_id] = chosen_engine
        to_await = self._send_input(EngineCoreRequestType.ADD_BATCH, requests, chosen_engine)
        # Notify coordinator when engines are idle (mirrors DPAsyncMPClient.add_request_async).
        if hasattr(self, "first_req_send_socket") and not self.engines_running:
            req_msg = msgspec.msgpack.encode(("FIRST_REQ", chosen_engine))
            await self.first_req_send_socket.send(req_msg)
        await to_await
    else:
        await self._send_input(EngineCoreRequestType.ADD_BATCH, requests)
    self._ensure_output_queue_task()


# ---------------------------------------------------------------------------
# AsyncMPClient.mega_request_step_update_async  (MEGA_REQUEST_STEP_UPDATE path)
# ---------------------------------------------------------------------------


async def mega_request_step_update_async(self, update: MegaRequestStepUpdate) -> None:
    """Send MEGA_REQUEST_STEP_UPDATE to EngineCore.

    Transmits a single logical message for all branches of a decode step.
    """
    update.client_index = self.client_index
    if hasattr(self, "current_wave"):
        update.current_wave = self.current_wave

    engine = None
    if (hasattr(self, "get_core_engine_for_request")
            and update.data_parallel_rank is not None):
        engine = self.core_engines[update.data_parallel_rank]
        if hasattr(self, "reqs_in_flight"):
            for child_id in update.child_beam_ids:
                self.reqs_in_flight[child_id] = engine
            for pruned_id in update.pruned_ids:
                self.reqs_in_flight.pop(pruned_id, None)

    to_await = self._send_input(EngineCoreRequestType.MEGA_REQUEST_STEP_UPDATE, update, engine)
    if hasattr(self, "first_req_send_socket") and not self.engines_running:
        req_msg = msgspec.msgpack.encode(("FIRST_REQ", engine))
        await self.first_req_send_socket.send(req_msg)
    await to_await
    self._ensure_output_queue_task()
