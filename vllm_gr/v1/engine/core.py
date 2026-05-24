# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patches for EngineCore / EngineCoreProc to support
ADD_BATCH and BEAM_FORK."""

from __future__ import annotations

import time
from functools import partial
from typing import Any

from vllm.logger import init_logger
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequestType,
    FinishReason,
)

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# _cache_beam_request / _handle_beam_fork  (ported from source core.py)
# ---------------------------------------------------------------------------


def _cache_beam_request(self, request) -> None:
    """Cache Request state for later beam forking.

    Thread-safe: uses beam_cache_lock to prevent data races.
    """
    with self.beam_cache_lock:
        self.beam_cache[request.request_id] = {
            "all_token_ids": list(request._all_token_ids),
            "block_hashes": list(request.block_hashes),
            "eos_token_id": request.eos_token_id,
            "lora_request": request.lora_request,
            "cache_salt": request.cache_salt,
            "mm_features": request.mm_features,
            "prompt_embeds": request.prompt_embeds,
            "num_prompt_tokens": request.num_prompt_tokens,
        }


def _handle_mega_request_step_update(self, update) -> None:
    """Apply a grouped step update for all active branches with batched locking."""

    from vllm.v1.request import Request
    from vllm.v1.engine import EngineCoreOutputs, EngineCoreOutput, FinishReason

    session_id = update.session_id
    B = update.beam_width

    # Validate parallel arrays
    if len(update.beam_tokens) != B or len(update.parent_beam_ids) != B or len(update.child_beam_ids) != B:
        logger.error("MEGA_REQUEST_STEP_UPDATE: session=%s length mismatch", session_id)
        for child_id in update.child_beam_ids:
            self.output_queue.put_nowait((
                update.client_index,
                EngineCoreOutputs(
                    engine_index=self.engine_index,
                    finished_requests={child_id},
                    outputs=[EngineCoreOutput(request_id=child_id, new_token_ids=[], finish_reason=FinishReason.ERROR)]
                )
            ))
        return

    with self.beam_cache_lock:
        session_cached = self.beam_cache.get(session_id)
        if session_cached is None and update.parent_beam_ids:
            session_cached = self.beam_cache.get(update.parent_beam_ids[0])

        if session_cached is not None:
            parent_states = [self.beam_cache.get(pid, session_cached) for pid in update.parent_beam_ids]

    if session_cached is None:
        logger.error("MEGA_REQUEST_STEP_UPDATE: session %s not in cache", session_id)
        for child_id in update.child_beam_ids:
            self.output_queue.put_nowait((
                update.client_index,
                EngineCoreOutputs(
                    engine_index=self.engine_index,
                    finished_requests={child_id},
                    outputs=[EngineCoreOutput(request_id=child_id, new_token_ids=[], finish_reason=FinishReason.ERROR)]
                )
            ))
        return

    arrival = time.time()
    requests_to_cache = []
    inputs_to_push = []

    # Parallel loop calculation (Safe to process heavy hashing outside the lock)
    for i, (parent_id, child_id, token_id, parent_state) in enumerate(zip(
        update.parent_beam_ids, update.child_beam_ids, update.beam_tokens, parent_states
    )):
        child_token_ids = parent_state["all_token_ids"] + [token_id]

        req = Request(
            request_id=child_id,
            prompt_token_ids=child_token_ids,
            sampling_params=update.sampling_params,
            pooling_params=None,
            eos_token_id=(
                update.eos_token_id if update.eos_token_id is not None else session_cached["eos_token_id"]
            ),
            client_index=update.client_index,
            arrival_time=arrival,
            lora_request=(update.lora_request or session_cached["lora_request"]),
            cache_salt=update.cache_salt or session_cached["cache_salt"],
            priority=update.priority,
            trace_headers=update.trace_headers,
            prompt_embeds=session_cached["prompt_embeds"],
            mm_features=session_cached["mm_features"],
            block_hasher=None,  # Skip O(N) hash computation
        )

        # Clone parent's block hashes and incrementally compute new ones
        req.block_hashes = list(parent_state["block_hashes"])
        if self.request_block_hasher is not None:
            req.get_hash_new_full_blocks = partial(self.request_block_hasher, req)
            req.block_hashes.extend(req.get_hash_new_full_blocks())

        # Metadata Enrichment
        req.is_mega_beam = True
        req.mega_beam_width = B
        req.mega_beam_index = i
        req.prefix_len = update.prefix_len

        requests_to_cache.append(req)
        inputs_to_push.append((EngineCoreRequestType.ADD, (req, update.current_wave)))

    # Optimization 2: Bulk Write Phase (Single Lock Commit)
    with self.beam_cache_lock:
        for req in requests_to_cache:
            self.beam_cache[req.request_id] = {
                "all_token_ids": req.prompt_token_ids,
                "block_hashes": req.block_hashes,
                "eos_token_id": req.eos_token_id,
                "lora_request": req.lora_request,
                "cache_salt": req.cache_salt,
                "prompt_embeds": req.prompt_embeds,
                "mm_features": req.mm_features,
            }
        self.beam_cache[session_id] = session_cached    
        if update.pruned_ids:
            for pid in update.pruned_ids:
                self.beam_cache.pop(pid, None)
        # If beam_width is 0, this is an explicit tear-down message for the session
        if update.beam_width == 0:
            self.beam_cache.pop(session_id, None)
            # Defensive check: clear out the session key if it was masquerading as a beam
            if update.pruned_ids:
                for pid in update.pruned_ids:
                    self.beam_cache.pop(pid, None)

    # Non-blocking concurrent queue pushes
    for item in inputs_to_push:
        self.input_queue.put_nowait(item)

    if update.pruned_ids:
        for abort_id in update.pruned_ids:
            self.aborts_queue.put_nowait(abort_id)


# ---------------------------------------------------------------------------
# EngineCoreProc.process_input_sockets replacement
# ---------------------------------------------------------------------------


def process_input_sockets(
    self,
    input_addresses,
    coord_input_address,
    identity,
    ready_event,
):
    """Replacement for EngineCoreProc.process_input_sockets with
    ADD_BATCH and BEAM_FORK decoders."""
    from contextlib import ExitStack

    import zmq
    from vllm.utils.network_utils import make_zmq_socket
    from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType
    from vllm.v1.serial_utils import MsgpackDecoder

    from vllm_gr.v1.engine.types import MegaRequestStepUpdate

    # Msgpack serialization decoding.
    add_request_decoder = MsgpackDecoder(EngineCoreRequest)
    add_batch_decoder = MsgpackDecoder(list[EngineCoreRequest])
    mega_request_decoder = MsgpackDecoder(MegaRequestStepUpdate)
    generic_decoder = MsgpackDecoder()

    with ExitStack() as stack, zmq.Context() as ctx:
        input_sockets = [
            stack.enter_context(
                make_zmq_socket(ctx, input_address, zmq.DEALER, identity=identity, bind=False)
            )
            for input_address in input_addresses
        ]
        if coord_input_address is None:
            coord_socket = None
        else:
            coord_socket = stack.enter_context(
                make_zmq_socket(
                    ctx,
                    coord_input_address,
                    zmq.XSUB,
                    identity=identity,
                    bind=False,
                )
            )
            # Send subscription message to coordinator.
            coord_socket.send(b"\x01")

        # Register sockets with poller.
        poller = zmq.Poller()
        for input_socket in input_sockets:
            input_socket.send(b"")
            poller.register(input_socket, zmq.POLLIN)

        if coord_socket is not None:
            assert coord_socket.recv() == b"READY"
            poller.register(coord_socket, zmq.POLLIN)

        ready_event.set()
        del ready_event
        while True:
            for input_socket, _ in poller.poll():
                type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                request: Any
                if request_type == EngineCoreRequestType.ADD:
                    req: EngineCoreRequest = add_request_decoder.decode(data_frames)
                    try:
                        request = self.preprocess_add_request(req)
                    except Exception:
                        self._handle_request_preproc_error(req)
                        continue

                elif request_type == EngineCoreRequestType.ADD_BATCH:
                    batch: list[EngineCoreRequest] = add_batch_decoder.decode(data_frames)
                    for req in batch:
                        try:
                            request = self.preprocess_add_request(req)
                        except Exception:
                            self._handle_request_preproc_error(req)
                            continue
                        self._cache_beam_request(request[0])
                        self.input_queue.put_nowait((EngineCoreRequestType.ADD, request))
                    continue

                elif request_type == EngineCoreRequestType.MEGA_REQUEST_STEP_UPDATE:
                    mega_update: MegaRequestStepUpdate = mega_request_decoder.decode(data_frames)
                    self._handle_mega_request_step_update(mega_update)
                    continue

                else:
                    request = generic_decoder.decode(data_frames)

                    if request_type == EngineCoreRequestType.ABORT:
                        self.aborts_queue.put_nowait(request)

                self.input_queue.put_nowait((request_type, request))
