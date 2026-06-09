# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import copy
import time
from collections.abc import AsyncGenerator, Mapping
from typing import Any

import numpy as np
from vllm.beam_search import BeamSearchSequence
from vllm.entrypoints.openai.protocol import VLLMValidationError
from vllm.inputs.data import PromptType, TokensPrompt
from vllm.inputs.parse import is_explicit_encoder_decoder_prompt
from vllm.logger import init_logger
from vllm.logprobs import FlatLogprobs
from vllm.lora.request import LoRARequest
from vllm.multimodal import MultiModalDataDict
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.utils import random_uuid

from vllm_gr.logprobs import extract_and_dedup_flat_logprobs, reconstruct_beam_logprobs
from vllm_gr.v1.metrics.stats import RequestStateStats
from vllm_gr.v1.engine.types import MegaRequestStepUpdate


logger = init_logger(__name__)

_dp_rank_counter = 0
_dp_rank_lock = asyncio.Lock()


# return rank for next beam search request in a round-robin manner across the data parallel ranks
async def _next_data_parallel_rank(data_parallel_size: int) -> int:
    global _dp_rank_counter

    async with _dp_rank_lock:
        rank = _dp_rank_counter % data_parallel_size
        _dp_rank_counter += 1
        return rank


async def _collect_beam_result(q):
    """Drain output queue until finished."""
    while True:
        out = await q.get()
        if out.finished:
            return out


async def _gather_beam_results(queues):
    """Gather results from multiple output queues."""
    tasks = [asyncio.create_task(_collect_beam_result(q)) for q in queues]
    return list(await asyncio.gather(*tasks))


async def _add_batch_step(
    engine_client,
    prompts_batch,
    lora_req_batch,
    request_id_batch,
    beam_search_params,
    use_beam_fork,
    trace_headers,
    priority=0,
    data_parallel_rank: int | None = None,
):
    """ADD_BATCH path: prepare and batch-send requests (step 0)."""
    queues = []
    engine_core_requests = []
    for i, (individual_prompt, lora_req) in enumerate(zip(prompts_batch, lora_req_batch)):
        request_id_item = f"{request_id_batch}-beam-{i}"
        q, ec_req = engine_client.prepare_request(
            request_id_item,
            individual_prompt,
            beam_search_params,
            lora_request=lora_req,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
        )
        queues.append(q)
        engine_core_requests.append(ec_req)

    await engine_client._add_requests_batch(engine_core_requests, use_batch_message=use_beam_fork)

    internal_ids = [r.request_id for r in engine_core_requests] if use_beam_fork else []
    output = await _gather_beam_results(queues)
    return output, internal_ids

async def _mega_request_step(
    engine_client,
    session_id: str,
    fork_info,
    prev_beam_internal_ids,
    all_beams,
    request_id_batch,
    beam_search_params,
    eos_token_id,
    lora_request,
    trace_headers,
    prefix_len: int,
    priority=0,
    data_parallel_rank: int | None = None,
):
    """MEGA_REQUEST path: single logical message update for all active branches."""
    parent_beam_ids = []
    child_beam_ids = []
    beam_tokens = []
    W = len(fork_info)

    # Map the current generation step's routing array
    for new_idx, (parent_beam_idx, tok) in enumerate(fork_info):
        parent_id = prev_beam_internal_ids[parent_beam_idx]
        child_id = f"{request_id_batch}-beam-{new_idx}"
        parent_beam_ids.append(parent_id)
        child_beam_ids.append(child_id)
        
        full_tokens = all_beams[new_idx].tokens
        beam_tokens.append(full_tokens[prefix_len:])

    # -------------------------------------------------------------------------
    # OPTIMIZATION: Declare true active beam sizing directly to the core sampler
    # -------------------------------------------------------------------------
    decode_sampling_params = copy.copy(beam_search_params)
    decode_sampling_params.n = W  # Explicitly tell the engine we have W beams

    # Listen exclusively to the parent session_id stream queue
    q = engine_client.register_beam_output(
        session_id,
        all_beams[0].tokens,
        decode_sampling_params,
        eos_token_id=eos_token_id,
        lora_request=lora_request,
        trace_headers=trace_headers,
        priority=priority,
        data_parallel_rank=data_parallel_rank,
    )

    # Detect branches that were pruned during top-K filtering
    used_parents = set(parent_beam_ids)
    pruned_ids = [pid for pid in prev_beam_internal_ids if pid not in used_parents]

    # Dispatch the consolidated persistent update to the client
    await engine_client.mega_request_step_update(
        MegaRequestStepUpdate(
            session_id=session_id,
            parent_beam_ids=parent_beam_ids,
            child_beam_ids=child_beam_ids,
            beam_tokens=beam_tokens,
            pruned_ids=pruned_ids,
            prefix_len=prefix_len,
            beam_width=W,
            sampling_params=decode_sampling_params,
            eos_token_id=eos_token_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
            data_parallel_rank=data_parallel_rank,
            priority=priority,
        )
    )

    # Await the singular unified engine execution response block
    single_output = await _collect_beam_result(q)

    # -------------------------------------------------------------------------
    # STRIDE INJECTION: Inject layout structural attributes directly onto the object
    # -------------------------------------------------------------------------
    if single_output.outputs and single_output.outputs[0].logprobs is not None:
        flat_lps = single_output.outputs[0].logprobs
        # Stash the stride length and beam width directly on the instance annotations
        flat_lps.stride = len(flat_lps.token_ids) // W if flat_lps.token_ids else (W + 1)
        flat_lps.beam_width = W

    # Return the singular raw output instance wrapped as a single-element list 
    # to maintain backward-compatible pipeline execution with the loop below
    return [single_output], child_beam_ids

async def _mega_request_cleanup(engine_client, session_id, final_ids, rank):
    """Signals the core engine to fully purge all remaining session cache allocations."""
    if not final_ids:
        return
    try:
        await engine_client.mega_request_step_update(
            MegaRequestStepUpdate(
                session_id=session_id,
                parent_beam_ids=[],
                child_beam_ids=[],
                beam_tokens=[],
                pruned_ids=final_ids,
                prefix_len=0,
                beam_width=0,
                sampling_params=SamplingParams(),
                data_parallel_rank=rank,
            )
        )
    except Exception as e:
        logger.error("Failed to clear mega_request cache for session %s: %s", session_id, e)

async def beam_search(
    self,
    prompt: PromptType,
    request_id: str,
    params: BeamSearchParams,
    lora_request: LoRARequest | None = None,
    trace_headers: Mapping[str, str] | None = None,
) -> AsyncGenerator[RequestOutput, None]:
    generation_time: float = 0.0
    num_generation_tokens: int = 0
    beam_width = params.beam_width
    max_tokens = params.max_tokens
    ignore_eos = params.ignore_eos
    temperature = params.temperature
    begin_token = params.begin_token
    end_token = params.end_token

    MICROSECONDS = 1000000
    priority = int(time.perf_counter() * MICROSECONDS)
    rank = None
    if (
        hasattr(self.engine_client, "vllm_config")
        and self.engine_client.vllm_config.parallel_config is not None
    ):
        data_parallel_size = self.engine_client.vllm_config.parallel_config.data_parallel_size
        if data_parallel_size is not None and data_parallel_size > 1:
            rank = await _next_data_parallel_rank(data_parallel_size)

    include_stop_str_in_output = params.include_stop_str_in_output
    if beam_width == 0:
        raise VLLMValidationError("Beam width must be greater than 0", parameter="beam_width", value=0)
        
    input_processor = self.input_processor
    tokenizer = input_processor.tokenizer
    if tokenizer is None:
        raise VLLMValidationError("You cannot use beam search when `skip_tokenizer_init=True`", parameter="skip_tokenizer_init", value=True)

    eos_token_id: int = tokenizer.eos_token_id
    sid_begin_token_id: int | None = None
    if begin_token is not None:
        sid_begin_token_id = tokenizer.convert_tokens_to_ids(begin_token)
        
    sid_end_token_id: int | None = None
    if end_token is not None:
        sid_end_token_id = tokenizer.convert_tokens_to_ids(end_token)

    if is_explicit_encoder_decoder_prompt(prompt):
        raise NotImplementedError

    prompt_text = prompt if isinstance(prompt, str) else prompt.get("prompt")
    prompt_token_ids = [] if isinstance(prompt, str) else prompt.get("prompt_token_ids", [])
    multi_modal_data = None if isinstance(prompt, str) else prompt.get("multi_modal_data")
    mm_processor_kwargs = None

    tokenized_length = len(prompt_token_ids)
    logprobs_num = beam_width
    
    beam_search_params = SamplingParams(
        logprobs=logprobs_num,
        max_tokens=1,
        temperature=temperature,
        detokenize=False,
        flat_logprobs=True,
    )
    initial_tokens = list(prompt_token_ids)
    initial_logprobs = FlatLogprobs()
    if sid_begin_token_id is not None:
        initial_tokens.append(sid_begin_token_id)
        initial_logprobs.append_fast([sid_begin_token_id], [0.0], [None], [None])

    all_beams = [
        BeamSearchSequence(
            tokens=initial_tokens,
            cum_logprob=0,
            logprobs=initial_logprobs,
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            lora_request=lora_request,
        )
    ]
    all_beams[0]._lp_parent = None
    all_beams[0]._lp_step_data = None
    completed = []

    beam_search_start: int = time.perf_counter()
    pre_calc = 0 if sid_begin_token_id is None else 1
    if sid_end_token_id is not None:
        pre_calc += 1

    use_batch = hasattr(self.engine_client, "prepare_request")
    use_mega_request = hasattr(self.engine_client, "mega_request_step_update")

    prev_beam_internal_ids: list[str] = []
    fork_info: list[tuple[int, int]] | None = None
    prefill_id: str | None = None
    
    for token in range(max_tokens - pre_calc):
        if token == 1:
            beam_search_start = time.perf_counter()

        prompts_batch, lora_req_batch = zip(
            *[
                (
                    TokensPrompt(
                        prompt_token_ids=beam.tokens,
                        multi_modal_data=beam.multi_modal_data,
                        mm_processor_kwargs=beam.mm_processor_kwargs,
                    ),
                    beam.lora_request,
                )
                for beam in all_beams
            ]
        )
        if token > 0:
            num_generation_tokens += len(all_beams)
        request_id_batch = f"{request_id}-{random_uuid()}"

        catalog_task = None
        if self.models.catalog is not None:
            def get_valid_tokens_set(beam) -> set[int]:
                generated_tokens = beam.tokens[len(prompt_token_ids) :]
                return self.models.catalog.valid(generated_tokens)

            async def _run_catalog(coros):
                return list(await asyncio.gather(*coros))

            catalog_task = asyncio.create_task(
                _run_catalog([asyncio.to_thread(get_valid_tokens_set, beam) for beam in all_beams])
            )

        gen_start = time.perf_counter()

        # Execute Engine Step Pass
        if use_mega_request and fork_info is not None:
            output, prev_beam_internal_ids = await _mega_request_step(
                self.engine_client,
                session_id=prefill_id,
                fork_info=fork_info,
                prev_beam_internal_ids=prev_beam_internal_ids,
                all_beams=all_beams,
                request_id_batch=request_id_batch,
                beam_search_params=beam_search_params,
                eos_token_id=eos_token_id,
                lora_request=lora_request,
                trace_headers=trace_headers,
                prefix_len=len(initial_tokens),
                priority=priority,
                data_parallel_rank=rank,
            )
        else:
            output, new_ids = await _add_batch_step(
                self.engine_client,
                prompts_batch,
                lora_req_batch,
                request_id_batch,
                beam_search_params,
                use_mega_request,
                trace_headers,
                priority=priority,
                data_parallel_rank=rank,
            )
            if new_ids:
                prev_beam_internal_ids = new_ids
                if prefill_id is None:
                    prefill_id = new_ids[0]

        valid_tokens_sets = None
        if catalog_task is not None:
            valid_tokens_sets = await catalog_task
        if token > 0:
            generation_time += time.perf_counter() - gen_start
            
        new_beams = []
        all_beams_token_id = []
        all_beams_logprob = []
        beam_flat_cache: list[tuple | None] = []

        # -------------------------------------------------------------------------
        # STRIDE-AWARE METADATA EXTRACTION FOR GENERATED BEAMS
        # -------------------------------------------------------------------------
        if use_mega_request and fork_info is not None:
            # Under the optimized mode, 'output' contains exactly 1 combined RequestOutput instance
            mega_result = output[0]
            if mega_result.outputs[0].finish_reason == "error":
                await _mega_request_cleanup(self.engine_client, prefill_id, prev_beam_internal_ids, rank)
                yield RequestOutput(request_id=request_id, prompt=prompt_text, outputs=[CompletionOutput(index=0, text="", token_ids=[], cumulative_logprob=None, logprobs=None, finish_reason="error")], finished=True, prompt_token_ids=prompt_token_ids, prompt_logprobs=None)
                return

            flat = mega_result.outputs[0].logprobs
            if flat is not None:
                stride_K = getattr(flat, 'stride', logprobs_num + 1)
                
                # Directly unpack the linear arrays in single stride loops
                for b_idx in range(len(fork_info)):
                    current_beam = all_beams[b_idx]
                    start_offset = b_idx * stride_K
                    end_offset = start_offset + stride_K
                    
                    # Strip the duplicate item: skip the leading token for choice mapping
                    clean_start = start_offset + 1
                    
                    token_ids_pos = flat.token_ids[clean_start:end_offset]
                    logprobs_pos = flat.logprobs[clean_start:end_offset]
                    ranks_pos = flat.ranks[clean_start:end_offset] if getattr(flat, 'ranks', None) else None
                    decoded_pos = flat.decoded_tokens[clean_start:end_offset] if getattr(flat, 'decoded_tokens', None) else None
                    
                    if valid_tokens_sets is not None:
                        valid_tokens_set = valid_tokens_sets[b_idx]
                        logprobs_pos = [lp if tid in valid_tokens_set else -float("inf") for tid, lp in zip(token_ids_pos, logprobs_pos)]
                        
                    beam_flat_cache.append((token_ids_pos, logprobs_pos, ranks_pos, decoded_pos))
                    all_beams_token_id.extend(token_ids_pos)
                    all_beams_logprob.extend(current_beam.cum_logprob + lp for lp in logprobs_pos)
            else:
                beam_flat_cache = [None] * len(fork_info)
        else:
            # Baseline execution path for prefill / batch loops
            for i, result in enumerate(output):
                current_beam = all_beams[i]
                if result.outputs[0].finish_reason == "error":
                    yield RequestOutput(request_id=request_id, prompt=prompt_text, outputs=[CompletionOutput(index=0, text="", token_ids=[], cumulative_logprob=None, logprobs=None, finish_reason="error")], finished=True, prompt_token_ids=prompt_token_ids, prompt_logprobs=None)
                    return

                if result.outputs[0].logprobs is not None:
                    flat = result.outputs[0].logprobs
                    token_ids_pos, logprobs_pos, ranks_pos, decoded_pos = extract_and_dedup_flat_logprobs(flat, logprobs_num)
                    if valid_tokens_sets is not None:
                        valid_tokens_set = valid_tokens_sets[i]
                        logprobs_pos = [lp if tid in valid_tokens_set else -float("inf") for tid, lp in zip(token_ids_pos, logprobs_pos)]
                    beam_flat_cache.append((token_ids_pos, logprobs_pos, ranks_pos, decoded_pos))
                    all_beams_token_id.extend(token_ids_pos)
                    all_beams_logprob.extend(current_beam.cum_logprob + lp for lp in logprobs_pos)
                else:
                    beam_flat_cache.append(None)

        all_beams_token_id = np.array(all_beams_token_id)
        all_beams_logprob = np.array(all_beams_logprob)

        if not ignore_eos:
            eos_idx = np.where(all_beams_token_id == eos_token_id)[0]
            for idx in eos_idx:
                current_beam = all_beams[idx // logprobs_num]
                cached = beam_flat_cache[idx // logprobs_num]
                assert cached is not None
                eos_beam = BeamSearchSequence(
                    tokens=current_beam.tokens + [eos_token_id] if include_stop_str_in_output else current_beam.tokens,
                    logprobs=initial_logprobs,
                    cum_logprob=float(all_beams_logprob[idx]),
                    finish_reason="stop",
                    stop_reason=eos_token_id,
                )
                eos_beam._lp_parent = current_beam
                eos_beam._lp_step_data = cached
                completed.append(eos_beam)
            all_beams_logprob[eos_idx] = -np.inf

        if all_beams_logprob.size > beam_width:
            topn_idx = np.argpartition(np.negative(all_beams_logprob), beam_width)[:beam_width]
        else:
            topn_idx = np.arange(all_beams_logprob.size)

        for idx in topn_idx:
            current_beam = all_beams[idx // logprobs_num]
            token_id = int(all_beams_token_id[idx])
            cached = beam_flat_cache[idx // logprobs_num]
            assert cached is not None
            new_beam = BeamSearchSequence(
                tokens=current_beam.tokens + [token_id],
                logprobs=initial_logprobs,
                lora_request=current_beam.lora_request,
                cum_logprob=float(all_beams_logprob[idx]),
                multi_modal_data=current_beam.multi_modal_data,
                mm_processor_kwargs=current_beam.mm_processor_kwargs,
            )
            new_beam._lp_parent = current_beam
            new_beam._lp_step_data = cached
            new_beams.append(new_beam)

        if use_mega_request:
            fork_info = [(idx // logprobs_num, int(all_beams_token_id[idx])) for idx in topn_idx]

        all_beams = new_beams

    if use_mega_request and prev_beam_internal_ids:
        await _mega_request_cleanup(self.engine_client, prefill_id, prev_beam_internal_ids, rank)

    if sid_end_token_id is not None:
        for beam in all_beams:
            beam.tokens.append(sid_end_token_id)
    completed.extend(all_beams)
    sorted_completed = sorted(completed, key=lambda x: x.cum_logprob, reverse=True)
    best_beams = sorted_completed[:beam_width]

    for beam in best_beams:
        reconstruct_beam_logprobs(beam, initial_logprobs, sid_end_token_id)

    for i, beam in enumerate(best_beams):
        tokens = beam.tokens[tokenized_length:-1] if beam.tokens[-1] == eos_token_id and not ignore_eos else beam.tokens[tokenized_length:]
        beam.text = tokenizer.decode(tokens)

    beam_search_decode_time = time.perf_counter() - beam_search_start
    beam_search_overhead = beam_search_decode_time - generation_time
    metrics = RequestStateStats(beam_search_overhead=beam_search_overhead, beam_search_decode_time=beam_search_decode_time, num_generation_tokens=num_generation_tokens)

    yield RequestOutput(
        request_id=request_id,
        prompt=prompt_text,
        outputs=[CompletionOutput(text=beam.text, cumulative_logprob=beam.cum_logprob, token_ids=beam.tokens[tokenized_length:], index=i, logprobs=beam.logprobs, finish_reason=beam.finish_reason if beam.finish_reason is not None else "length", stop_reason=beam.stop_reason) for i, beam in enumerate(best_beams)],
        finished=True,
        prompt_token_ids=prompt_token_ids,
        prompt_logprobs=None,
        metrics=metrics,
    )