# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import time
from collections.abc import AsyncGenerator, Mapping
from typing import Any

import numpy as np
from vllm.beam_search import BeamSearchSequence
from vllm.entrypoints.openai.protocol import VLLMValidationError
from vllm.inputs.data import PromptType, TokensPrompt
from vllm.inputs.parse import is_explicit_encoder_decoder_prompt
from vllm.logprobs import Logprob
from vllm.lora.request import LoRARequest
from vllm.multimodal import MultiModalDataDict
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.utils import random_uuid
from vllm.utils.async_utils import collect_from_async_generator

from vllm_gr.v1.metrics.stats import RequestStateStats


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
    include_stop_str_in_output = params.include_stop_str_in_output
    if beam_width == 0:
        raise VLLMValidationError(
            "Beam width must be greater than 0", parameter="beam_width", value=0
        )
    input_processor = self.input_processor
    tokenizer = input_processor.tokenizer
    if tokenizer is None:
        raise VLLMValidationError(
            "You cannot use beam search when `skip_tokenizer_init=True`",
            parameter="skip_tokenizer_init",
            value=True,
        )

    eos_token_id: int = tokenizer.eos_token_id  # type: ignore
    sid_begin_token_id: int | None = None
    if begin_token is not None:
        sid_begin_token_id = tokenizer.convert_tokens_to_ids(begin_token)
        if sid_begin_token_id is None or sid_begin_token_id == -1:
            raise VLLMValidationError(
                "begin_token must be a valid token in the tokenizer vocabulary",
                parameter="begin_token",
                value=begin_token,
            )
    sid_end_token_id: int | None = None
    if end_token is not None:
        sid_end_token_id = tokenizer.convert_tokens_to_ids(end_token)
        if sid_end_token_id is None or sid_end_token_id == -1:
            raise VLLMValidationError(
                "end_token must be a valid token in the tokenizer vocabulary",
                parameter="end_token",
                value=end_token,
            )

    if is_explicit_encoder_decoder_prompt(prompt):
        raise NotImplementedError

    prompt_text: str | None
    prompt_token_ids: list[int]
    multi_modal_data: MultiModalDataDict | None
    if isinstance(prompt, str):
        prompt_text = prompt
        prompt_token_ids = []
        multi_modal_data = None
    else:
        prompt_text = prompt.get("prompt")  # type: ignore
        prompt_token_ids = prompt.get("prompt_token_ids", [])  # type: ignore
        multi_modal_data = prompt.get("multi_modal_data")  # type: ignore

    mm_processor_kwargs: dict[str, Any] | None = None

    # This is a workaround to fix multimodal beam search; this is a
    # bandaid fix for 2 small problems:
    # 1. Multi_modal_data on the processed_inputs currently resolves to
    #    `None`.
    # 2. preprocessing above expands the multimodal placeholders. However,
    #    this happens again in generation, so the double expansion causes
    #    a mismatch.
    # TODO - would be ideal to handle this more gracefully.

    tokenized_length = len(prompt_token_ids)

    logprobs_num = beam_width
    beam_search_params = SamplingParams(
        logprobs=logprobs_num, max_tokens=1, temperature=temperature, detokenize=False
    )
    initial_tokens = list(prompt_token_ids)
    initial_logprobs = []
    if sid_begin_token_id is not None:
        initial_tokens.append(sid_begin_token_id)
        initial_logprobs.append({sid_begin_token_id: Logprob(logprob=0.0)})

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
    completed = []

    beam_search_start: int = time.perf_counter()
    pre_calc = 0 if sid_begin_token_id is None else 1
    if sid_end_token_id is not None:
        pre_calc += 1
    if max_tokens - pre_calc < 0:
        raise VLLMValidationError(
            f"max tokens should be not lower than {pre_calc} to accomedate reserved begin/end tokens.",
            parameter="max_tokens",
            value=max_tokens,
        )
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
        tasks = []
        request_id_batch = f"{request_id}-{random_uuid()}"

        for i, (individual_prompt, lora_req) in enumerate(zip(prompts_batch, lora_req_batch)):
            request_id_item = f"{request_id_batch}-beam-{i}"
            task = asyncio.create_task(
                collect_from_async_generator(
                    self.engine_client.generate(
                        individual_prompt,
                        beam_search_params,
                        request_id_item,
                        lora_request=lora_req,
                        trace_headers=trace_headers,
                    )
                )
            )
            tasks.append(task)

        catalog_tasks = []
        if self.models.catalog is not None:

            def get_valid_tokens_set(beam) -> set[int]:
                generated_tokens = beam.tokens[len(prompt_token_ids) :]
                return self.models.catalog.valid(generated_tokens)

            for beam in all_beams:
                catalog_tasks.append(asyncio.to_thread(get_valid_tokens_set, beam))

        gen_start = time.perf_counter()
        if catalog_tasks:
            all_results = await asyncio.gather(*(tasks + catalog_tasks))
            output = [x[0] for x in all_results[: len(tasks)]]
            valid_tokens_sets = all_results[len(tasks) :]
        else:
            output = [x[0] for x in await asyncio.gather(*tasks)]
        if token > 0:
            generation_time += time.perf_counter() - gen_start
        new_beams = []
        # Store all new tokens generated by beam
        all_beams_token_id = []
        # Store the cumulative probability of all tokens
        # generated by beam search
        all_beams_logprob = []
        # Iterate through all beam inference results
        for i, result in enumerate(output):
            current_beam = all_beams[i]

            # check for error finish reason and abort beam search
            if result.outputs[0].finish_reason == "error":
                # yield error output and terminate beam search
                yield RequestOutput(
                    request_id=request_id,
                    prompt=prompt_text,
                    outputs=[
                        CompletionOutput(
                            index=0,
                            text="",
                            token_ids=[],
                            cumulative_logprob=None,
                            logprobs=None,
                            finish_reason="error",
                        )
                    ],
                    finished=True,
                    prompt_token_ids=prompt_token_ids,
                    prompt_logprobs=None,
                )
                return

            if result.outputs[0].logprobs is not None:
                logprobs = result.outputs[0].logprobs[0]
                if len(logprobs) > logprobs_num:
                    logprobs = dict(list(logprobs.items())[:logprobs_num])
                if self.models.catalog is not None and catalog_tasks:
                    valid_tokens_set = valid_tokens_sets[i]
                    for token_id in list(logprobs.keys()):
                        if token_id not in valid_tokens_set:
                            logprobs[token_id].logprob = -float("inf")
                all_beams_token_id.extend(list(logprobs.keys()))
                all_beams_logprob.extend(
                    [current_beam.cum_logprob + obj.logprob for obj in logprobs.values()]
                )

        # Handle the token for the end of sentence (EOS)
        all_beams_token_id = np.array(all_beams_token_id)
        all_beams_logprob = np.array(all_beams_logprob)

        if not ignore_eos:
            # Get the index position of eos token in all generated results
            eos_idx = np.where(all_beams_token_id == eos_token_id)[0]
            for idx in eos_idx:
                current_beam = all_beams[idx // logprobs_num]
                result = output[idx // logprobs_num]
                assert result.outputs[0].logprobs is not None
                logprobs_entry = result.outputs[0].logprobs[0]
                completed.append(
                    BeamSearchSequence(
                        tokens=current_beam.tokens + [eos_token_id]
                        if include_stop_str_in_output
                        else current_beam.tokens,
                        logprobs=current_beam.logprobs + [logprobs_entry],
                        cum_logprob=float(all_beams_logprob[idx]),
                        finish_reason="stop",
                        stop_reason=eos_token_id,
                    )
                )
            # After processing, set the log probability of the eos condition
            # to negative infinity.
            all_beams_logprob[eos_idx] = -np.inf

        # Processing non-EOS tokens
        # Get indices of the top beam_width probabilities
        if all_beams_logprob.size > beam_width:
            topn_idx = np.argpartition(np.negative(all_beams_logprob), beam_width)[:beam_width]
        else:
            topn_idx = np.arange(all_beams_logprob.size)

        for idx in topn_idx:
            current_beam = all_beams[idx // logprobs_num]
            result = output[idx // logprobs_num]
            token_id = int(all_beams_token_id[idx])
            assert result.outputs[0].logprobs is not None
            logprobs_entry = result.outputs[0].logprobs[0]
            new_beams.append(
                BeamSearchSequence(
                    tokens=current_beam.tokens + [token_id],
                    logprobs=current_beam.logprobs + [logprobs_entry],
                    lora_request=current_beam.lora_request,
                    cum_logprob=float(all_beams_logprob[idx]),
                    multi_modal_data=current_beam.multi_modal_data,
                    mm_processor_kwargs=current_beam.mm_processor_kwargs,
                )
            )

        all_beams = new_beams

    if sid_end_token_id is not None:
        for beam in all_beams:
            beam.tokens.append(sid_end_token_id)
            beam.logprobs.append({sid_end_token_id: Logprob(logprob=0.0)})
    completed.extend(all_beams)
    sorted_completed = sorted(completed, key=lambda x: x.cum_logprob, reverse=True)
    best_beams = sorted_completed[:beam_width]

    for beam in best_beams:
        if beam.tokens[-1] == eos_token_id and not ignore_eos:
            # Skip the eos token in the text.
            tokens = beam.tokens[tokenized_length:-1]
        else:
            tokens = beam.tokens[tokenized_length:]
        beam.text = tokenizer.decode(tokens)

    beam_search_decode_time = time.perf_counter() - beam_search_start
    beam_search_overhead = beam_search_decode_time - generation_time
    metrics = RequestStateStats(
        beam_search_overhead=beam_search_overhead,
        beam_search_decode_time=beam_search_decode_time,
        num_generation_tokens=num_generation_tokens,
    )

    out = RequestOutput(
        request_id=request_id,
        prompt=prompt_text,
        outputs=[
            CompletionOutput(
                text=beam.text,  # type: ignore
                cumulative_logprob=beam.cum_logprob,
                token_ids=beam.tokens[tokenized_length:],
                index=i,
                logprobs=beam.logprobs,
                finish_reason=beam.finish_reason if beam.finish_reason is not None else "length",
                stop_reason=beam.stop_reason,
            )
            for (i, beam) in enumerate(best_beams)
        ],
        finished=True,
        prompt_token_ids=prompt_token_ids,
        prompt_logprobs=None,
        metrics=metrics,
    )
    yield out
