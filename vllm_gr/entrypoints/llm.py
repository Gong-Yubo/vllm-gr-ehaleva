# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from typing import cast

import numpy as np
from tqdm.auto import tqdm
from vllm.beam_search import (
    BeamSearchInstance,
    BeamSearchOutput,
    BeamSearchSequence,
    create_sort_beams_key_function,
)
from vllm.inputs import TextPrompt, TokensPrompt
from vllm.logger import init_logger
from vllm.logprobs import FlatLogprobs
from vllm.lora.request import LoRARequest
from vllm.sampling_params import BeamSearchParams, SamplingParams

from vllm_gr.logprobs import extract_and_dedup_flat_logprobs, reconstruct_beam_logprobs

logger = init_logger(__name__)


def beam_search(
    self,
    prompts: list[TokensPrompt | TextPrompt],
    params: BeamSearchParams,
    lora_request: list[LoRARequest] | LoRARequest | None = None,
    use_tqdm: bool = False,
    concurrency_limit: int | None = None,
) -> list[BeamSearchOutput]:
    """
    Generate sequences using beam search.

    Args:
        prompts: A list of prompts. Each prompt can be a string or a list
            of token IDs.
        params: The beam search parameters.
        lora_request: LoRA request to use for generation, if any.
        use_tqdm: Whether to use tqdm to display the progress bar.
        concurrency_limit: The maximum number of concurrent requests.
            If None, the number of concurrent requests is unlimited.
    """
    # TODO: how does beam search work together with length penalty,
    # frequency, penalty, and stopping criteria, etc.?
    logger.debug("Called beam_search")
    beam_width = params.beam_width
    max_tokens = params.max_tokens
    temperature = params.temperature
    ignore_eos = params.ignore_eos
    length_penalty = params.length_penalty
    begin_token = params.begin_token if hasattr(params, "begin_token") else None
    end_token = params.end_token if hasattr(params, "end_token") else None

    lora_requests = self._get_beam_search_lora_requests(lora_request, prompts)

    tokenizer = self.get_tokenizer()
    sid_begin_token_id: int | None = None
    if begin_token is not None:
        sid_begin_token_id = tokenizer.convert_tokens_to_ids(begin_token)
        if sid_begin_token_id is None or sid_begin_token_id == -1:
            logger.warning(
                f"begin_token must be a valid token in the tokenizer vocabulary, got token {begin_token}"
            )
            sid_begin_token_id = None
    sid_end_token_id: int | None = None
    if end_token is not None:
        sid_end_token_id = tokenizer.convert_tokens_to_ids(end_token)
        if sid_end_token_id is None or sid_end_token_id == -1:
            logger.warning(
                f"end_token must be a valid token in the tokenizer vocabulary, got token {end_token}"
            )
            sid_end_token_id = None
    sort_beams_key = create_sort_beams_key_function(
        tokenizer.eos_token_id,
        length_penalty,
    )

    if use_tqdm and concurrency_limit is not None:
        logger.warning(
            "Progress bar is not supported when using concurrency_limit. Disabling progress bar."
        )
        use_tqdm = False

    if concurrency_limit is None:
        concurrency_limit = len(prompts)

    def create_tokens_prompt_from_beam(beam: BeamSearchSequence) -> TokensPrompt:
        token_prompt_kwargs: TokensPrompt = {"prompt_token_ids": beam.tokens}
        if beam.multi_modal_data is not None:
            token_prompt_kwargs["multi_modal_data"] = beam.multi_modal_data

        if beam.mm_processor_kwargs is not None:
            token_prompt_kwargs["mm_processor_kwargs"] = beam.mm_processor_kwargs
        return TokensPrompt(**token_prompt_kwargs)

    logprobs_num = beam_width
    # generate beam_width candidates at each step
    beam_search_params = SamplingParams(
        logprobs=logprobs_num,
        max_tokens=1,
        temperature=temperature,
        skip_clone=True,  # Internal beam search, safe to skip clone
        detokenize=False,
        flat_logprobs=True,
    )
    # NOTE: FlatLogprobs intentionally passed where BeamSearchSequence
    # expects list[dict[int, Logprob]].  Intermediate logprobs are never
    # read as dicts; the final beam.logprobs is replaced during
    # reconstruction at the end.
    # This reference is shared read-only across all beams — do not mutate
    # after the initial append below.
    initial_logprobs = FlatLogprobs()
    if sid_begin_token_id is not None:
        initial_logprobs.append_fast([sid_begin_token_id], [0.0], [None], [None])

    instances: list[BeamSearchInstance] = []

    for lora_req, prompt in zip(lora_requests, prompts):
        # Add multimodal processor kwargs & data
        mm_kwargs = {}
        if "multi_modal_data" in prompt:
            mm_kwargs["multi_modal_data"] = prompt["multi_modal_data"]
        if "mm_processor_kwargs" in prompt:
            mm_kwargs["mm_processor_kwargs"] = prompt["mm_processor_kwargs"]

        if "prompt_token_ids" in prompt:
            prompt = cast(TokensPrompt, prompt)  # Needed for mypy
            prompt_tokens = prompt["prompt_token_ids"]
        else:
            prompt_tokens = tokenizer.encode(prompt["prompt"])

        if sid_begin_token_id is not None or sid_end_token_id is not None:
            prompt_tokens = list(prompt_tokens)
            if sid_begin_token_id is not None:
                prompt_tokens.append(sid_begin_token_id)

        instances.append(
            BeamSearchInstance(
                prompt_tokens,
                lora_request=lora_req,
                logprobs=None,
                **mm_kwargs,
            ),
        )
        instances[-1].beams[0].logprobs = initial_logprobs
        # Deferred logprobs: parent pointers on BeamSearchSequence (a plain
        # @dataclass without __slots__).  If upstream adds __slots__, these
        # dynamic attrs will break — upstream the fields if that happens.
        instances[-1].beams[0]._lp_parent = None
        instances[-1].beams[0]._lp_step_data = None

    pre_calc = 0 if sid_begin_token_id is None else 1
    if sid_end_token_id is not None:
        pre_calc += 1
    if max_tokens - pre_calc < 0:
        logger.warning(
            "max_tokens is less than the number of tokens to be pre-calculated. "
            "Setting max_tokens to the number of pre-calculated tokens."
        )
        max_tokens = pre_calc
    for prompt_start in range(0, len(prompts), concurrency_limit):
        instances_batch = instances[prompt_start : prompt_start + concurrency_limit]

        token_iter = range(max_tokens - pre_calc)
        if use_tqdm:
            token_iter = tqdm(token_iter, desc="Beam search", unit="token", unit_scale=False)
            logger.warning(
                "The progress bar shows the upper bound on token steps and "
                "may finish early due to stopping conditions. It does not "
                "reflect instance-level progress."
            )
        for _ in token_iter:
            all_beams: list[BeamSearchSequence] = list(
                sum((instance.beams for instance in instances_batch), [])
            )
            pos = [0] + list(
                itertools.accumulate(len(instance.beams) for instance in instances_batch)
            )
            instance_start_and_end: list[tuple[int, int]] = list(zip(pos[:-1], pos[1:]))

            if len(all_beams) == 0:
                break

            # create corresponding batch entries for prompt & optional lora
            prompts_batch, lora_req_batch = zip(
                *[(create_tokens_prompt_from_beam(beam), beam.lora_request) for beam in all_beams]
            )

            # only runs for one step
            # we don't need to use tqdm here
            output = self.generate(
                prompts_batch,
                sampling_params=beam_search_params,
                use_tqdm=False,
                lora_request=lora_req_batch,
            )

            for (start, end), instance in zip(instance_start_and_end, instances_batch):
                # Gather all logprobs and cumulative scores for this instance
                all_beams_token_id = []
                all_beams_logprob = []
                beam_flat_cache: list[tuple | None] = []
                beam_outputs = output[start:end]  # outputs for this instance's beams

                for i, (current_beam, result) in enumerate(zip(all_beams[start:end], beam_outputs)):
                    if result.outputs[0].logprobs is not None:
                        flat = result.outputs[0].logprobs
                        token_ids_pos, logprobs_pos, ranks_pos, decoded_pos = (
                            extract_and_dedup_flat_logprobs(flat, logprobs_num)
                        )
                        beam_flat_cache.append(
                            (token_ids_pos, logprobs_pos, ranks_pos, decoded_pos)
                        )
                        all_beams_token_id.extend(token_ids_pos)
                        all_beams_logprob.extend(
                            current_beam.cum_logprob + lp for lp in logprobs_pos
                        )
                    else:
                        raise ValueError(
                            "Beam search generation expected non-empty logprobs "
                            "for each beam, but received 'None'. Please ensure "
                            "that sampling parameters are configured to return "
                            "logprobs when using beam search."
                        )

                # Convert to numpy for efficient processing
                all_beams_token_id = np.array(all_beams_token_id)
                all_beams_logprob = np.array(all_beams_logprob)

                # Handle EOS tokens
                if not ignore_eos:
                    eos_idx = np.where(all_beams_token_id == tokenizer.eos_token_id)[0]
                    for idx in eos_idx:
                        beam_idx = idx // logprobs_num
                        current_beam = all_beams[start:end][beam_idx]
                        cached = beam_flat_cache[beam_idx]
                        completed_beam = BeamSearchSequence(
                            tokens=current_beam.tokens + [tokenizer.eos_token_id],
                            logprobs=initial_logprobs,
                            cum_logprob=float(all_beams_logprob[idx]),
                            finish_reason="stop",
                            stop_reason=tokenizer.eos_token_id,
                            multi_modal_data=current_beam.multi_modal_data,
                            mm_processor_kwargs=current_beam.mm_processor_kwargs,
                            lora_request=current_beam.lora_request,
                        )
                        completed_beam._lp_parent = current_beam
                        completed_beam._lp_step_data = cached
                        instance.completed.append(completed_beam)
                    # Exclude EOS from further consideration
                    all_beams_logprob[eos_idx] = -np.inf

                # Select top beams
                if len(all_beams_logprob) <= beam_width:
                    topn_idx = np.arange(len(all_beams_logprob))
                else:
                    topn_idx = np.argpartition(np.negative(all_beams_logprob), beam_width)[
                        :beam_width
                    ]
                new_beams = []
                for idx in topn_idx:
                    beam_idx = idx // logprobs_num
                    current_beam = all_beams[start:end][beam_idx]
                    cached = beam_flat_cache[beam_idx]
                    new_beam = BeamSearchSequence(
                        tokens=current_beam.tokens + [int(all_beams_token_id[idx])],
                        logprobs=initial_logprobs,
                        cum_logprob=float(all_beams_logprob[idx]),
                        multi_modal_data=current_beam.multi_modal_data,
                        mm_processor_kwargs=current_beam.mm_processor_kwargs,
                        lora_request=current_beam.lora_request,
                    )
                    new_beam._lp_parent = current_beam
                    new_beam._lp_step_data = cached
                    new_beams.append(new_beam)

                instance.beams = new_beams

    outputs = []
    for instance in instances:
        # Append sid_end_token_id only to active beams (not EOS-completed
        # beams that already terminated naturally).
        if sid_end_token_id is not None:
            for beam in instance.beams:
                beam.tokens.append(sid_end_token_id)
        # Move remaining active beams into the completed list.
        instance.completed.extend(instance.beams)
        sorted_completed = sorted(instance.completed, key=sort_beams_key, reverse=True)
        best_beams = sorted_completed[:beam_width]

        # Reconstruct FlatLogprobs from parent pointers for the best beams.
        for beam in best_beams:
            reconstruct_beam_logprobs(beam, initial_logprobs, sid_end_token_id)
            beam.text = tokenizer.decode(beam.tokens)
        outputs.append(BeamSearchOutput(sequences=best_beams))

    return outputs
