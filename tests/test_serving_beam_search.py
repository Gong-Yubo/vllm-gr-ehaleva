# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

try:
    from vllm_gr.sampling_params import BeamSearchParams
except ImportError:
    from vllm.sampling_params import BeamSearchParams

try:
    from vllm_gr.v1.metrics.stats import RequestStateStats
except ImportError:
    from vllm.v1.metrics.stats import RequestStateStats

import os
import time
from typing import Any

import psutil
import pytest
import torch
from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput

if not torch.cuda.is_available():
    pytest.skip("CUDA is not available", allow_module_level=True)


async def beam_search_test(loops: int):
    model_name = "OpenOneRec/OneRec-1.7B"
    beam_width = 512
    max_tokens = 5

    engine_args = AsyncEngineArgs(
        model=model_name,
        trust_remote_code=True,
        enforce_eager=True,
        enable_log_requests=False,
        max_logprobs=2 * beam_width,
    )

    async with build_async_engine_client_from_engine_args(engine_args) as engine_client:
        base_model_paths = [BaseModelPath(name=model_name, model_path=model_name)]
        models = OpenAIServingModels(
            engine_client=engine_client, base_model_paths=base_model_paths, lora_modules=None
        )

        serving = OpenAIServing(
            engine_client=engine_client,
            models=models,
            request_logger=None,
        )

        # Use the specific prompt
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        prompt_file = os.path.join(os.path.dirname(__file__), "resources/single_one_rec_prompt.txt")
        with open(prompt_file, "r") as f:
            prompt_text = f.read()
        prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)

        beam_search_params: dict[str, Any] = {
            "beam_width": beam_width,
            "max_tokens": max_tokens,
            "ignore_eos": False,
            "temperature": 0.0,
        }
        if hasattr(BeamSearchParams, "begin_token"):
            beam_search_params["begin_token"] = "<|sid_begin|>"
            beam_search_params["end_token"] = "<|sid_end|>"

        params = BeamSearchParams(**beam_search_params)

        if loops <= 0:
            loops = 1
        print(f"Starting beam search with {model_name}, beam_width={beam_width} iterations={loops}")
        metrics: RequestStateStats = RequestStateStats()
        final_output: RequestOutput | None = None
        start: float = time.perf_counter()
        for i in range(loops):
            generator = serving.beam_search(prompt, "req_id", params)
            async for output in generator:
                final_output = output
                if final_output and final_output.metrics:
                    metrics.num_generation_tokens += final_output.metrics.num_generation_tokens
                    if hasattr(final_output.metrics, "beam_search_overhead"):
                        metrics.beam_search_overhead += getattr(
                            output.metrics, "beam_search_overhead", 0
                        )
                    if hasattr(final_output.metrics, "beam_search_decode_time"):
                        metrics.beam_search_decode_time += getattr(
                            output.metrics, "beam_search_decode_time", 0
                        )

        end = time.perf_counter()
        print(f"metrics={metrics} elapsed mean:{(end - start) / loops}")
        assert final_output is not None
        assert len(final_output.outputs) == beam_width


@pytest.mark.slow
@pytest.mark.asyncio
async def test_serving_beam_search(loops: int):
    await beam_search_test(loops)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_serving_gr_beam_search(loops: int):
    from vllm_gr.patch import run_patch

    run_patch()
    assert hasattr(BeamSearchParams, "begin_token")
    await beam_search_test(loops)
    torch.cuda.synchronize()
    before_gpu_mem: int = torch.cuda.memory_allocated()
    before_host_mem: int = psutil.Process().memory_info().rss
    await beam_search_test(loops)
    await beam_search_test(loops)
    torch.cuda.synchronize()
    after_gpu_mem: int = torch.cuda.memory_allocated()
    after_host_mem: int = psutil.Process().memory_info().rss
    print(
        f"before host mem={before_host_mem / 1024**3:.2f}GB, after host mem={after_host_mem / 1024**3:.2f}GB"
    )
    print(
        f"before gpu mem={before_gpu_mem / 1024**3:.2f}GB, after gpu mem={after_gpu_mem / 1024**3:.2f}GB"
    )
    # These thresholds (500MB host RAM, 50MB GPU RAM) match the minimum safe
    # memory headroom required for the 1.7B model to load and run without OOM.
    # They are intentionally conservative to avoid instability on smaller machines.
    if loops > 1:
        # Allow 500MB tolerance
        assert after_host_mem - before_host_mem <= 500 * 1024**2
        # Allow 50MB tolerance
        assert after_gpu_mem - before_gpu_mem <= 50 * 1024**2
