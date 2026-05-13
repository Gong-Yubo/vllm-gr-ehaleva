# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import multiprocessing
import os
import random
import tempfile
from typing import Any

import pytest
import torch

try:
    from vllm_gr.sampling_params import BeamSearchParams
except ImportError:
    from vllm.sampling_params import BeamSearchParams


from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.inputs import TokensPrompt

if not torch.cuda.is_available():
    pytest.skip("CUDA is not available", allow_module_level=True)


async def _serving_catalog_generation_impl(loops: int) -> None:
    # Mockup catalog: list of 10 lists of 5 tokens
    # We use tokens that are compatible with the model's tokenizer/format
    catalog_data = []
    random.seed(42)
    for i in range(50000):
        item = [
            "<|sid_begin|>",
            f"<s_a_{random.randint(0, 7999)}>",
            f"<s_b_{random.randint(0, 7999)}>",
            f"<s_c_{random.randint(0, 7999)}>",
            "<|sid_end|>",
        ]
        catalog_data.append(item)
    for i in range(8000):
        catalog_data.append(
            ["<|sid_begin|>", "<s_a_4247>", "<s_b_6120>", f"<s_c_{i}>", "<|sid_end|>"]
        )

    print(f"catalog created with {len(catalog_data)} items")
    # Create temporary catalog file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp_catalog:
        json.dump(catalog_data, tmp_catalog)
        catalog_path = tmp_catalog.name

    try:
        model_name = "OpenOneRec/OneRec-1.7B"

        engine_args = AsyncEngineArgs(
            model=model_name,
            trust_remote_code=True,
            enforce_eager=True,
            enable_log_requests=False,
            max_logprobs=1024,
        )
        # Inject catalog path
        setattr(engine_args, "catalog_path", catalog_path)

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

            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            # Dummy prompt
            prompt_file = os.path.join(
                os.path.dirname(__file__), "resources/single_one_rec_prompt.txt"
            )
            with open(prompt_file, "r") as f:
                prompt_text = f.read()
            prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)

            beam_width = 1024
            beam_search_params_dict: dict[str, Any] = {
                "beam_width": beam_width,
                "max_tokens": 5,
                "ignore_eos": False,
                "temperature": 0.0,
            }
            if hasattr(BeamSearchParams, "begin_token"):
                beam_search_params_dict["begin_token"] = "<|sid_begin|>"
                beam_search_params_dict["end_token"] = "<|sid_end|>"

            params = BeamSearchParams(**beam_search_params_dict)

            print(f"Starting beam search with catalog constraint, loops={loops}...")

            # Prepare catalog set for verification
            catalog_set = set("".join(item) for item in catalog_data)
            if loops <= 0:
                loops = 1
            for i in range(loops):
                generator = serving.beam_search(prompt, f"req_id_catalog_{i}", params)

                final_output = None
                async for output in generator:
                    final_output = output

                assert final_output is not None
                assert len(final_output.outputs) == beam_width

                print(f"Loop {i}: Verifying {len(final_output.outputs)} generated choices...")
                not_in_catalog = []
                for choice in final_output.outputs:
                    content = choice.text
                    if content not in catalog_set:
                        not_in_catalog.append(content)
                if not_in_catalog:
                    print("Generated content not found in catalog:")
                    for i, content in enumerate(not_in_catalog):
                        print(f"{i}: {content}")
                    assert False, "Some Generated content not found in catalog."

    finally:
        if os.path.exists(catalog_path):
            os.remove(catalog_path)


def _run_serving_catalog_generation_target(loops: int) -> None:
    from vllm_gr.patch import run_patch

    run_patch()
    asyncio.run(_serving_catalog_generation_impl(loops))


@pytest.mark.slow  # type: ignore
def test_serving_catalog_generation(loops: int) -> None:
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=_run_serving_catalog_generation_target, args=(loops,))
    p.start()
    p.join()
    assert p.exitcode == 0
