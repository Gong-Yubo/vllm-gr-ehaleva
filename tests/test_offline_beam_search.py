# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing
import os
import time
from typing import Any, Dict

import psutil
import pytest
import torch
from vllm import LLM

try:
    from vllm_gr.sampling_params import BeamSearchParams
except ImportError:
    from vllm.sampling_params import BeamSearchParams

if not torch.cuda.is_available():
    pytest.skip("CUDA is not available", allow_module_level=True)


def _load_prompt_text() -> str:
    prompt_file = os.path.join(os.path.dirname(__file__), "resources/single_one_rec_prompt.txt")
    with open(prompt_file, "r") as f:
        return f.read()


def _run_offline_beam_search(loops: int, use_patch: bool) -> None:
    model_name = "OpenOneRec/OneRec-1.7B"
    beam_width = 512
    max_tokens = 5

    if use_patch:
        import vllm_gr.init  # noqa: F401

        assert hasattr(BeamSearchParams, "begin_token")

    params_dict: Dict[str, Any] = {
        "beam_width": beam_width,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": False,
    }
    if hasattr(BeamSearchParams, "begin_token"):
        params_dict["begin_token"] = "<|sid_begin|>"
        params_dict["end_token"] = "<|sid_end|>"

    params = BeamSearchParams(**params_dict)

    prompt_text = _load_prompt_text()
    prompts = [{"prompt": prompt_text}]
    print(f"Running offline beam search (patch={use_patch}) loops={loops}")

    # vllm-gr is designed to use beam_width for max_logprobs (not 2 * beam_width)
    # to improve performance. Reflect that here when use_patch is True.
    max_logprobs = beam_width if use_patch else 2 * beam_width

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_logprobs=max_logprobs,
    )
    final_output = None
    start = time.perf_counter()
    for _ in range(max(1, loops)):
        outputs = llm.beam_search(prompts, params)
        assert len(outputs) == len(prompts)
        final_output = outputs[0]
        assert len(final_output.sequences) == beam_width
    end = time.perf_counter()
    print(f"Elapsed mean: {(end - start) / max(1, loops):.4f}s")


def _run_offline_beam_search_target(loops: int) -> None:
    _run_offline_beam_search(loops, use_patch=False)


@pytest.mark.slow  # type: ignore
def test_offline_beam_search(loops: int) -> None:
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=_run_offline_beam_search_target, args=(loops,))
    p.start()
    p.join()
    assert p.exitcode == 0


def _run_offline_gr_beam_search_target(loops: int) -> None:
    before_gpu = torch.cuda.memory_allocated()
    before_host = psutil.Process().memory_info().rss
    _run_offline_beam_search(loops, use_patch=True)
    _run_offline_beam_search(loops, use_patch=True)
    torch.cuda.synchronize()
    after_gpu = torch.cuda.memory_allocated()
    after_host = psutil.Process().memory_info().rss
    print(
        f"host mem delta={(after_host - before_host) / 1024**3:.2f}GB, "
        f"gpu mem delta={(after_gpu - before_gpu) / 1024**3:.2f}GB"
    )
    # These thresholds (500MB host RAM, 50MB GPU RAM) match the minimum safe
    # memory headroom required for the 1.7B model to load and run without OOM.
    # They are intentionally conservative to avoid instability on smaller machines.
    if loops > 1:
        assert after_host - before_host <= 500 * 1024**2
        assert after_gpu - before_gpu <= 50 * 1024**2


@pytest.mark.slow  # type: ignore
def test_offline_gr_beam_search(loops: int) -> None:
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=_run_offline_gr_beam_search_target, args=(loops,))
    p.start()
    p.join()
    assert p.exitcode == 0
