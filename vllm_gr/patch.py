# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_gr.entrypoints.openai.beam_search_patch import patch_beam_search
from vllm_gr.entrypoints.openai.beam_search_patch import patch_sampling


def run_patch():
    patch_beam_search()
    patch_sampling()
