# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_gr.engine.arg_utils import patch_add_cli_args
from vllm_gr.entrypoints.beam_search_patch import patch_offline_beam_search
from vllm_gr.entrypoints.openai.beam_search_patch import (
    patch_beam_search,
    patch_sampling,
)
from vllm_gr.entrypoints.openai.serving_models import patch_OpenAIServingModels_init


def run_patch():
    patch_beam_search()
    patch_sampling()
    patch_add_cli_args()
    patch_OpenAIServingModels_init()
    patch_offline_beam_search()
