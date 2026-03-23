# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger

from vllm_gr.engine.arg_utils import patch_add_cli_args
from vllm_gr.entrypoints.beam_search_patch import patch_offline_beam_search
from vllm_gr.entrypoints.openai.beam_search_patch import (
    patch_batch_and_fork,
    patch_beam_search,
    patch_sampling,
)
from vllm_gr.entrypoints.openai.serving_models import patch_OpenAIServingModels_init

logger = init_logger(__name__)

_PATCHED = False


def run_patch():
    global _PATCHED
    if _PATCHED:
        logger.info("vllm-gr patches have already been applied.")
        return
    _PATCHED = True
    patch_beam_search()
    patch_sampling()
    patch_add_cli_args()
    patch_OpenAIServingModels_init()
    patch_offline_beam_search()
    patch_batch_and_fork()
