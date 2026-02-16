# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm_gr.entrypoints.openai.serving_engine import beam_search
from vllm_gr.entrypoints.openai.protocol import to_beam_search_params


def patch_beam_search():
    OpenAIServing.beam_search = beam_search


def patch_sampling():
    ChatCompletionRequest.to_beam_search_params = to_beam_search_params
