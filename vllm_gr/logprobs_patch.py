# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patch for FlatLogprobs.append_fast to use extend instead of
per-element append."""

from vllm.logprobs import FlatLogprobs

from vllm_gr.logprobs import append_fast as _append_fast


def patch_flat_logprobs():
    FlatLogprobs.append_fast = _append_fast
