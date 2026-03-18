# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vllm-gr: High-throughput serving plugin for HSTU generative recommendation models."""

__version__ = "0.1.0"

try:
    import torch
    from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available

    if torch.cuda.is_available() and is_flash_attn_varlen_func_available():
        from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionBackend  # noqa: F401
except ImportError:
    pass
