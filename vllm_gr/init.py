# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

from vllm_gr.patch import run_patch

# Automatically apply vllm_gr patches on import, unless explicitly disabled.
# To disable auto-patching, set environment variable VLLM_GR_AUTOPATCH to "0", "false", or "no".
if os.getenv("VLLM_GR_AUTOPATCH", "1").lower() not in ("0", "false", "no"):
    run_patch()
