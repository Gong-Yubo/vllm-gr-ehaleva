# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from vllm.entrypoints.cli.main import main as vllm_main
import logging

logger = logging.getLogger(__name__)


def main():
    if "serve" in sys.argv and "--gr" in sys.argv:
        logger.info(
            "vllm detected --gr flag in serve command. Launching vllm-gr specialized serving path."
        )
        # Launch vLLM-gr specialized serving path
        from vllm.entrypoints.utils import cli_env_setup
        from vllm_gr.patch import run_patch

        run_patch()
        cli_env_setup()
        sys.argv.remove("--gr")

    # Delegate to original vLLM CLI
    vllm_main()


if __name__ == "__main__":
    main()
