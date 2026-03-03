import logging
import sys

from vllm.entrypoints.cli.main import main as vllm_main

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    """Entry point for vllm-gr benchmark.
    The one_rec_patch is imported via __init__.py."""
    if "--gr" in sys.argv:
        logger.info(
            "vllm detected --gr flag in serve command. Launching vllm-gr specialized serving path."
        )
        from vllm_gr.patch import run_patch

        run_patch()
        sys.argv.remove("--gr")
    vllm_main()
