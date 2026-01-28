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

        cli_env_setup()

        # For now remove the --gr flag from sys.argv to avoid issues in vLLM CLI
        idx = sys.argv.index("--gr")
        # Remove the flag and its value (if it has one)
        if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("--"):
            sys.argv.pop(idx + 1)  # Remove value
        sys.argv.pop(idx)  # Remove flag

    # Delegate to original vLLM CLI
    vllm_main()


if __name__ == "__main__":
    main()
