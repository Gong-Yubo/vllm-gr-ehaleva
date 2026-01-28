from vllm.entrypoints.cli.main import main as vllm_main

if __name__ == "__main__":
    """Entry point for vllm-gr benchmark.
    The one_rec_patch is imported via __init__.py."""
    vllm_main()
