import vllm.benchmarks.datasets
import vllm.benchmarks.throughput
import atexit
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
import sys
from vllm.benchmarks.lib.endpoint_request_func import (
    RequestFuncInput,
    async_request_openai_chat_completions,
)
import vllm.benchmarks.lib.endpoint_request_func
from vllm.benchmarks.datasets import (
    get_samples as original_get_samples,
    add_dataset_parser as original_add_parser,
)
from vllm.benchmarks.throughput import (
    add_cli_args as original_add_cli_args,
    get_requests as original_get_requests,
    validate_args as original_validate_args,
    main as original_main,
)
from benchmarks.open_one_rec.open_one_rec_dataset import OneRecDataset

# ============================================================================
# Constants
# ============================================================================
DATASET_NAME = "onerec"
DEFAULT_TASK_TYPES = "rec_reason,item_understand,ad,product,label_cond,video,interactive,label_pred"
_n_idx = sys.argv.index("--n") if "--n" in sys.argv else -1
DEFAULT_N_BEAMS = int(sys.argv[_n_idx + 1]) if _n_idx != -1 and _n_idx + 1 < len(sys.argv) else 1
USE_BEAM_SEARCH = DEFAULT_N_BEAMS > 1
atexit.register(cleanup_dist_env_and_memory)

# ============================================================================
# Beam Search Patching
# ============================================================================
original_init = RequestFuncInput.__init__


def __init__(self, *args, use_beam_search=USE_BEAM_SEARCH, n=DEFAULT_N_BEAMS, **kwargs):
    """Patched init to support beam search parameters."""
    original_init(self, *args, **kwargs)
    self.use_beam_search = use_beam_search
    self.n = n


RequestFuncInput.__init__ = __init__


async def patched_async_request_openai_chat_completions(request_func_input, pbar=None):
    """Add beam search parameters to request payload."""
    if request_func_input.use_beam_search:
        if request_func_input.extra_body is None:
            request_func_input.extra_body = {}
        request_func_input.extra_body["use_beam_search"] = True
        request_func_input.extra_body["n"] = request_func_input.n

    return await async_request_openai_chat_completions(request_func_input, pbar)


# ============================================================================
# Dataset Patching
# ============================================================================
def _add_dataset_to_parser_choices(parser, dataset_name):
    """Helper to add a dataset to parser choices."""
    for action in parser._actions:
        if hasattr(action, "dest") and action.dest == "dataset_name":
            if dataset_name not in action.choices:
                action.choices = list(action.choices) + [dataset_name]
            return True
    return False


def _add_onerec_arguments(parser):
    """Helper to add OneRec-specific arguments to parser."""
    onerec_group = parser.add_argument_group("onerec dataset options")
    onerec_group.add_argument(
        "--task-types",
        type=str,
        default=DEFAULT_TASK_TYPES,
        help="Comma-separated list of task types",
    )


def _add_beams_arguments(parser):
    """Helper to add beam search arguments to parser."""
    beams_group = parser.add_argument_group("beam search options")
    beams_group.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N_BEAMS,
        help="Number of beams for beam search",
    )


def patched_get_samples(args, tokenizer):
    """Patch get_samples to handle OneRecDataset."""
    if not hasattr(args, "request_id_prefix"):
        args.request_id_prefix = ""

    if args.dataset_name == DATASET_NAME:
        dataset = OneRecDataset(
            task_types=args.task_types.split(","),
            model_path=args.model,
            dataset_path=args.dataset_path,
            disable_shuffle=args.disable_shuffle,
            tokenizer=tokenizer,
        )
        return dataset.sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            request_id_prefix=args.request_id_prefix,
            no_oversample=args.no_oversample,
        )

    return original_get_samples(args, tokenizer)


def patched_add_parser(parser):
    """Patch add_dataset_parser to include onerec in choices."""
    original_add_parser(parser)
    _add_dataset_to_parser_choices(parser, DATASET_NAME)
    _add_onerec_arguments(parser)
    _add_beams_arguments(parser)
    return parser


def patched_add_cli_args(parser):
    """Patch add_cli_args to include onerec arguments."""
    original_add_cli_args(parser)
    _add_dataset_to_parser_choices(parser, DATASET_NAME)
    _add_onerec_arguments(parser)
    return parser


def patched_get_requests(args, tokenizer):
    """Patch get_requests to handle OneRecDataset."""
    if args.dataset_name == DATASET_NAME:
        dataset = OneRecDataset(
            dataset_path=args.dataset_path,
            task_types=args.task_types.split(","),
            model_path=args.model,
            tokenizer=tokenizer,
        )
        return dataset.sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
        )

    return original_get_requests(args, tokenizer)


def patched_validate_args(args):
    """Patch validate_args to handle onerec dataset."""
    if args.dataset_name != DATASET_NAME:
        original_validate_args(args)


def patched_main(args):
    """Patch main to fix tokenizer initialization."""
    if args.tokenizer is None and hasattr(args, "model"):
        args.tokenizer = args.model

    original_main(args)


# ============================================================================
# Apply Patches
# ============================================================================
print("Patching vllm.benchmarks for OneRec support")

vllm.benchmarks.datasets.get_samples = patched_get_samples
vllm.benchmarks.datasets.add_dataset_parser = patched_add_parser
vllm.benchmarks.throughput.add_cli_args = patched_add_cli_args
vllm.benchmarks.throughput.get_requests = patched_get_requests
vllm.benchmarks.throughput.validate_args = patched_validate_args
vllm.benchmarks.throughput.main = patched_main
vllm.benchmarks.lib.endpoint_request_func.async_request_openai_chat_completions = (
    patched_async_request_openai_chat_completions
)
