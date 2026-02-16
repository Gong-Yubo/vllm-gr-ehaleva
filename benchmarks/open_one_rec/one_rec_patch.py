import aiohttp
from tqdm import tqdm
import vllm.benchmarks.datasets
import vllm.benchmarks.throughput
import atexit
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
import sys
import time
import traceback
import json
import os
from typing import Literal
from vllm.benchmarks.lib.endpoint_request_func import (
    RequestFuncInput,
    ASYNC_REQUEST_FUNCS,
    _update_payload_common,
    _validate_api_url,
    _get_chat_content,
    StreamedResponseHandler,
    _update_headers_common,
    RequestFuncOutput,
)
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


def _get_default_n_beams_from_argv():
    """
    Parses sys.argv to find the value for --n.
    NOTE: This is a workaround. Reading directly from sys.argv at module import
    is fragile. A more robust solution would involve passing the parsed 'n'
    argument from the benchmark's main function down to where RequestFuncInput
    is created.
    """
    try:
        if "--n" in sys.argv:
            idx = sys.argv.index("--n")
            if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-"):
                n_value = sys.argv[idx + 1]
                return int(n_value)
    except (ValueError, IndexError):
        pass  # Fallback to 1 if parsing fails
    return 1


DEFAULT_N_BEAMS = _get_default_n_beams_from_argv()
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


async def patched_async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
    mm_position: Literal["first", "last"] = "last",
):
    """Add beam search parameters to request payload."""
    api_url = request_func_input.api_url
    _validate_api_url(api_url, "OpenAI Chat Completions API", "chat/completions")

    content = _get_chat_content(request_func_input, mm_position=mm_position)

    payload = {
        "model": request_func_input.model_name
        if request_func_input.model_name
        else request_func_input.model,
        "messages": [
            {"role": "user", "content": content},
        ],
        "use_beam_search": request_func_input.use_beam_search,
        "n": request_func_input.n,
        "temperature": 0.0,
        "max_tokens": request_func_input.output_len,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
        "vllm_xargs": {"begin_token": "<|sid_begin|>", "end_token": "<|sid_end|>"},
    }
    _update_payload_common(payload, request_func_input)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
    }
    _update_headers_common(headers, request_func_input)

    output = RequestFuncOutput()
    output.prompt_len = request_func_input.prompt_len

    generated_text = ""
    ttft = 0.0
    st = time.perf_counter()
    output.start_time = st
    most_recent_timestamp = st
    try:
        async with session.post(url=api_url, json=payload, headers=headers) as response:
            if response.status == 200:
                handler = StreamedResponseHandler()
                async for chunk_bytes in response.content.iter_any():
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue

                    messages = handler.add_chunk(chunk_bytes)
                    for message in messages:
                        # NOTE: SSE comments (often used as pings) start with
                        # a colon. These are not JSON data payload and should
                        # be skipped.
                        if message.startswith(":"):
                            continue

                        chunk = message.removeprefix("data: ")

                        if chunk != "[DONE]":
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            if choices := data.get("choices"):
                                content = choices[0]["delta"].get("content")
                                # First token
                                if ttft == 0.0:
                                    ttft = timestamp - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                generated_text += content or ""
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get("completion_tokens")

                            most_recent_timestamp = timestamp

                output.generated_text = generated_text
                output.success = True
                output.latency = most_recent_timestamp - st
            else:
                output.error = response.reason or ""
                output.success = False
    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


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
ASYNC_REQUEST_FUNCS["openai-chat"] = patched_async_request_openai_chat_completions
