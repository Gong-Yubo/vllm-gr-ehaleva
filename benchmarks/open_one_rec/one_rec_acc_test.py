import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import requests

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("Error: vLLM is not installed. Please install it to run this benchmark.")
    sys.exit(1)

from benchmarks.open_one_rec.open_one_rec_loader import TASK_REGISTRY, DataLoaderWrapper

try:
    from vllm_gr.sampling_params import BeamSearchParams
except ImportError:
    from vllm.sampling_params import BeamSearchParams

logger = logging.getLogger(__name__)


def get_sampling_params(gen_config: Dict[str, Any]) -> SamplingParams:
    """
    Construct vLLM SamplingParams from the task generation configuration.
    """
    # Extract standard vLLM parameters
    n = gen_config.get("num_return_sequences", 1)
    max_tokens = gen_config.get("max_new_tokens", 5)
    temperature = gen_config.get("temperature", 1.0)
    top_p = gen_config.get("top_p", 1.0)
    top_k = gen_config.get("top_k", -1)
    frequency_penalty = gen_config.get("frequency_penalty", 0.0)
    presence_penalty = gen_config.get("presence_penalty", 0.0)

    # Handle logprobs
    logprobs = None
    if gen_config.get("return_logprobs", False):
        # vLLM expects an integer for logprobs (number of top logprobs to return)
        # The config has 'logprobs': 10000, which is quite high, we clamp it for safety in this example
        logprobs = gen_config.get("logprobs", 1)

    # Handle Beam Search
    num_beams = gen_config.get("num_beams", 1)

    # Extract custom parameters (not passed to vLLM, but read as requested)
    num_return_thinking = gen_config.get("num_return_thinking_sequences", 0)
    max_thinking_tokens = gen_config.get("max_new_thinking_tokens", 0)

    logger.info(f"  [Config Read] num_return_sequences: {n}")
    logger.info(f"  [Config Read] num_beams: {num_beams}")
    logger.info(
        f"  [Config Read] num_return_thinking_sequences: {num_return_thinking} (Custom param)"
    )
    logger.info(f"  [Config Read] max_new_thinking_tokens: {max_thinking_tokens} (Custom param)")

    return SamplingParams(
        n=n,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        logprobs=logprobs,
        stop_token_ids=None,  # Could extract stop tokens if present in config
    )


class MockCompletionOutput:
    def __init__(self, text, logprobs=None):
        self.text = text
        self.logprobs = logprobs


class MockRequestOutput:
    def __init__(self, prompt, outputs: List[MockCompletionOutput]):
        self.prompt = prompt
        self.outputs = outputs


def run_api_inference(
    prompts: List[str], model: str, api_url: str, api_key: str, gen_config: Dict[str, Any]
) -> List[MockRequestOutput]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # Map generation config to API parameters
    api_params = {
        "model": model,
        "use_beam_search": True if (gen_config.get("num_beams", 1) > 1) else False,
        "max_tokens": gen_config.get("max_new_tokens", 1024),
        "temperature": gen_config.get("temperature", 0.0),
        "top_p": gen_config.get("top_p", 1.0),
        "n": gen_config.get("num_return_sequences", 1),
        "vllm_xargs": {"begin_token": "<|sid_begin|>", "end_token": "<|sid_end|>"},
    }

    def _send_request(prompt):
        response = None
        payload = {"messages": [{"role": "user", "content": prompt}], **api_params}
        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            # Extract content
            if "choices" in data and data["choices"]:
                mock_outputs = []
                for choice in data["choices"]:
                    message = choice.get("message", {})
                    content = message.get("content", "")
                    reasoning = message.get("reasoning_content", "")  # Support for reasoning models

                    full_text = ""
                    if reasoning:
                        full_text += reasoning
                    if content:
                        full_text += content
                    mock_outputs.append(MockCompletionOutput(full_text))
                return MockRequestOutput(prompt, mock_outputs)
            return MockRequestOutput(prompt, [MockCompletionOutput("")])
        except Exception as e:
            status_code = response.status_code if response is not None else "N/A"
            response_text = response.text if response is not None else "N/A"
            logger.error(f"Request failed. Status code: {status_code}")
            logger.error(f"[DEBUG] Response text: {response_text}")
            logger.error(f"Request failed for prompt: {prompt[:50]}... Error: {e}")
            return MockRequestOutput(prompt, [MockCompletionOutput("")])

    # Run in parallel
    logger.info(f"Sending {len(prompts)} requests to {api_url}...")
    with ThreadPoolExecutor(max_workers=min(len(prompts), 16)) as executor:
        results = list(executor.map(_send_request, prompts))

    return results


def main():
    parser = argparse.ArgumentParser(description="Run OpenOneRec Benchmark")
    parser.add_argument(
        "--model", type=str, default="OpenOneRec/OneRec-1.7B", help="Path to model or HF hub name"
    )
    parser.add_argument("--dataset-name", default="onerec", help="Name of the dataset to use.")
    parser.add_argument("--dataset-path", type=str, default="./data", help="Path to data directory")
    parser.add_argument(
        "--task-type", type=str, required=True, choices=TASK_REGISTRY.keys(), help="Task to run"
    )
    parser.add_argument("--num-prompts", type=int, help="Number of samples to run")
    parser.add_argument("--temperature", type=float, help="Temperature for generation")
    parser.add_argument(
        "--endpoint", type=str, default=None, help="Endpoint URL for OpenAI-compatible API"
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="Host for OpenAI-compatible API"
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend", default="openai-chat", help="Name of the backend to use.")
    parser.add_argument(
        "--api-key", type=str, default="EMPTY", help="API Key for OpenAI-compatible API"
    )
    parser.add_argument(
        "--use-beam-search", action="store_true", help="Whether to use beam search for generation."
    )
    parser.add_argument("--n", type=int, help="Beam width to use if beam search is enabled.")
    parser.add_argument("--gr", action="store_true", help="Whether to use vllm-gr.")

    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Directory to save debug evaluation files.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for evaluators.")

    args = parser.parse_args()

    # 1. Load Configuration
    if args.gr:
        logger.info("Detected --gr flag, enabling vllm-gr mode for accuracy test.")
        from vllm_gr.patch import run_patch

        run_patch()

    if args.task_type not in TASK_REGISTRY:
        print(f"Task {args.task_type} not found in configuration.")
        print(f"Available tasks: {list(TASK_REGISTRY.keys())}")
        sys.exit(1)

    task_reg = TASK_REGISTRY[args.task_type]
    task_config = task_reg.config
    gen_config = task_config.get("generation_config", {})

    logger.info(f"Loaded configuration for task: {task_config['name']}")
    logger.info(f"Description: {task_config.get('description')}")

    if args.temperature is not None:
        gen_config["temperature"] = args.temperature
        logger.info(f"Using temperature from CLI parameter: {args.temperature}")
    if args.n is not None:
        gen_config["num_beams"] = args.n
        gen_config["num_return_sequences"] = args.n  # Ensure num_return_sequences matches num_beams
        logger.info(f"Using beam search with beam width from CLI parameter: {args.n}")
    else:
        logger.info(
            f"Using beam search according to the config with beam width: {gen_config.get('num_beams', 1)}"
        )

    # 2. Initialize Engine
    llm = None
    if args.endpoint:
        logger.info(f"\nUsing OpenAI-compatible API at: {args.endpoint}")
        logger.info(f"Model: {args.model}")
    else:
        logger.info(f"\nInitializing vLLM with model: {args.model}")
        try:
            llm = LLM(
                model=args.model,
                trust_remote_code=True,
                max_logprobs=gen_config.get("num_beams", 1) * 2,
            )
        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            sys.exit(1)

    # 3. Prepare Data
    # prompts = mock_data_loader(task_config, args.num_prompts)  # Replace with real data loading
    logger.info(f"Loading real data from {args.dataset_path}...")
    loader = DataLoaderWrapper(
        model_path=args.model, benchmark_version="v1.0", data_dir=args.dataset_path
    )
    split = task_config.get("splits", ["test"])[0]
    dataset = loader.load_data(
        task_name=args.task_type,
        split=split,
        sample_size=args.num_prompts if args.num_prompts else None,
    )

    prompts = []
    references = {}
    sample_ids = []

    for sample_id, item in dataset.items():
        prompt = item["prompt"]
        prompts.append(prompt)
        references[sample_id] = item["ground_truth"]
        sample_ids.append(sample_id)

    # 4. Run Inference
    logger.info("\nStarting Inference...")
    gen_start = time.perf_counter()
    if args.endpoint:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        outputs = run_api_inference(prompts, args.model, api_url, args.api_key, gen_config)
    else:
        params = BeamSearchParams(
            beam_width=gen_config["num_beams"],
            max_tokens=gen_config["max_new_tokens"],
            temperature=gen_config["temperature"],
        )
        # Add vLLM-GR specific parameters if available
        if hasattr(BeamSearchParams, "begin_token"):
            params.begin_token = "<|sid_begin|>"
            params.end_token = "<|sid_end|>"
        prompts_dict = [{"prompt": prompt} for prompt in prompts]

        raw_offline_outputs = llm.beam_search(prompts_dict, params)

        # Normalize beam_search outputs to match MockRequestOutput format
        outputs = []
        for i, raw_output in enumerate(raw_offline_outputs):
            # beam_search returns BeamSearchOutput with .outputs containing BeamSearchSequence objects
            # Each BeamSearchSequence has .text attribute
            mock_completions = []
            prompt_len = len(prompts[i])
            if hasattr(raw_output, "sequences"):
                for beam_seq in raw_output.sequences:
                    text = (
                        beam_seq.text[prompt_len:] if hasattr(beam_seq, "text") else str(beam_seq)
                    )
                    logprobs = getattr(beam_seq, "cum_logprob", None)
                    mock_completions.append(MockCompletionOutput(text, logprobs))
            else:
                # Fallback if format is different
                mock_completions.append(MockCompletionOutput(str(raw_output)))

            outputs.append(MockRequestOutput(prompts[i], mock_completions))

    gen_end = time.perf_counter()

    # Print a sample output
    if outputs:
        logger.info("\nSample Output [0]:")
        logger.info(f"Prompt: {outputs[0].prompt}")
        logger.info(f"Generated Text: {outputs[0].outputs[0].text}")
        if outputs[0].outputs[0].logprobs:
            logger.info(f"Logprobs available: {outputs[0].outputs[0].logprobs}")

    # 5. Run Evaluation
    logger.info("\n--- Running Evaluation ---")

    # Prepare samples dictionary in the format expected by the evaluator.
    # This mimics the structure of a test_generated.json file.
    samples_dict = {}
    for i, output in enumerate(outputs):
        sample_id = sample_ids[i]

        # Get original data from the loaded dataset
        original_sample = dataset.get(sample_id)
        if not original_sample:
            logger.warning(
                f"Warning: Could not find original sample data for sample_id {sample_id}. Skipping."
            )
            continue

        # Get generations from the model output
        if len(output.outputs) > 1:
            generations = [o.text for o in output.outputs]
        else:
            generations = [output.outputs[0].text]

        samples_dict[sample_id] = {
            "prompt": original_sample["prompt"],
            "generations": generations,
            "ground_truth": original_sample["ground_truth"],
            "metadata": original_sample.get("metadata", {}),
        }

    # Instantiate evaluator and compute metrics
    evaluator_class = task_reg.evaluator_class
    evaluator = evaluator_class(
        samples=samples_dict,
        task_name=args.task_type,
        task_config=task_config,
        data_dir=args.dataset_path,
        predictions_dir=args.result_dir,
        debug=args.debug,
    )
    metrics, per_sample_metrics = evaluator.evaluate()

    print(f"\n--- Evaluation Results for {args.task_type} on model {args.model} ---")

    model_params = {}
    model_params["model"] = args.model
    model_params["task"] = args.task_type
    model_params["beam_width"] = gen_config.get("num_beams", 1)
    model_params["temperature"] = gen_config.get("temperature", 0.0)
    model_params["time_per_sample"] = (gen_end - gen_start) / len(prompts) if prompts else 0.0
    metrics = {**model_params, **metrics}
    print(json.dumps(metrics, indent=2))
    if args.result_dir is not None:
        os.makedirs(args.result_dir, exist_ok=True)
        bw = model_params["beam_width"]
        filename = f"{args.model.split('/')[-1]}_{args.task_type}_bw{bw}_{len(prompts)}.json"
        output_path = os.path.join(args.result_dir, filename)
        # Save metrics (a Python dict) as JSON
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
