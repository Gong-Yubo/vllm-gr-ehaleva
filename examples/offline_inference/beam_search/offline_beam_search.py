# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import time

from vllm.entrypoints.llm import LLM

try:
    from vllm_gr.sampling_params import BeamSearchParams
except ImportError:
    from vllm.sampling_params import BeamSearchParams


def beam_search_example(model, beam_width: int, compare: str):
    with open("../../../tests/resources/single_one_rec_prompt.txt", "r", encoding="utf-8") as f:
        long_recommendation_texts = [f.read()]
    prompts = [
        {"prompt": long_recommendation_texts[0]},
    ]
    # Initialize LLM
    llm = LLM(
        model=model,
        max_logprobs=beam_width * 2,
        max_num_seqs=512 * 5,
        max_num_batched_tokens=4096 * 5,
    )

    params = BeamSearchParams(beam_width=beam_width, max_tokens=5)
    if hasattr(BeamSearchParams, "begin_token"):
        params.begin_token = "<|sid_begin|>"
        params.end_token = "<|sid_end|>"

    if compare != "none":
        # Send beam search request to Vanilla vLLM
        start = time.perf_counter()
        outputs = llm.beam_search(prompts, params)
        end = time.perf_counter()
        if compare in ["latency", "both"]:
            print(f"Vanilla vLLM with BW={beam_width} took {end - start:.6f} seconds")

        if compare in ["output", "both"]:
            # Print results
            for index, output in enumerate(outputs):
                for i in range(len(output.sequences)):
                    generated_text = output.sequences[i].text
                    print(
                        f"Generated text {index}:{i} : {generated_text[len(long_recommendation_texts[index]) :]}"
                    )

    # import vllm-gr.init module to apply the vllm-gr patch to vLLM.
    import vllm_gr.init  # noqa: F401

    # Send beam search request to vllm-gr
    start = time.perf_counter()
    outputs = llm.beam_search(prompts, params)
    end = time.perf_counter()
    print(f"vLLM-gr with BW={beam_width} took {(end - start):.6f} seconds")

    if compare in ["output", "both"]:
        # Print results
        for index, output in enumerate(outputs):
            for i in range(len(output.sequences)):
                generated_text = output.sequences[i].text
                print(
                    f"Generated text {index}:{i} : {generated_text[len(long_recommendation_texts[index]) :]}"
                )


def main():
    parser = argparse.ArgumentParser(description="Offline Beam Search Runner")

    parser.add_argument("--model", type=str, required=True, help="Model name or path")

    parser.add_argument("--beam_width", type=int, default=128, help="Beam width for beam search")

    parser.add_argument(
        "--compare",
        type=str,
        choices=["none", "output", "latency", "both"],
        default="none",
        help="Comparison to vanilla vLLM mode: none, output, latency, or both",
    )

    args = parser.parse_args()
    beam_search_example(model=args.model, beam_width=args.beam_width, compare=args.compare)


if __name__ == "__main__":
    main()
