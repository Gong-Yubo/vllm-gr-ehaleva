# Offline Beam Search Demo

This directory contains a standalone example demonstrating how to run beam search in **offline mode** using the `vllm-gr` framework for generative recommendation models.

## Installation

Follow the instructions in  
[`docs/quickstart.md`](../../../docs/quickstart.md)  
to install the required dependencies and set up your environment.

## Running the Example

This example shows how to run offline beam search on the **OneRec‑1.7B** model using `vllm-gr`.

```bash
python offline_beam_search.py \
    --model OpenOneRec/OneRec-1.7B \
    --beam_width 128 \
    --compare latency
```

Key arguments:
- `--model`: Name or path of the model to load (required).
- `--beam_width`: Beam width used during beam search (default: 128).
- `--compare`: Comparison mode against vanilla vLLM. Options: none, latency, output, both (default: none).

# Expected result
The script prints the latency of running the predefined input through vllm-gr, and—if comparison is enabled—also reports the latency of vanilla vLLM.

Example:
```bash
python offline_beam_search.py --model OpenOneRec/OneRec-1.7B --compare latency
...
...
Vanilla vLLM with BW=128 took 4.882346 seconds
vLLM-gr with BW=128 took 0.627150 seconds
```
