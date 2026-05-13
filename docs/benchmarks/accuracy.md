# Accuracy Benchmark

## Overview

This guide explains how to run the accuracy benchmark using the [`one_rec_acc_test.py`](../../benchmarks/open_one_rec/one_rec_acc_test.py) script.

## Prerequisites

- Ensure that **vllm-gr** is installed. See the [Quickstart](../quickstart.md) guide.

- Ensure that you have downloaded the [OneRec benchmark dataset](README.md#open-one-rec-benchmark-dataset).

---

## Running the Benchmark

### 1. Offline Accuracy Mode (Simple, Local)

```bash
# Single command — everything runs in one process
python -m benchmarks.open_one_rec.one_rec_acc_test \
  --model OpenOneRec/OneRec-1.7B \
  --dataset-path ./data \
  --task-type video \
  --num-prompts 1000 \
  --result-dir results/acc_test_results
```

### 2. Online, API Server Mode (Production-like)

Run the server and client separately for a production‑style setup.

#### Terminal 1 — Start the vLLM API server

```bash
vllm serve \
  --gr \
  --model OpenOneRec/OneRec-1.7B \
  --max-logprobs 1024 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

#### Terminal 2 — Run the accuracy benchmark against the server

```bash
python -m benchmarks.open_one_rec.one_rec_acc_test \
  --endpoint /v1/chat/completions \
  --backend openai-chat \
  --model OpenOneRec/OneRec-1.7B \
  --dataset-name  onerec \
  --dataset-path ./data \
  --task-type video \
  --num-prompts 1000 \
  --result-dir results/acc_test_results

```

optionally, one can modify the beam width from the default dataset definition. See [Benchmark related parameters](#benchmark-related-parameters) below.

## Usage

### Command Line Arguments

#### Core Arguments

| Argument  | Type | Default                | Description              |
|-----------|------|------------------------|--------------------------|
| `--model` | str  | OpenOneRec/OneRec-1.7B | Model ID or path to load |

#### Benchmark related parameters

| Argument            | Type  | Default          | Description                                                               |
|---------------------|-------|------------------|---------------------------------------------------------------------------|
| `--dataset-name`    | str   | onerec           | Must be `onerec` for OneRec benchmark                                     |
| `--dataset-path`    | str   | ./data           | Path to benchmark data directory                                          |
| `--num-prompts`     | int   | 1000             | Number of prompts to sample                                               |
| `--task-type`       | str   | -                | Task to run (ad, product, label_cond, video, interactive, label_pred)     |
| `--result-dir`      | str   | -                | Directory to save debug evaluation files                                  |
| `--debug`           | -     | -                | Enable debug mode for evaluators                                          |
| `--use-beam-search` | flag  | -                | Enable beam search during generation                                      |
| `--n`               | int   | dataset-specific | Override the dataset default beam width (number of beams) for beam search |
| `--temperature`     | float | dataset-specific | Override the dataset default temperature                                  |

#### Arguments for Online benchmark

| Argument     | Type | Default       | Description                                                            |
|--------------|------|---------------|------------------------------------------------------------------------|
| `--endpoint` | str  | -             | API endpoint URL for the online benchmark (e.g., /v1/chat/completions) |
| `--backend`  | str  | 'openai-chat' | Must be `openai-chat` for OneRec online benchmark                      |
| `--api-key`  | str  | -             | API Key for OpenAI-compatible API                                      |

## Expected Output

The benchmark prints accuracy metrics summarizing model performance.
Example:

```bash

...
...
--- Evaluation Results ---
{
  "total_samples": 100,
  "pass@1": 0.03,
  "position1_pass@1": 0.0,
  "recall@1": 0.004523809523809524,
  "pass@32": 0.16,
  "position1_pass@32": 0.04,
  "recall@32": 0.024341269841269844,
  "pid_pass@1": 0.03,
  "pid_position1_pass@1": 0.0,
  "pid_recall@1": 0.004523809523809524,
  "pid_pass@32": 0.16,
  "pid_position1_pass@32": 0.04,
  "pid_recall@32": 0.024341269841269844,
  "select_k_strategy": "first_k",
  "evaluation_mode": "both",
  "sid_to_pid_strategy": "most_popular_after_downsampling"
}
```
