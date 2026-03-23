# Throughput Benchmark

## Overview

This guide explains how to run the throughput benchmark using the [`one_rec_main.py`](../../benchmarks/open_one_rec/one_rec_main.py) script.

## Prerequisites

- Ensure that **vllm-gr** is installed. See the [Quickstart](../quickstart.md) guide.

- Ensure that you have downloaded the [OneRec benchmark dataset](README.md#open-one-rec-benchmark-dataset).

---

## Running the Benchmark

### 1. Offline Throughput Mode (Simple, Local)

```bash
# Single command — everything runs in one process
python -m benchmarks.open_one_rec.one_rec_main bench throughput \
  --gr \
  --model OpenOneRec/OneRec-1.7B \
  --max-logprobs 1024 \
  --dataset-name onerec \
  --dataset-path ./data \
  --task-types video \
  --num-prompts 100 \
  --output-json results/throughput.json  \
  --use-beam-search \
  --n 128
```

### 2. Online, API Server Mode (Production-like)

Run the server and client separately for a production‑style setup.

#### Terminal 1 — Start the vLLM API server

```bash

  vllm-gr  serve \
  --gr \
  --model OpenOneRec/OneRec-1.7B \
  --max-logprobs 1024 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

#### Terminal 2 — Run the throughput benchmark against the server

```bash

python -m benchmarks.open_one_rec.one_rec_main bench serve \
  --endpoint /v1/chat/completions \
  --backend openai-chat  \
  --model OpenOneRec/OneRec-1.7B  \
  --dataset-name  onerec \
  --dataset-path ./data  \
  --task-types video  \
  --num-prompts 100 \
  --use-beam-search \
  --n 128
```

## Usage

### Command Line Arguments

#### Server Core Arguments

| Argument         | Type | Default | Description                                                     |
|------------------|------|---------|-----------------------------------------------------------------|
| `--gr`           |      |         | Whether to run vllm-gr utilities                                |
| `--model`        | str  | -       | Model ID or path to load                                        |
| `--max-logprobs` | int  | -       | Max log Probabilities output by model -- to support beam search |

#### Core Arguments

| Argument  | Type | Default | Description              |
|-----------|------|---------|--------------------------|
| `--model` | str  | -       | Model ID or path to load |

#### Benchmark related parameters

| Argument         | Type | Default | Description                               |
|------------------|------|---------|-------------------------------------------|
| `--dataset-name` | str  | -       | Must be `onerec` for OneRec benchmark     |
| `--dataset-path` | str  | -       | Path to benchmark data directory          |
| `--num-prompts`  | int  | -       | Number of prompts to sample               |
| `--task-types`   | str  | -       | Comma-separated list of task types to run |
| `--output-json`  | str  | -       | Path to save benchmark results as JSON    |

#### Beam Search Parameters (Optional)

| Argument            | Type | Default | Description                          |
|---------------------|------|---------|--------------------------------------|
| `--use-beam-search` | flag | `False` | Enable beam search during generation |
| `--n`               | int  | -       | Number of beams for beam search      |

#### Arguments for Online benchmark

| Argument     | Type | Default       | Description                                       |
|--------------|------|---------------|---------------------------------------------------|
| `--endpoint` | str  | -             | API endpoint URL for online benchmark             |
| `--backend`  | str  | 'openai-chat' | Must be `openai-chat` for OneRec online benchmark |

## Expected Output

The output will include throughput metrics that can be used to evaluate the model's performance under load conditions.

```bash

============ Serving Benchmark Result ============
Successful requests:                     100
Failed requests:                         0
Benchmark duration (s):                  49.75
Total input tokens:                      241086
Total generated tokens:                  64000
Request throughput (req/s):              2.01
Output token throughput (tok/s):         1286.55
Peak output token throughput (tok/s):    4096.00
Peak concurrent requests:                100.00
Total token throughput (tok/s):          6132.96
---------------Time to First Token----------------
Mean TTFT (ms):                          43927.85
Median TTFT (ms):                        44508.24
P99 TTFT (ms):                           49620.72
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          0.02
Median TPOT (ms):                        0.01
P99 TPOT (ms):                           0.22
---------------Inter-token Latency----------------
Mean ITL (ms):                           0.05
Median ITL (ms):                         0.01
P99 ITL (ms):                            0.03
==================================================
```
