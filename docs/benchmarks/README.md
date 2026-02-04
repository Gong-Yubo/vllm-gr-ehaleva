# OneRec Benchmark Documentation

OneRec Benchmark is a comprehensive Recommendation Instruction-Following benchmark for evaluating large language models on multiple recommendation and understanding tasks. Find more about [OpenOneRec benchmark](https://github.com/Kuaishou-OneRec/OpenOneRec/tree/main)

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Task Types](#task-types)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)

## Overview

OneRec is a benchmark suite designed to evaluate generative recommendation models across distinct task types. The most relevant of those are:

| Task Type | Category | Description | Sample Size |
|-----------|----------|-------------|------------|
| `ad` | Recommendation | Predict ad engagement | 500 |
| `product` | Recommendation | Product recommendation reasoning | 1000 |
| `label_cond` | Recommendation | Conditional label prediction | 800 |
| `video` | Recommendation | Video recommendation scoring | 1500 |
| `interactive` | Recommendation | Interactive engagement prediction | 600 |
| `label_pred` | Classification | User engagement classification | 346190 |

### Key Features

- **Multi-task evaluation**: Benchmark models across diverse recommendation scenarios
- **Flexible task selection**: Run all or specific task types
- **Chat template support**: Custom prompt formatting with Jinja2 templates
- **Thinking mode**: Optional extended reasoning for capable models (e.g., Qwen3)
- **Comprehensive metrics**: Task-specific evaluation (AUC, F1, BERTScore, WIP scoring)
- **Beam search support**: Configurable beam search parameters for generation

## Quick Start

OneRec supports two operation modes. Choose based on your needs:

### Mode 1: Throughput Mode (Simple, Local)

For quick local benchmarking without an API server:

```bash
# Single command - everything in one process
python -m benchmarks.open_one_rec.one_rec_main bench throughput \
  --dataset-name onerec \
  --model OpenOneRec/OneRec-1.7B \
  --dataset-path ./data \
  --task-types video \
  --num-prompts 1000 \
  --output-json results/throughput.json  

```

**Best for**: Quick testing, development, maximizing throughput

### Mode 2: API Server Mode (Production-like)

For production-like testing with separate server and client:

```bash
# Terminal 1: Start vLLM API server
  vllm-gr serve \
  --model OpenOneRec/OneRec-1.7B \
  --max-logprobs 1024 \
  --default-chat-template-kwargs '{"enable_thinking": false}'


# Terminal 2: Run benchmark against server
python -m benchmarks.open_one_rec.one_rec_main bench serve \
  --endpoint http://localhost:8000/v1 \
  --model OpenOneRec/OneRec-1.7B \
  --backend openai-chat \
  --dataset-name  onerec \
  --dataset-path ./data \
  --task-types video \
  --num-prompts 1000
```

**Best for**: Production-like testing, distributed deployment, load testing

### With Beam Search

Both modes support beam search:

```bash
# Throughput mode with beam search
python -m  benchmarks.open_one_rec.one_rec_main bench throughput \
  --model OpenOneRec/OneRec-1.7B \
  --dataset-name onerec \
  --dataset-path ./data \
  --task-types video \
  --num-prompts 1000 \
  --use-beam-search \
  --n 8 \
  --output-json results/throughput.json  


# API mode with beam search (requires API server)
python -m benchmarks.open_one_rec.one_rec_main bench serve \
  --endpoint http://localhost:8000/v1 \
  --model OpenOneRec/OneRec-1.7B \
  --backend openai-chat \
  --dataset-name onerec \
  --dataset-path ./data \  
  --task-types video \
  --num-prompts 1000 \
  --use-beam-search \
  --n 8 \
  --save-result \
  --result-dir results/serve \  
  --save-detailed  
```

### Operation Modes Comparison

| Aspect | Throughput Mode | API Server Mode |
|--------|---|---|
| Setup | 1 command | 2 terminals |
| Latency | Lower | Higher (+HTTP) |
| Throughput | Higher | Lower |
| Production-like | No | Yes |
| Network overhead | None | Yes (HTTP) |
| Use case | Local testing | Deployment simulation |

## Installation

You can refer also to [`Quickstart`](../quickstart.md) menu.

### Prerequisites

- Python 3.11+
- Pip 23+
- CUDA 11.8+ (for GPU inference)
- 32GB+ RAM (recommended)

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/JiusiServe/vllm-gr.git
cd vllm-gr
```

2. **Installation**

```bash
pip install -e .
```

3. **Verify installation**

```bash
python -c "import benchmarks.open_one_rec; print('✓ Installation successful')"
```

## Open One Rec Benchmark Dataset

### Get Dataset

Use your huggingface username to approve this link before you download:

<https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF>

```bash
mkdir -p OpenOneRec  
cd OpenOneRec
git lfs install  
git clone https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF ./data
cd -
```

### Data Structure

OneRec expects data in **Parquet format** with the following schema:

```text
├── data/
│   ├── rec_reason/
│   │   └── rec_reason_test.parquet
│   ├── item_understand/
│   │   └── item_understand_test.parquet
│   └── ...
```

### Parquet Schema

Each Parquet file must contain these columns:

| Column | Type | Description |
|--------|------|-------------|
| `messages` | str/list | Chat messages in JSON format or list of dicts |
| `metadata` | str/dict | JSON metadata containing answer and context |

## Usage

### Command Line Arguments

#### Core Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset-name` | str | - | Must be `onerec` for OneRec benchmark |
| `--model` | str | - | Model ID or path to load |
| `--dataset-path` | str | None | Path to benchmark data directory |
| `--num-prompts` | int | 1000 | Number of prompts to sample |
| `--disable-shuffle` | bool | False | Disable shuffling of samples |

#### OneRec Specific Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--task-types` | str | All 8 tasks | Comma-separated list of task types to run |
| `--use-beam-search` | bool | False | Enable beam search during generation |
| `--n` | int | 8 | Number of beams for beam search |

#### Generation Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max-tokens` | int | 128 | Maximum tokens to generate |
| `--temperature` | float | 0.8 | Sampling temperature |
| `--top-p` | float | 0.95 | Nucleus sampling parameter |

## API Server Interactions

### Message Format

Messages follow the OpenAI chat format:

```json
[
  {
    "role": "system",
    "content": "You are a helpful recommendation assistant."
  },
  {
    "role": "user",
    "content": "Recommend a video based on user history..."
  }
]
```

Or with multi-modal content:

```json
[
  {
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe this video"},
      {"type": "image_url", "image_url": {"url": "..."}}
    ]
  }
]
```

### Metadata Format

```json
{
  "answer": "Expected model output",
  "user_id": 12345,
  "item_id": 67890,
  "context": "Additional context",
  "ground_truth": "Expected answer"
}
```

### Example Data Row

```python
{
    "messages": '[{"role": "user", "content": "Recommend a video"}]',
    "metadata": '{"answer": "Video ID: ABC123", "user_id": 1001, "item_id": 501}'
}
```

### Creating Sample Data

```python
import pandas as pd
import json

def create_sample_data(output_path: str, num_samples: int = 100):
    """Create sample OneRec data."""
    records = []

    for i in range(num_samples):
        record = {
            "messages": json.dumps([
                {"role": "user", "content": f"Sample prompt {i}"}
            ]),
            "metadata": json.dumps({
                "answer": f"Sample answer {i}",
                "user_id": i,
                "item_id": i * 10,
            })
        }
        records.append(record)

    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    print(f"✓ Created sample data: {output_path}")

# Usage
create_sample_data("./data/video/video_test.parquet", 100)
```

## Task Types

### 1. Ad Engagement (`ad`)

Predict and explain ad engagement for user interactions.

- **Category**: Recommendation
- **Sample Size**: 500
- **Output**: Engagement prediction
- **Metrics**: AUC, accuracy

### 2. Product Recommendation (`product`)

Generate product recommendations with reasoning.

- **Category**: Recommendation
- **Sample Size**: 1000
- **Output**: Product ID + recommendation reason
- **Metrics**: Precision@K, NDCG

### 3. Label Conditional (`label_cond`)

Predict item labels based on conditional information.

- **Category**: Recommendation
- **Sample Size**: 800
- **Output**: Predicted label
- **Metrics**: Accuracy, F1

### 4. Video Recommendation (`video`)

Score and rank videos for recommendation.

- **Category**: Recommendation
- **Sample Size**: 1500
- **Output**: Video ranking/scores
- **Metrics**: NDCG, MAP

### 5. Interactive (`interactive`)

Predict engagement in interactive scenarios.

- **Category**: Recommendation
- **Sample Size**: 600
- **Output**: Engagement prediction
- **Metrics**: AUC, F1

### 6. Label Prediction (`label_pred`)

Binary classification (discrimination) of user engagement labels.

- **Category**: Classification
- **Sample Size**: 346190
- **Output**: Binary prediction (yes/no)
- **Metrics**: AUC, weighted AUC
- **Special**: Uses logprobs-based classification

```python
# Example
# Expected output: "是" (yes) or "否" (no)
# Extracted from model logprobs
```

## Evaluation

### Evaluation Workflow

1. **Load Data**: Benchmark loads samples from Parquet files
2. **Generate Responses**: Model generates outputs using configured parameters
3. **Extract Metrics**: Task-specific evaluators compute metrics
4. **Save Results**: Results saved to JSON files

### Evaluator Types

For all task types listed in the [Task Types](#task-types) section
the `RecommendationEvaluator` is the default Evaluator

| Evaluator | Tasks | Metrics |
|-----------|-------|---------|
| `RecommendationEvaluator` | ad, product, label_cond, video, interactive | BLEU, ROUGE, BERTScore |


### Running Evaluation

```bash
# Evaluation runs automatically after generation
# Results saved to output-json

python -m benchmarks.open_one_rec.one_rec_main bench throughput \
  --model OpenOneRec/OneRec-1.7B \
  --dataset-name onerec \
  --dataset-path ./data \
  --num-prompts 100 \
  --output-json results/throughput  
```

### Result Format

```json
{
  "model": "OpenOneRec/OneRec-1.7B",
  "task_type": "product",
  "metrics": {
    "bleu": 0.45,
    "rouge_1": 0.52,
    "bertscore_f1": 0.68,
    "average": 0.55
  },
  "samples": 1000,
  "timestamp": "2026-01-20T10:30:00"
}
```

## Troubleshooting

### Common Issues

#### 1. Data Not Found

```text
FileNotFoundError: Data file not found: ./data/video/video_test.parquet
```

**Solution**: Ensure data directory structure matches task names:

```bash
ls -R ./data/
# Output should show:
# ./data/rec_reason/rec_reason_test.parquet
# ./data/item_understand/item_understand_test.parquet
# ...
```

#### 2. Tokenizer Not Found

```text
RuntimeError: Failed to load tokenizer from /path/to/model
```

**Solution**: Verify model path and check tokenizer compatibility:

```bash
python -c "from transformers import AutoTokenizer; \
  AutoTokenizer.from_pretrained('your-model-id', trust_remote_code=True)"
```

#### 3. Out of Memory

```text
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solution**: Reduce batch size and max tokens:

```bash
--num-prompts 100 \
--max-tokens 64 \
--dtype float16
```

#### 4. Beam Search Not Working

```text
extra_body parameters ignored
```

**Solution**: Use vLLM serving endpoint instead of local inference:

```bash
# Start server
vllm serve OpenOneRec/OneRec-1.7B --gpu-memory-utilization 0.9

# Use endpoint
--endpoint http://localhost:8000/v1
```

### Debug Mode

Enable verbose logging:

```bash
python -m benchmarks.open_one_rec.one_rec_main bench throughput \
  --model OpenOneRec/OneRec-1.7B \
  --dataset-name onerec \
  --dataset-path ./data \
  --task-types ad,label_pred,video \
  --num-prompts 100 \
  --use-beam-search \
  --n 8 \
  --output-json results/throughput
  -v  # Verbose logging
```

### Validate Setup

```bash
# Test imports
python -c "
from benchmarks.open_one_rec import OneRecDataset
from benchmarks.open_one_rec.open_one_rec_loader import BaseLoader
print('✓ All imports successful')
"

# Test data loading
python -c "
from benchmarks.open_one_rec import OneRecDataset
dataset = OneRecDataset(
    task_types=['video'],
    model_path='OpenOneRec/OneRec-1.7B',
    dataset_path='./data'
)
print(f'✓ Loaded {len(dataset.data)} samples')
"
```


## License

This benchmark is built upon [OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec), which is licensed under Apache 2.0.

## Contact & Support

For issues, questions, or contributions, please open an issue on [GitHub](https://github.com/JiusiServe/vllm-gr/issues).
