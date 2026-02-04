
# Installation

## Prerequisites

- Python 3.11+
- Pip 23+
- CUDA 11.8+ (for GPU inference)
- 32GB+ RAM (recommended)

## Setup

1. **Clone the repository**

```bash
git clone https://github.com/JiusiServe/vllm-gr.git
cd vllm-gr
```

2. **Create virtual environment**

This step is not mandatory. You can use a previously installed environment.

```bash
uv venv --python 3.12
source .venv/bin/activate
python -m ensurepip --upgrade
ln -s -r .venv/bin/pip3 .venv/bin/pip
#make sure pip is installed correctly
pip --version
#pip 25.0.1 from /your-path/vllm-gr/.venv/lib/python3.12/site-packages/pip (python 3.12)
```

3. **Installation**

For standard installation, run
```bash
pip install -e .
```
Alternatively, for developer mode, run
```bash
pip install -e .[dev]
```

4. **Verify installation**

```bash
python -c "import benchmarks.open_one_rec; print('✓ Installation successful')"
```

## Quick Start

### API Server Mode (Production-like)

To run vllm-gr in a server--client configuration, use two terminals

### On the first terminal: Run the server

Use vllm-gr directly

```bash
vllm-gr serve \
  --model OpenOneRec/OneRec-1.7B \
  --max-logprobs 1024 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

Or use vllm serve with the plugin

```bash
vllm serve --gr \
  --model OpenOneRec/OneRec-1.7B \
  --max-logprobs 1024 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

### On the second terminal: Run single request against server with CLI

```bash
# CLI Single request Inference
curl -X POST "http://localhost:8000/v1/chat/completions" -H "Content-Type: application/json" \
  --data '{
    "model": "OpenOneRec/OneRec-1.7B",
    "messages": [
      {
        "role": "user",
        "content": "这是一个视频：<|sid_begin|><s_a_346><s_b_6566><s_c_5603><|sid_end|>，帮我总结一下这个视频讲述了什么内容"
      }
    ],
    "use_beam_search": true,  
    "n": 5,  
    "temperature": 0.5,  
    "max_tokens": 100,  
    "ignore_eos": false,  
    "length_penalty": 1.0  
  }'
```

Or use the [benchmark](benchmarks/README.md) to run a more complete benchmark test.
