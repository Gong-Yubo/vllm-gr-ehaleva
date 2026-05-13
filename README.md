<h1 align="center">
  <img src="docs/assets/vLLM-GR.png" alt="vLLM-GR">
</h1>

<h3 align="center">
Efficient Generative Recommendation Serving
</h3>

---

vLLM-GR is an extension to [vLLM](https://github.com/vllm-project/vllm) designed specifically for **Generative Recommendation (GR) workloads**. It introduces beam-aware execution, memory optimizations, and scheduling strategies that target the unique features and constraints of large-scale recommendation systems built on LLMs.

Unlike traditional LLM serving—optimized for long, conversational generation—GR workloads require **massive beam widths, shared prefixes, short decoding steps, and ultra-low tail latency**. vLLM-GR builds on vLLM’s proven foundations (KV caching with PagedAttention, OpenAI-compatible serving) while adding GR-specific optimizations that can be enabled without impacting standard inference paths.

---

## Why vLLM-GR?

Generative recommendation systems shift the problem from *selecting items* to *generating personalized experiences*. This change stresses inference engines in ways standard serving stacks are not designed for:

* Large beam widths (often ≥128)
* Heavy prefix sharing across beams, which calls for KV reuse
* Short, structured decoding outputs
* Tight P99 latency targets (often <200ms)

vLLM-GR addresses these challenges by introducing:

* **Beam-aware KV cache layouts** (shared prefix + per-beam suffix)
* **BeamAttention** mechanism to eliminate redundant memory traffic
* **Efficient beam search and pruning** on CPU and GPU
* **Overlapped CPU/GPU scheduling** to reduce non-compute overheads

All features are designed to be **opt-in**, preserving vLLM’s general-purpose behavior when GR is disabled.

## Supported Hardware

vLLM-GR runs on **NVIDIA GPUs** as well as **Huawei Ascend NPUs**.

| Backend | Notes |
| --- | --- |
| NVIDIA GPU | Relies on CUDA 12+ |
| Huawei Ascend NPU | Supported via the CANN runtime |

## Getting Started

Install vllm-gr with pip from source

```bash
git clone https://github.com/JiusiServe/vllm-gr.git
cd vllm-gr
pip install --editable .
```

Visit the documentation pages to learn more

* [`Quickstart`](docs/quickstart.md):
  Instructions for setting up vLLM-GR and an example showing how to run a GR workload with vLLM-GR enabled.

* [`Quickstart-Ascend`](docs/quickstart-ascend.md):
  Instructions for setting up vLLM-GR on Huawei Ascend NPU environment.

* [`Benchmark`](docs/benchmarks/README.md):
  Performance results and methodology, including GR-focused benchmarks (e.g. Open One Rec).

---

## Project Status

vLLM-GR is under active development.

The long-term goal is to upstream mature components into the main vLLM project while maintaining vLLM-GR as a reference implementation for GR workloads.

---

## Scope and Design Principles

* vLLM remains **general-purpose**; GR features are isolated and optional.
* Improvements must introduce **zero performance regression** when GR is disabled.

---

## Contributing

Contributions, design discussions, and performance investigations are welcome. If you are working on large-scale recommendation systems or GR workloads, your feedback is especially valuable.

---

## Acknowledgements

vLLM-GR builds on the vLLM ecosystem and the broader open-source community advancing efficient LLM inference.

Let’s build something fast, scalable, and a little bit audacious!
