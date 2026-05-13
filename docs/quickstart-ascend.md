
# Installation for Ascend NPU

## Prerequisites

- Python 3.11
- Pip 23+
- CANN 8.5.0+
- Conda 24.11+
- 32GB+ RAM (recommended)

## Setup

1. **Clone the repository**

```bash
git clone https://github.com/JiusiServe/vllm-gr.git
cd vllm-gr
```

2. **Create environment**

### install CANN

- Follow miniconda installation procedure on [conda installation](https://www.anaconda.com/docs/getting-started/miniconda/install/linux-install) if not yet installed.
- For details about CANN installation on various platforms, please refer to [CANN installation](https://ascend.github.io/docs/sources/ascend/quick_install.html)
- This example was verified for 910B architecture

```bash
export MY_ENV=xGr-install # any valid env name will do
conda create -n ${MY_ENV}  -c conda-forge python=3.11 numactl gcc=10.3.0  gxx=10.3.0
conda activate ${MY_ENV}
conda install -c https://repo.huaweicloud.com/ascend/repos/conda/ ascend::cann-toolkit==8.5.0 ascend::cann-910b-ops==8.5.0
source ${CONDA_PREFIX}/Ascend/ascend-toolkit/set_env.sh
conda install -c https://repo.huaweicloud.com/ascend/repos/conda ascend::cann-nnal==8.5.0
source ${CONDA_PREFIX}/Ascend/nnal/atb/set_env.sh
```

3. **Install vllm-gr**

For standard installation, run

```bash
pip install .
```

Alternatively, for developer mode, run

```bash
pip install -e .[dev]
```

4. **Install vllm-ascend**

```bash
pip install vllm-ascend==0.14.0rc1
```

5. **Verify installation**

```bash
python -c "import benchmarks.open_one_rec; print('✓ Installation successful')"
```

## Quick Start

Refer to [quickstart](quickstart.md#quick-start) for deployment
