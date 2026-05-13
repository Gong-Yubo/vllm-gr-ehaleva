#!/bin/bash
set -euo pipefail
# test development suite
./tests/development/test_resource.sh
# test pre-commit
source "./tools/resource.sh"
idle_gpu=$(get_gpu)
rc=$?
if [ $rc -ne 0 ] || [ -z "$idle_gpu" ]; then
    echo "No GPUs found that are 0% utilized, using < 16MiB memory, and error-free. (exit code: $rc)"
    exit 1
fi
cuda_dev=$idle_gpu
echo "running on device $cuda_dev"
CUDA_VISIBLE_DEVICES=$cuda_dev \
pre-commit run --all-files
