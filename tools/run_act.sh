#!/bin/bash
set -euo pipefail

source "./tools/resource.sh"
idle_gpu=$(get_gpu 4)


if [ -z "$idle_gpu" ]; then
    echo "No GPUs found that are 0% utilized, using < 4MiB memory, and error-free."
    exit 1
fi

# Lock the GPU to prevent other scripts from using it
lock_pid=$(lock_gpu "$idle_gpu")
if [ -z "$lock_pid" ]; then
    echo "Failed to lock GPU $idle_gpu"
    exit 1
fi

# Ensure we unlock the GPU on exit
trap 'echo "Unlocking GPU $idle_gpu (PID: $lock_pid)..."; unlock_gpu "$lock_pid"' EXIT

cuda_dev=$idle_gpu
echo "running on device $cuda_dev"
if [ -z "${HF_TOKEN:-}" ]; then
    token=""
else
    secret_file=$(mktemp)
    echo "HF_TOKEN=$HF_TOKEN" > "$secret_file"
    token="--secret-file $secret_file"
    # Update trap to also clean up the secret file
    trap 'rm -f "$secret_file"; echo "Unlocking GPU $idle_gpu (PID: $lock_pid)..."; unlock_gpu "$lock_pid"' EXIT
fi

act "$token" --container-options "--runtime=nvidia --gpus device=$cuda_dev" --artifact-server-path "${HOME}/artifacts" \
-P self-hosted=ghcr.io/catthehacker/ubuntu:act-latest
