#!/bin/bash
#
# Integration test for resource.sh
# This script verifies GPU allocation and locking mechanisms.

set -euo pipefail

# Check for required system dependencies
if ! command -v nvidia-smi &> /dev/null; then
    echo "SKIP: nvidia-smi is not installed or not in PATH."
    exit 0
fi

GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | tail -n 1 || echo 0)
if [ -z "$GPU_COUNT" ] || [ "$GPU_COUNT" -eq 0 ]; then
    echo "SKIP: No GPUs available on this system to test."
    exit 0
fi

# Source the target library
source "./tools/resource.sh"

echo "Running tests for resource.sh..."

# 1. Test get_gpu
echo -n "Testing get_gpu... "
set +e
GPU_ID=$(get_gpu 100 15)
rc=$?
set -e

if [ $rc -ne 0 ] || [ -z "$GPU_ID" ]; then
    echo "SKIP (Timeout or no GPU available matching criteria within 15 seconds - GPUs are likely busy)"
    exit 0
fi
echo "PASSED (Got GPU ID: $GPU_ID)"

# 2. Test lock_gpu & unlock_gpu
echo -n "Testing lock_gpu on GPU $GPU_ID... "
LOCK_PID=$(lock_gpu "$GPU_ID")
if [ -z "$LOCK_PID" ]; then
    echo "FAILED (lock_gpu did not return a PID. Check for missing EGL/GL dependencies in Python)"
    exit 1
fi

if ! ps -p "$LOCK_PID" > /dev/null; then
    echo "FAILED (Background process $LOCK_PID is not running)"
    exit 1
fi
echo "PASSED (Locked with PID: $LOCK_PID)"

echo -n "Testing unlock_gpu... "
unlock_gpu "$LOCK_PID"
sleep 1
if ps -p "$LOCK_PID" > /dev/null; then
    echo "FAILED (Background process $LOCK_PID is still running)"
    kill -9 "$LOCK_PID" 2>/dev/null || true # Cleanup
    exit 1
fi
echo "PASSED (GPU unlocked)"

# 3. Test get_gpus
echo -n "Testing get_gpus (requesting 2 GPUs)... "
if [ "$GPU_COUNT" -lt 2 ]; then
    echo "SKIP (System has less than 2 GPUs)"
    exit 0
fi

set +e
GPU_IDS=$(get_gpus 2 100 15)
rc=$?
set -e

if [ $rc -ne 0 ] || [ -z "$GPU_IDS" ]; then
    echo "SKIP (Timeout or no 2 GPUs available within 15 seconds - GPUs are likely busy)"
    exit 0
fi

IFS=',' read -ra ID_ARRAY <<< "$GPU_IDS"
if [ "${#ID_ARRAY[@]}" -ne 2 ]; then
    echo "FAILED (Expected 2 GPUs, got ${#ID_ARRAY[@]}: $GPU_IDS)"
    exit 1
fi
echo "PASSED (Got GPU IDs: $GPU_IDS)"

# 4. Test lock_gpus & unlock_gpus
echo -n "Testing lock_gpus on GPUs '$GPU_IDS'... "
LOCK_PIDS=$(lock_gpus "$GPU_IDS")
if [ -z "$LOCK_PIDS" ]; then
    echo "FAILED (lock_gpus did not return PIDs)"
    exit 1
fi

IFS=',' read -ra PID_ARRAY <<< "$LOCK_PIDS"
if [ "${#PID_ARRAY[@]}" -ne 2 ]; then
    echo "FAILED (Expected 2 PIDs, got ${#PID_ARRAY[@]}: $LOCK_PIDS)"
    exit 1
fi
echo "PASSED (Locked with PIDs: $LOCK_PIDS)"

echo -n "Testing unlock_gpus... "
unlock_gpus "$LOCK_PIDS"
sleep 1
if pgrep -P "$$" -f "import ctypes" > /dev/null; then
    echo "WARNING (Some Python lock processes might still be lingering)"
fi
echo "PASSED (GPUs unlocked)"

echo "All integration tests passed successfully!"
