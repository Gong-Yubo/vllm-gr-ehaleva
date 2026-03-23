#!/bin/bash
# Script to build wheel for vllm-gr
# Usage: ./build_wheel.sh [--python PYTHON]

set -e

PYTHON="python3"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --python)
            PYTHON="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--python PYTHON]"
            exit 1
            ;;
    esac
done

echo "Building wheel with Python: $PYTHON"

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Clean previous builds
rm -rf dist/ build/ ./*.egg-info

# Install build dependencies if needed
"$PYTHON" -m pip install --quiet build wheel

# Build wheel using python -m build
"$PYTHON" -m build --wheel

echo "Wheel built successfully!"
ls -lh dist/*.whl
