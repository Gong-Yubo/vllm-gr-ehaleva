#!/bin/bash
set -e

# Run all tests recursively under tests/
python3 -m pytest --run-slow -s tests
