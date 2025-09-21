#!/usr/bin/env bash
set -euo pipefail

# GPU selection (optional)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PORT=${PORT:-9000}
echo "Starting Sanctra ASR GPU on port ${PORT}"
uvicorn server.main:app --host 0.0.0.0 --port "${PORT}"
