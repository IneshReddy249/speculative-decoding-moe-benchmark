#!/bin/bash
# run_benchmarks.sh — sweep concurrency levels for a running server
# usage: ./run_benchmarks.sh <label> <port>
# example: ./run_benchmarks.sh baseline 30000
set -euo pipefail

LABEL="${1:?Usage: $0 <label> <port>}"
PORT="${2:?Usage: $0 <label> <port>}"

MODEL_PATH="${MODEL_PATH:-/home/shadeform/models/GLM-4.7-Flash}"
DATASET="${DATASET:-sharegpt.json}"
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

for c in 1 4 8 16 32; do
    echo "=== ${LABEL} c=${c} ==="
    python3 -m sglang.bench_serving \
        --backend sglang-oai-chat \
        --base-url "http://127.0.0.1:${PORT}" \
        --model "$MODEL_PATH" \
        --dataset-name sharegpt \
        --dataset-path "$DATASET" \
        --num-prompts 64 \
        --max-concurrency "$c" \
        --sharegpt-output-len 256 \
        --extra-request-body '{"chat_template_kwargs": {"enable_thinking": false}}' \
        --output-file "${RESULTS_DIR}/${LABEL}_c${c}.json" \
        --warmup-requests 4 \
        --seed 42
    echo ""
done

echo "done. results in ${RESULTS_DIR}/"
