#!/bin/bash
#SBATCH --job-name=qwen25-coder-lora
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=qwen25-coder-lora-%j.log

set -euo pipefail

: "${TRAIN_JSONL:?Set TRAIN_JSONL to the server-side training JSONL path}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the server-side adapter output path}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Coder-7B-Instruct}"

args=(
  --train-jsonl "$TRAIN_JSONL"
  --output-dir "$OUTPUT_DIR"
  --model "$MODEL_PATH"
  --max-steps "${MAX_STEPS:-10}"
)
if [[ -n "${EVAL_JSONL:-}" ]]; then
  args+=(--eval-jsonl "$EVAL_JSONL")
fi

"$PYTHON_BIN" "$SCRIPT_DIR/train.py" "${args[@]}"
