#!/bin/bash
#SBATCH --job-name=kev-sql-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=kev-sql-smoke-%j.log

set -euo pipefail
: "${KEV_REPO:?Set server-side Kev repository path}"
: "${KEV_PYTHON:?Set server-side Kev Python path}"
: "${TRAIN_JSONL:?Set server-side training JSONL path}"
: "${OUTPUT_DIR:?Set server-side output path}"
test -f "$KEV_REPO/kev/train.py"
test -x "$KEV_PYTHON"
test -f "$TRAIN_JSONL"
mkdir -p "$(dirname "$OUTPUT_DIR")"
cd "$KEV_REPO"
"$KEV_PYTHON" -m kev.train --data "$TRAIN_JSONL" \
  --base Qwen/Qwen3.5-4B-Base --init_from jaredpalmer/kev-4b \
  --epochs 1 --lr 2e-5 --batch 1 --accum 2 \
  --dtype bf16 --checkpointing 1 --device cuda --out "$OUTPUT_DIR"
