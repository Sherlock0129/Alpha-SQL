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
: "${KEV_INIT_FROM:?Set local Kev checkpoint directory}"
test -f "$KEV_REPO/kev/train.py"
test -x "$KEV_PYTHON"
test -f "$TRAIN_JSONL"
test -f "$KEV_REPO/Qwen/Qwen3.5-4B-Base/config.json"
test -f "$KEV_INIT_FROM/head.pt"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$(dirname "$OUTPUT_DIR")"
cd "$KEV_REPO"
"$KEV_PYTHON" -m kev.train --data "$TRAIN_JSONL" \
  --base Qwen/Qwen3.5-4B-Base --init_from "$KEV_INIT_FROM" \
  --epochs 1 --lr 2e-5 --batch 1 --accum 2 \
  --dtype bf16 --checkpointing 1 --device cuda --out "$OUTPUT_DIR"
