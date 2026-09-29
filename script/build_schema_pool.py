"""Preprocess BIRD subsets and generate resumable MCTS candidate pools."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def require_key() -> None:
    from alphasql.llm_call.runtime import configure_environment
    configure_environment()
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("DASHSCOPE_API_KEY is missing; configure it in .env.local before generating candidates")


def preprocess(split: str) -> None:
    subset = ROOT / "data" / "bird" / "schema_pool" / f"{split}.json"
    database_root = ROOT / "data" / "bird" / split / f"{split}_databases"
    output = ROOT / "data" / "preprocessed" / "schema_pool"
    command = [
        sys.executable, "-m", "alphasql.runner.preprocessor",
        "--data_file_path", str(subset),
        "--database_root_dir", str(database_root),
        "--save_root_dir", str(output),
        "--lsh_threshold", "0.5",
        "--lsh_signature_size", "128",
        "--lsh_n_gram", "3",
        "--lsh_top_k", "20",
        "--edit_similarity_threshold", "0.3",
        "--embedding_similarity_threshold", "0.6",
        "--n_parallel_processes", "2",
        "--max_dataset_samples", "-1",
        "--data_split", split,
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def generate(split: str, profile: str = "full") -> None:
    config_stem = f"schema_pool_{split}" if profile == "full" else f"schema_pool_{profile}_{split}"
    config = ROOT / "config" / f"{config_stem}.yaml"
    output_name = "schema_pool" if profile == "full" else f"schema_pool_{profile}"
    output = ROOT / "results" / output_name / split
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "generation.log"
    print(f"Generating {split} candidates; detailed log: {log_path}", flush=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        subprocess.run(
            [sys.executable, "-m", "alphasql.runner.mcts_runner", str(config)],
            cwd=ROOT,
            check=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preprocess", "generate", "all"))
    parser.add_argument("--split", choices=("train", "dev", "both"), default="both")
    parser.add_argument("--profile", choices=("full", "pilot", "dense"), default="full")
    args = parser.parse_args()
    require_key()
    splits = ("train", "dev") if args.split == "both" else (args.split,)
    for selected_split in splits:
        if args.action in ("preprocess", "all"):
            preprocess(selected_split)
        if args.action in ("generate", "all"):
            generate(selected_split, args.profile)
