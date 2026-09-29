"""Build resumable direct-generation train/dev candidate pools."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def generate(split: str, limit: int | None = None):
    output = ROOT / "results" / "direct_schema_pool" / split
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "alphasql.runner.direct_candidate_runner",
        "--tasks-file-path", str(ROOT / "data" / "preprocessed" / "schema_pool" / split / "tasks.pkl"),
        "--db-root-dir", str(ROOT / "data" / "bird" / split / f"{split}_databases"),
        "--save-root-dir", str(output),
        "--samples", "3",
        "--n-processes", "4",
    ]
    if limit is not None:
        command.extend(("--limit", str(limit)))
    log_path = output / "generation.log"
    print(f"Generating direct {split} candidates; log: {log_path}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "dev", "both"), default="both")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    splits = ("train", "dev") if args.split == "both" else (args.split,)
    for selected in splits:
        generate(selected, args.limit)

