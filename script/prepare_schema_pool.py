"""Create deterministic, database-balanced BIRD subsets for reranker candidates."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]


def balanced_sample(rows, size: int, seed: int):
    groups = defaultdict(list)
    for row in rows:
        groups[(str(row["db_id"]), str(row.get("difficulty", "unknown")))].append(row)
    rng = random.Random(seed)
    queues = {}
    for key, values in groups.items():
        rng.shuffle(values)
        queues[key] = deque(values)
    chosen = []
    keys = sorted(queues)
    while len(chosen) < min(size, len(rows)):
        made_progress = False
        for key in keys:
            if queues[key] and len(chosen) < size:
                chosen.append(queues[key].popleft())
                made_progress = True
        if not made_progress:
            break
    return sorted(chosen, key=lambda row: int(row["question_id"]))


def prepare(split: str, size: int, seed: int, database_count: int, output_root: Path):
    source = ROOT / "data" / "bird" / split / f"{split}.json"
    if not source.exists():
        raise FileNotFoundError(f"Install BIRD {split} first: {source}")
    rows = json.loads(source.read_text(encoding="utf-8"))
    # Official train.json omits question_id. Preserve the original array index,
    # matching Alpha-SQL's Preprocessor convention, before reordering the subset.
    rows = [dict(row, question_id=row.get("question_id", index)) for index, row in enumerate(rows)]
    database_root = ROOT / "data" / "bird" / split / f"{split}_databases"
    available_db_ids = sorted(
        {str(row["db_id"]) for row in rows},
        key=lambda db_id: ((database_root / db_id / f"{db_id}.sqlite").stat().st_size, db_id),
    )
    selected_db_ids = set(available_db_ids[:database_count])
    rows = [row for row in rows if str(row["db_id"]) in selected_db_ids]
    selected = balanced_sample(rows, size, seed)
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"{split}.json"
    output.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "split": split,
        "seed": seed,
        "requested_size": size,
        "selected_size": len(selected),
        "database_count": len({row["db_id"] for row in selected}),
        "database_selection": "smallest SQLite files for bounded phase-one preprocessing",
        "database_ids": sorted({row["db_id"] for row in selected}),
        "difficulty_counts": {
            difficulty: sum(row.get("difficulty", "unknown") == difficulty for row in selected)
            for difficulty in sorted({row.get("difficulty", "unknown") for row in selected})
        },
        "question_ids": [row["question_id"] for row in selected],
        "path": str(output.relative_to(ROOT)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--dev-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-db-count", type=int, default=12)
    parser.add_argument("--dev-db-count", type=int, default=6)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "bird" / "schema_pool")
    args = parser.parse_args()
    if not args.output_root.is_absolute():
        args.output_root = ROOT / args.output_root
    manifests = [
        prepare("train", args.train_size, args.seed, args.train_db_count, args.output_root),
        prepare("dev", args.dev_size, args.seed + 1, args.dev_db_count, args.output_root),
    ]
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifests, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifests, ensure_ascii=False, indent=2))
