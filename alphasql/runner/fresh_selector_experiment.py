"""Freeze previously unused BIRD databases for a paired selector experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import random
import re
import subprocess

from alphasql.algorithm.compiler.validation import validate_sql
from alphasql.database.database_manager import DatabaseManager
from alphasql.runner.task import Task


DB_ID = re.compile(r'"db_id"\s*:\s*"([^"]+)"')


def previously_used_databases(*roots: Path) -> set[str]:
    command = ["rg", "-o", "--no-filename", DB_ID.pattern,
               *map(str, roots), "-g", "*.json", "-g", "*.jsonl"]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    if result.returncode not in (0, 1):
        raise RuntimeError(f"Prior database audit failed: {result.stderr}")
    return set(DB_ID.findall(result.stdout))


def prepare(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Frozen experiment directory already exists: {output}")
    data_path, db_root = Path(args.data_path), Path(args.database_root_dir)
    dataset = json.loads(data_path.read_text(encoding="utf-8"))
    by_db: dict[str, list[int]] = {}
    for index, row in enumerate(dataset):
        by_db.setdefault(row["db_id"], []).append(index)
    used = previously_used_databases(Path(args.prior_results_dir),
                                     Path(args.prior_experiments_dir))
    eligible = [db for db in sorted(by_db) if db not in used
                and len(by_db[db]) >= args.questions_per_db
                and (db_root / db / f"{db}.sqlite").is_file()]
    if len(eligible) < args.databases:
        raise ValueError(f"Only {len(eligible)} unseen eligible databases")
    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    selected_dbs = eligible[:args.databases]
    tasks, skipped = [], []
    for db_id in selected_dbs:
        question_ids = by_db[db_id].copy()
        rng.shuffle(question_ids)
        db_path = db_root / db_id / f"{db_id}.sqlite"
        schema = DatabaseManager.get_database_schema(db_id, str(db_root))
        count = 0
        for question_id in question_ids:
            row = dataset[question_id]
            gold = row.get("SQL") or row.get("sql")
            execution = validate_sql(str(db_path), gold, args.gold_timeout)
            if not execution.valid:
                skipped.append({"question_id": question_id, "db_id": db_id,
                                "reason": execution.status})
                continue
            tasks.append(Task(question_id=question_id, db_id=db_id,
                              question=row["question"], evidence=row.get("evidence") or "",
                              sql=None, difficulty=row.get("difficulty"),
                              table_schema_dict=schema.tables))
            count += 1
            if count == args.questions_per_db:
                break
        if count < args.questions_per_db:
            raise ValueError(f"Only {count} executable gold SQLs in {db_id}")
    output.mkdir(parents=True)
    task_path = output / "tasks.pkl"
    with task_path.open("wb") as handle:
        pickle.dump(tasks, handle)
    manifest = {
        "protocol": "fresh-database paired selector experiment v1",
        "seed": args.seed, "selected_databases": selected_dbs,
        "question_ids": [task.question_id for task in tasks],
        "prior_used_databases": sorted(used),
        "skipped_invalid_reference": skipped,
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "tasks_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
        "gold_sql_exported_to_tasks": False,
        "generator": {"model": "Qwen3.8-27B", "samples": 2,
                      "temperature": .8, "prompt_paths": ["skeleton", "query_plan"],
                      "prompt_path_samples": 1},
        "budget_per_question": {"max_generation_rounds": 2,
                                "max_llm_requests_per_round": 8,
                                "max_jev_requests": 2,
                                "max_unique_candidates": 16},
        "label_note": "Reference SQL used only for executable-gold check and offline labels",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False,
                                                   indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"databases": selected_dbs, "questions": len(tasks),
                      "invalid_gold_skipped": len(skipped), "output": str(output)},
                     ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/bird/train/train.json")
    parser.add_argument("--database-root-dir", default="data/bird/train/train_databases")
    parser.add_argument("--prior-results-dir", default="results")
    parser.add_argument("--prior-experiments-dir", default="data/experiments")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--databases", type=int, default=6)
    parser.add_argument("--questions-per-db", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--gold-timeout", type=int, default=15)
    args = parser.parse_args()
    if args.databases < 2 or args.questions_per_db < 1:
        raise ValueError("At least two databases and one question per database required")
    prepare(args)


if __name__ == "__main__":
    main()
