"""Offline execution labels and database splits for frozen initial SQL pools.

These records contain no JEV scores. They measure generation coverage only and
must not be passed to the dual-calibration trainer as if JEV had been called.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from alphasql.algorithm.compiler.validation import validate_sql
from alphasql.algorithm.selection.training_data import (
    _ast_edit_count, _execution_signature, _gold_requires_order,
    generator_identity, pool_id, split_by_database, weak_diagnosis,
)


def _read_pools(directories: Sequence[str | Path]) -> list[dict[str, Any]]:
    if not directories:
        raise ValueError("At least one pool directory is required")
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    seen: set[str] = set()
    for raw_directory in directories:
        directory = Path(raw_directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        identity = generator_identity(manifest["generator"])
        identities.add(identity)
        for path in sorted(directory.glob("*.json")):
            if path.name in {"manifest.json", "failures.json"}:
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            key = str(row["question_id"])
            if key in seen:
                raise ValueError(f"Duplicate question across pool directories: {key}")
            if row.get("pool_schema_version") != 1 or row.get("jev_status") != "deferred":
                raise ValueError(f"Invalid frozen initial pool: {path}")
            if generator_identity(row["generator"]) != identity:
                raise ValueError(f"Generator mismatch between manifest and {path}")
            if not (directory / "candidates" / f"{key}.pkl").is_file():
                raise FileNotFoundError(f"Missing frozen candidates for {key}: {directory}")
            seen.add(key)
            rows.append(row)
    if len(identities) != 1:
        raise ValueError("Initial pool directories use different generator versions")
    return rows


def label_pool(row: Mapping[str, Any], gold_sql: str, db_path: str,
               timeout: int = 60, near_miss_max_edits: int = 4) -> dict[str, Any]:
    gold = validate_sql(db_path, gold_sql, timeout)
    if not gold.valid:
        raise ValueError(f"Gold SQL did not execute for {row['question_id']}")
    ordered = _gold_requires_order(gold_sql)
    reference = _execution_signature(gold.rows, ordered)
    candidates = []
    for index, item in enumerate(row["candidates"]):
        execution = validate_sql(db_path, item["sql"], timeout)
        correct = int(execution.valid and
                      _execution_signature(execution.rows, ordered) == reference)
        edits = _ast_edit_count(item["sql"], gold_sql) if execution.valid and not correct else None
        candidates.append({
            "candidate_index": index, "route": item.get("route"),
            "valid": execution.valid, "execution_label": correct,
            "ast_edit_count": edits,
            "near_miss": bool(execution.valid and not correct and edits is not None
                              and edits <= near_miss_max_edits),
            "weak_diagnosis": (weak_diagnosis(item["sql"], gold_sql)
                               if execution.valid and not correct else
                               {"source": "not_applicable", "labels": None}),
        })
    return {
        "question_id": row["question_id"], "db_id": row["db_id"],
        "generator": row["generator"],
        "initial_pool_id": pool_id(row["db_id"], row["question_id"],
                                   row["candidates"], generator_identity(row["generator"])),
        "covered_label": int(any(item["execution_label"] for item in candidates)),
        "candidate_count": len(candidates), "candidates": candidates,
        "jev_status": "deferred",
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                            for row in rows), encoding="utf-8")


def _label(args: argparse.Namespace) -> None:
    dataset = json.loads(Path(args.data_path).read_text(encoding="utf-8"))
    by_id = {str(row.get("question_id", index)): row
             for index, row in enumerate(dataset)}
    rows = []
    for pool in _read_pools(args.pool_dirs):
        key = str(pool["question_id"])
        gold = by_id[key]
        if gold["db_id"] != pool["db_id"]:
            raise ValueError(f"Database mismatch for {key}")
        db_id = pool["db_id"]
        db_path = Path(args.database_root_dir) / db_id / f"{db_id}.sqlite"
        rows.append(label_pool(pool, gold.get("SQL") or gold.get("sql"),
                               str(db_path), args.execution_timeout,
                               args.near_miss_max_edits))
    rows.sort(key=lambda row: str(row["question_id"]))
    _write_jsonl(Path(args.output_path), rows)
    print({"questions": len(rows), "covered": sum(row["covered_label"] for row in rows),
           "candidates": sum(row["candidate_count"] for row in rows),
           "output": args.output_path})


def _split(args: argparse.Namespace) -> None:
    source = Path(args.labels)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len({generator_identity(row["generator"]) for row in rows}) != 1:
        raise ValueError("Cannot split mixed generator versions")
    parts = split_by_database(rows, args.validation_fraction, args.test_fraction, args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        _write_jsonl(output / f"{name}.jsonl", part)
    manifest = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "seed": args.seed, "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "jev_status": "deferred",
        "partitions": {name: {"questions": len(part),
                              "covered": sum(row["covered_label"] for row in part),
                              "databases": sorted({row["db_id"] for row in part})}
                       for name, part in parts.items()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False,
                                                   indent=2) + "\n", encoding="utf-8")
    print(manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    label = sub.add_parser("label")
    label.add_argument("--pool-dirs", nargs="+", required=True)
    label.add_argument("--data-path", required=True)
    label.add_argument("--database-root-dir", required=True)
    label.add_argument("--output-path", required=True)
    label.add_argument("--execution-timeout", type=int, default=60)
    label.add_argument("--near-miss-max-edits", type=int, default=4)
    label.set_defaults(func=_label)
    split = sub.add_parser("split")
    split.add_argument("--labels", required=True)
    split.add_argument("--output-dir", required=True)
    split.add_argument("--validation-fraction", type=float, default=0.15)
    split.add_argument("--test-fraction", type=float, default=0.15)
    split.add_argument("--seed", type=int, default=42)
    split.set_defaults(func=_split)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
