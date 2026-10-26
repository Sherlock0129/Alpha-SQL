"""Build rich offline data for the adaptive SQL selector.

The ``label`` and ``label-actions`` commands alone read gold SQL. ``prepare-actions``
exports only inference-visible state for independent, equal-budget action runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from alphasql.algorithm.selection.training_data import (
    generator_identity, label_action_arms, label_trace, make_action_manifest, split_by_database,
    stratum_counts,
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                              for row in rows), encoding="utf-8")


def _dataset(path: str | Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    by_id = {str(row.get("question_id", i)): row for i, row in enumerate(rows)}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate question IDs in gold dataset")
    return by_id


def _gold_sql(row: Mapping[str, Any]) -> str:
    sql = row.get("SQL") or row.get("sql")
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("Gold dataset row has no SQL")
    return sql


def _raw_results(directory: str | Path) -> dict[str, dict[str, Any]]:
    results = {}
    for path in sorted(Path(directory).glob("*.json")):
        if path.name == "failures.json":
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        key = str(result["question_id"])
        if key in results:
            raise ValueError(f"Duplicate run result for {key}")
        results[key] = result
    return results


def _calibration_row(row: Mapping[str, Any]) -> dict[str, Any]:
    last = row["rounds"][-1]
    return {"question_id": str(row["question_id"]), "db_id": row["db_id"],
            "coverage_raw": last["jev"]["pool_covered"],
            "covered_label": last["covered_label"],
            "generation_difficulty": last["difficulty"]["generation_difficulty"],
            "selection_difficulty": last["difficulty"]["selection_difficulty"],
            "candidates": [{"valid": bool(item["execution_rechecked_valid"]),
                            "raw": item["candidate_raw"],
                            "label": item["execution_label"]}
                           for item in last["candidates"]]}


def _label(args: argparse.Namespace) -> None:
    dataset = _dataset(args.data_path)
    rows = []
    for key, result in _raw_results(args.results_dir).items():
        gold = dataset[key]
        if gold["db_id"] != result["db_id"]:
            raise ValueError(f"Database mismatch for {key}")
        db_id = result["db_id"]
        db_path = Path(args.database_root_dir) / db_id / f"{db_id}.sqlite"
        rows.append(label_trace(result, _gold_sql(gold), str(db_path),
                                args.execution_timeout, args.near_miss_max_edits))
    _write_jsonl(args.output_path, rows)
    if args.calibration_output:
        _write_jsonl(args.calibration_output, [_calibration_row(row) for row in rows])
    print({"questions": len(rows), "strata": stratum_counts(rows),
           "output": args.output_path})


def _split(args: argparse.Namespace) -> None:
    rows = _read_jsonl(args.labels)
    if len({(generator_identity(row["generator"]),
             json.dumps(row.get("judge", {}), sort_keys=True),
             json.dumps(row.get("calibrator", {}), sort_keys=True))
            for row in rows}) != 1:
        raise ValueError("Split data must use one frozen generator, judge and calibrator")
    parts = split_by_database(rows, args.validation_fraction, args.test_fraction, args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        _write_jsonl(output / f"{name}.jsonl", part)
        _write_jsonl(output / f"{name}_calibration.jsonl",
                     [_calibration_row(row) for row in part])
    manifest = {"source_sha256": hashlib.sha256(Path(args.labels).read_bytes()).hexdigest(),
                "seed": args.seed, "validation_fraction": args.validation_fraction,
                "test_fraction": args.test_fraction,
                "generator_versions": sorted({row["generator"]["version"] for row in rows}),
                "partitions": {name: {"questions": len(part),
                                      "databases": sorted({row["db_id"] for row in part}),
                                      "strata": stratum_counts(part)}
                               for name, part in parts.items()}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False,
                                                   indent=2) + "\n", encoding="utf-8")
    print(manifest)


def _prepare_actions(args: argparse.Namespace) -> None:
    rows = _read_jsonl(args.labels)
    versions = {(generator_identity(row["generator"]),
                 json.dumps(row.get("judge", {}), sort_keys=True),
                 json.dumps(row.get("calibrator", {}), sort_keys=True))
                for row in rows}
    if len(versions) != 1:
        raise ValueError("Action probes require one frozen generator version")
    raw = _raw_results(args.results_dir)
    budget = {"max_new_candidates": args.max_new_candidates,
              "max_model_calls": args.max_model_calls}
    if args.max_tokens is not None:
        budget["max_tokens"] = args.max_tokens
    manifest = make_action_manifest(rows, raw, args.per_stratum, args.seed, budget)
    _write_jsonl(args.output_path, manifest)
    print({"states": len(manifest), "sampling": "up to per_stratum from each difficulty group",
           "output": args.output_path})


def _label_actions(args: argparse.Namespace) -> None:
    rows = _read_jsonl(args.labels)
    gold = {key: _gold_sql(value) for key, value in _dataset(args.data_path).items()}
    output = label_action_arms(_read_jsonl(args.arms), rows, gold,
                               args.database_root_dir, args.execution_timeout)
    _write_jsonl(args.output_path, output)
    print({"states": len(output), "complete_four_arm_states": sum(
        item["complete"] for item in output), "output": args.output_path})


def _probe_actions(args: argparse.Namespace) -> None:
    from alphasql.runner.selector_action_probe import probe_action

    states = _read_jsonl(args.manifest)
    if args.limit is not None:
        states = states[:args.limit]
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    completed = []
    failures = []
    for state in states:
        for action in state["actions"]:
            path = root / str(state["state_id"]) / f"{action}.json"
            if path.is_file() and not args.overwrite:
                completed.append(json.loads(path.read_text(encoding="utf-8")))
                continue
            try:
                arm = probe_action(state, action, args.database_root_dir, root, args.model)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(arm, ensure_ascii=False,
                                                allow_nan=False, indent=2) + "\n",
                                     encoding="utf-8")
                temporary.replace(path)
                completed.append(arm)
            except Exception as exc:
                failures.append({"state_id": state["state_id"], "action": action,
                                 "error_type": type(exc).__name__, "error": str(exc)})
    _write_jsonl(root / "arms.jsonl", completed)
    (root / "failures.json").write_text(json.dumps(failures, ensure_ascii=False,
                                                    indent=2) + "\n", encoding="utf-8")
    print({"arms": len(completed), "failures": len(failures), "output": str(root)})


def _audit(args: argparse.Namespace) -> None:
    rows = _read_jsonl(args.labels)
    gold = _dataset(args.data_path)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for state in row["rounds"]:
            for index, candidate in enumerate(state["candidates"]):
                if not candidate["execution_rechecked_valid"]:
                    continue
                labels = candidate["weak_diagnosis"]["labels"] or {}
                names = (["execution_positive"] if candidate["execution_label"] else
                         [name for name, present in labels.items() if present]
                         or ["unclassified"])
                for name in names:
                    groups.setdefault(name, []).append({
                        "question_id": row["question_id"], "db_id": row["db_id"],
                        "round": state["round"], "candidate_index": index,
                        "question": row["question"], "evidence": row["evidence"],
                        "candidate_sql": candidate["sql"],
                        "gold_sql": _gold_sql(gold[str(row["question_id"])]),
                        "weak_diagnosis": candidate["weak_diagnosis"],
                        "near_miss": candidate["near_miss"],
                        "manual_labels": None, "audit_note": ""})
    rng = random.Random(args.seed)
    sampled = []
    seen = set()
    for name, group in sorted(groups.items()):
        rng.shuffle(group)
        count = 0
        for item in group:
            key = (str(item["question_id"]), item["round"], item["candidate_index"])
            if key in seen:
                continue
            sampled.append(dict(item, audit_bucket=name))
            seen.add(key)
            count += 1
            if count >= args.per_group:
                break
    _write_jsonl(args.output_path, sampled)
    print({"audit_items": len(sampled), "output": args.output_path,
           "contains_gold_sql": True})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    label = sub.add_parser("label")
    label.add_argument("--results-dir", required=True)
    label.add_argument("--data-path", required=True)
    label.add_argument("--database-root-dir", required=True)
    label.add_argument("--output-path", required=True)
    label.add_argument("--calibration-output")
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
    prepare = sub.add_parser("prepare-actions")
    prepare.add_argument("--labels", required=True)
    prepare.add_argument("--results-dir", required=True)
    prepare.add_argument("--output-path", required=True)
    prepare.add_argument("--per-stratum", type=int, default=50)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--max-new-candidates", type=int, default=4)
    prepare.add_argument("--max-model-calls", type=int, default=3)
    prepare.add_argument("--max-tokens", type=int)
    prepare.set_defaults(func=_prepare_actions)
    action_label = sub.add_parser("label-actions")
    action_label.add_argument("--arms", required=True)
    action_label.add_argument("--labels", required=True)
    action_label.add_argument("--data-path", required=True)
    action_label.add_argument("--database-root-dir", required=True)
    action_label.add_argument("--output-path", required=True)
    action_label.add_argument("--execution-timeout", type=int, default=60)
    action_label.set_defaults(func=_label_actions)
    probe = sub.add_parser("probe-actions",
                           help="Run external LLM/JEV calls for four equal-budget arms")
    probe.add_argument("--manifest", required=True)
    probe.add_argument("--database-root-dir", required=True)
    probe.add_argument("--output-dir", required=True)
    probe.add_argument("--model", help="Must match the frozen generator model")
    probe.add_argument("--limit", type=int)
    probe.add_argument("--overwrite", action="store_true")
    probe.set_defaults(func=_probe_actions)
    audit = sub.add_parser("audit")
    audit.add_argument("--labels", required=True)
    audit.add_argument("--data-path", required=True)
    audit.add_argument("--output-path", required=True)
    audit.add_argument("--per-group", type=int, default=20)
    audit.add_argument("--seed", type=int, default=42)
    audit.set_defaults(func=_audit)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
