"""Merge independently collected selector runs after checking frozen versions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any

from alphasql.algorithm.selection.training_data import generator_identity


def merge_shards(shard_dirs: list[Path], tasks_path: Path, output: Path,
                 allow_incomplete: bool = False) -> dict[str, Any]:
    with tasks_path.open("rb") as handle:
        tasks = pickle.load(handle)
    expected = {str(task.question_id) for task in tasks}
    if len(expected) != len(tasks):
        raise ValueError("Task file has duplicate question IDs")
    results: dict[str, tuple[dict[str, Any], Path]] = {}
    identities: set[tuple[str, str, str]] = set()
    response_models: set[str] = set()
    failures = []
    usage_lines = []
    for directory in shard_dirs:
        if (directory / "failures.json").is_file():
            failures.extend(json.loads((directory / "failures.json").read_text(encoding="utf-8")))
        usage = directory / "llm_usage.jsonl"
        if usage.is_file():
            usage_lines.extend(line for line in usage.read_text(encoding="utf-8").splitlines()
                               if line.strip())
        for path in sorted(directory.glob("*.json")):
            if path.name == "failures.json":
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            key = str(row["question_id"])
            if key not in expected:
                raise ValueError(f"Unexpected question ID: {key}")
            if key in results:
                raise ValueError(f"Duplicate question ID across shards: {key}")
            if row.get("schema_version") != 2 or not row.get("rounds"):
                raise ValueError(f"Incomplete v2 selector trace: {path}")
            identity = (generator_identity(row["generator"]),
                        json.dumps(row.get("judge", {}), sort_keys=True),
                        json.dumps(row.get("calibrator", {}), sort_keys=True))
            identities.add(identity)
            for state in row["rounds"]:
                judgement = state["jev"]
                if len(judgement.get("candidate", [])) != state["pool_size"]:
                    raise ValueError(f"Incomplete candidate scores: {path}, round {state['round']}")
                if judgement.get("called"):
                    if any(judgement.get(f"gap_{name}") is None for name in
                           ("schema", "projection", "filters", "logic")):
                        raise ValueError(f"Missing diagnosis: {path}, round {state['round']}")
                    if not judgement.get("model"):
                        raise ValueError(f"Missing JEV model: {path}, round {state['round']}")
                    response_models.add(judgement["model"])
                elif any(item["execution"]["valid"] for item in state["candidates"]):
                    raise ValueError(f"JEV skipped a valid pool: {path}, round {state['round']}")
            results[key] = row, path
    if len(identities) != 1:
        raise ValueError("Merged shards use different generator, judge or calibrator versions")
    if len(response_models) != 1:
        raise ValueError(f"Merged shards use mixed or missing JEV response models: {response_models}")
    missing = sorted(expected - set(results),
                     key=lambda value: (0, int(value)) if value.isdigit() else (1, value))
    if missing and not allow_incomplete:
        raise ValueError(f"Missing {len(missing)} questions; first IDs: {missing[:10]}")
    output.mkdir(parents=True, exist_ok=True)
    for key, (_, source) in results.items():
        destination = output / f"{key}.json"
        payload = source.read_bytes()
        if destination.is_file() and destination.read_bytes() != payload:
            raise ValueError(f"Conflicting merged result for question {key}")
        if not destination.is_file():
            temporary = destination.with_suffix(".json.tmp")
            temporary.write_bytes(payload)
            temporary.replace(destination)
    (output / "llm_usage.jsonl").write_text("\n".join(usage_lines) + "\n",
                                            encoding="utf-8")
    (output / "failures.json").write_text(json.dumps(failures, ensure_ascii=False,
                                                     indent=2) + "\n", encoding="utf-8")
    manifest = {
        "tasks_sha256": hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
        "expected_questions": len(expected), "merged_questions": len(results),
        "missing_question_ids": missing, "failure_records": len(failures),
        "generator_identity": next(iter(identities))[0],
        "jev_response_models": sorted(response_models),
        "shard_dirs": [str(path.resolve()) for path in shard_dirs],
        "llm_calls_logged": len(usage_lines),
    }
    metadata = output / "metadata"
    metadata.mkdir(exist_ok=True)
    (metadata / "merge_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False,
                                                          indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--tasks-file-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    manifest = merge_shards(args.shard_dirs, args.tasks_file_path,
                            args.output_dir, args.allow_incomplete)
    print({"expected_questions": manifest["expected_questions"],
           "merged_questions": manifest["merged_questions"],
           "missing_questions": len(manifest["missing_question_ids"]),
           "failure_records": manifest["failure_records"],
           "jev_response_models": manifest["jev_response_models"],
           "llm_calls_logged": manifest["llm_calls_logged"],
           "output": str(args.output_dir)})


if __name__ == "__main__":
    main()
