"""Prepare an isolated mixed-model DeepEye continuation from verified checkpoints.

This command performs no network calls and never reads or writes an API key.
It intentionally creates a new frozen run because the model and provider change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

from deepeye import runner


DEFAULT_MODEL = "Qwen3.6-35B-A3B"
DEFAULT_BASE_URL = "https://aigc-api.hkust-gz.edu.cn/api/v2"
SNAPSHOTS = ("dataset.snapshot", "value_retrieval.snapshot")


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshot_ids(path: Path) -> list[int]:
    ids = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.append(int(json.loads(line)["input"]["question_id"]))
    return ids


def _checkpoint_ids(path: Path) -> list[int]:
    ids = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                artifact = row["stage_artifact"]
                if not artifact.get("direct_linked_tables_and_columns") or not artifact.get("reversed_linked_tables_and_columns"):
                    raise ValueError(f"Incomplete schema linking checkpoint item: {row['item_id']}")
                ids.append(int(row["item_id"]))
    return ids


def _validate_source(source: Path) -> tuple[dict, dict, str]:
    manifest = _json(source / "manifest.json")
    recovery = _json(source / "CHECKPOINT_RECOVERY.json")
    expected = [int(i) for i in manifest["question_ids"]]
    retained = [int(i) for i in recovery["retained_ids"]]
    retry = [int(i) for i in recovery["retry_ids"]]
    if len(expected) != 300 or len(set(expected)) != 300:
        raise ValueError("Source is not a frozen, unique 300-question run")
    if len(retained) != len(set(retained)) or set(retained) | set(retry) != set(expected) or set(retained) & set(retry):
        raise ValueError("Recovery IDs do not partition the frozen question set")
    checkpoint = source / "schema_linking.artifacts" / "schema_linking.jsonl"
    if _checkpoint_ids(checkpoint) != retained:
        raise ValueError("Safe schema linking checkpoint differs from the recovery audit")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for name in SNAPSHOTS:
        snapshot = _json(source / name)
        if snapshot.get("format") != "structured_dataset_snapshot":
            raise ValueError(f"Unsupported snapshot format: {name}")
        items_path = source / snapshot["snapshot_root"] / "items.jsonl"
        if _snapshot_ids(items_path) != expected:
            raise ValueError(f"Snapshot IDs differ from the frozen run: {name}")
    db_root = Path(manifest["dataset_root"]) / manifest["dataset_split"]
    labels = json.loads((db_root / f"{manifest['dataset_split']}.json").read_text(encoding="utf-8"))
    db_ids = {record["db_id"] for index, record in enumerate(labels)
              if int(record.get("question_id", index)) in set(expected)}
    for db_id in db_ids:
        if not (source / "vector_database" / db_id / "success_flag").is_file():
            raise ValueError(f"Missing reusable value index for {db_id}")
    return manifest, recovery, digest


def prepare_continuation(
    source_run: str,
    target_run: str,
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    workers: int = 8,
) -> Path:
    if source_run == target_run:
        raise ValueError("Continuation needs a distinct run name")
    if workers < 1:
        raise ValueError("workers must be positive")
    source, target = runner.run_dir(source_run), runner.run_dir(target_run)
    if target.exists():
        raise FileExistsError(f"Target run already exists: {target}")
    source_manifest, recovery, checkpoint_sha256 = _validate_source(source)
    args = SimpleNamespace(
        data_root=source_manifest["dataset_root"],
        split=source_manifest["dataset_split"],
        question_ids=",".join(str(i) for i in source_manifest["question_ids"]),
        ids_file=None,
        max_samples=None,
        profile=source_manifest["profile"],
        run_name=target_run,
        workers=workers,
        few_shot=source_manifest["few_shot"],
    )
    environment = {
        "DEEPEYE_MODEL": model,
        "DEEPEYE_BASE_URL": base_url,
        "DEEPEYE_EMBEDDING_MODEL": source_manifest["embedding_model"],
        "DEEPEYE_EMBEDDING_BASE_URL": source_manifest["embedding_base_url"],
    }
    previous = {key: os.environ.get(key) for key in environment}
    try:
        os.environ.update(environment)
        runner.prepare(args)
        runner.load_manifest(args, verify_inputs=True)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    for name in SNAPSHOTS:
        shutil.copy2(source / name, target / name)
        snapshot = _json(target / name)
        shutil.copytree(source / snapshot["snapshot_root"], target / snapshot["snapshot_root"])
        snapshot["dataset_config"]["save_path"] = str(target / "dataset.snapshot")
        runner.write_json(target / name, snapshot)
    shutil.copytree(source / "vector_database", target / "vector_database")
    shutil.copytree(source / "schema_linking.artifacts", target / "schema_linking.artifacts")
    metadata = target / "schema_linking.artifacts" / "meta.json"
    meta = _json(metadata)
    meta["save_path"] = str(target / "schema_linking.snapshot")
    runner.write_json(metadata, meta)
    copied_sha256 = hashlib.sha256((target / "schema_linking.artifacts" / "schema_linking.jsonl").read_bytes()).hexdigest()
    if copied_sha256 != checkpoint_sha256:
        raise OSError("Copied schema linking checkpoint failed SHA256 verification")
    runner.write_json(target / "RESUME_PROVENANCE.json", {
        "mixed_models": True,
        "source_run": source_run,
        "source_model": source_manifest["model"],
        "new_model": model,
        "new_base_url": base_url,
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_retained_schema_linking_ids": recovery["retained_ids"],
        "new_schema_linking_ids": recovery["retry_ids"],
        "source_stages": ["preprocess", "index-values", "retrieve-values"],
        "new_stages": ["generate", "revise", "select"],
        "api_usage_note": "Historical source usage remains in the source run; this run records only new successful chat responses.",
        "network_calls": 0,
    })
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", default="bird-train-300-qwen35-p32")
    parser.add_argument("--run-name", default="bird-train-300-hkust-qwen36-resume")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    target = prepare_continuation(args.source_run, args.run_name,
                                  model=args.model, base_url=args.base_url, workers=args.workers)
    print(f"Offline continuation prepared: {target}")


if __name__ == "__main__":
    main()
