"""Offline, recoverable quarantine of an interrupted linking checkpoint tail."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from deepeye.runner import run_dir, write_json


def quarantine_link_tail(out: Path, keep_first: int, expected_records: int, reason: str):
    checkpoint = out/"schema_linking.artifacts/schema_linking.jsonl"
    lines = checkpoint.read_text(encoding="utf-8").splitlines(keepends=True)
    records = [json.loads(line) for line in lines]
    if len(records) != expected_records or not 0 <= keep_first <= len(records):
        raise ValueError("Checkpoint changed or invalid keep count; refusing mutation")
    ids = [int(row["item_id"]) for row in records]
    manifest = json.loads((out/"manifest.json").read_text(encoding="utf-8"))
    expected = set(manifest["question_ids"])
    if len(set(ids)) != len(ids) or not set(ids) <= expected:
        raise ValueError("Duplicate/unexpected checkpoint IDs; refusing mutation")
    for row in records[:keep_first]:
        artifact = row["stage_artifact"]
        if not artifact.get("direct_linked_tables_and_columns") or not artifact.get("reversed_linked_tables_and_columns"):
            raise ValueError("Retained prefix contains incomplete linking; refusing mutation")
    backup = out/"schema_linking.checkpoint.before-billing-recovery.jsonl"
    if backup.exists():
        raise FileExistsError("Recovery backup already exists; do not overwrite original evidence")
    shutil.copy2(checkpoint,backup)
    temporary = checkpoint.with_suffix(".recovery.tmp")
    temporary.write_text("".join(lines[:keep_first]),encoding="utf-8")
    temporary.replace(checkpoint)
    kept = ids[:keep_first]
    result = {"reason":reason,"original_records":len(records),"retained_records":len(kept),
              "quarantined_records":len(records)-len(kept),"retained_ids":kept,
              "retry_ids":sorted(expected-set(kept)),"backup":str(backup),
              "backup_sha256":hashlib.sha256(backup.read_bytes()).hexdigest(),
              "network_calls":0}
    write_json(out/"CHECKPOINT_RECOVERY.json",result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name",required=True)
    parser.add_argument("--keep-first",type=int,required=True)
    parser.add_argument("--expected-records",type=int,required=True)
    parser.add_argument("--reason",required=True)
    args = parser.parse_args()
    result = quarantine_link_tail(run_dir(args.run_name),args.keep_first,args.expected_records,args.reason)
    print(json.dumps({key:value for key,value in result.items() if key != "retained_ids"},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
