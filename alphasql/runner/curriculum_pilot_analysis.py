"""Derive train/validation-only diagnostics from a completed offline P0a run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def hash_json(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def analyze(root):
    summary = json.loads((root / "paired_summary.json").read_text(encoding="utf-8"))
    frozen = json.loads((root / "frozen_records.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "frozen_manifest.json").read_text(encoding="utf-8"))
    by_question = {}
    for row in frozen["validation"]:
        by_question.setdefault(row["question_id"], {})[row["candidate_index"]] = row
    exposures = {a: read_lines(root / a / "exposure.jsonl") for a in summary}
    probes = {a: read_lines(root / a / "probe_trace.jsonl") for a in summary}
    lp, ea = exposures["LP1"], exposures["EA1"]
    probability_diffs = [[abs(x - y) for x, y in zip(l["probabilities"], e["probabilities"])]
                         for l, e in zip(probes["LP1"], probes["EA1"])]
    outcome_rows = []
    for arm in summary:
        last = read_lines(root / arm / "checkpoint_predictions.jsonl")[-1]
        for row in last["questions"]:
            candidates = by_question[row["question_id"]]
            selected = candidates.get(row["selected_candidate_index"])
            correct = [c for c in candidates.values() if c["label"]]
            outcome = ("coverage_miss" if not correct else "both_correct" if row["sc_correct"] and row["correct"]
                       else "regression" if row["sc_correct"] else "fixed" if row["correct"] else "unfixed_hard")
            outcome_rows.append({"arm": arm, "question_id": row["question_id"],
                "db_id": next(iter(candidates.values()))["db_id"], "outcome": outcome,
                "selected_index": row["selected_candidate_index"], "sc_index": row["sc_candidate_index"],
                "selected_correct": row["correct"], "sc_correct": row["sc_correct"],
                "positive_indices": json.dumps([c["candidate_index"] for c in correct]),
                "question": next(iter(candidates.values()))["question"],
                "selected_sql": "" if selected is None else selected["sql"],
                "positive_reference_sql": correct[0]["sql"] if correct else ""})
    with (root / "error_analysis.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(outcome_rows[0]))
        writer.writeheader()
        writer.writerows(outcome_rows)
    with (root / "learning_curve.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["arm", "step", "train_correct", "validation_correct", "fixes", "regressions", "train_pair_loss"])
        writer.writeheader()
        for arm in probes:
            for row in probes[arm]:
                writer.writerow({"arm": arm, "step": row["step"], "train_correct": row["train"]["correct"],
                                 "validation_correct": row["validation"]["correct"], "fixes": row["validation"]["fixes"],
                                 "regressions": row["validation"]["regressions"], "train_pair_loss": row["train_pair_loss_mean"]})
    diagnostics = {
        "frozen_data_hashes_verified": all(hash_json(frozen[k]) == manifest[k + "_sha256"] for k in ("train", "validation")),
        "code_hashes_verified": all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha for p, sha in manifest["code_sha256"].items()),
        "ea_vs_lp": {"different_base_slots": sum(sum(x != y for x, y in zip(l["base_pairs"], e["base_pairs"])) for l, e in zip(lp, ea)),
                      "total_base_slots": sum(len(r["base_pairs"]) for r in lp),
                      "different_base_batches": sum(l["base_pairs"] != e["base_pairs"] for l, e in zip(lp, ea)),
                      "max_absolute_probability_difference": max(map(max, probability_diffs)),
                      "max_total_variation": max(sum(d) / 2 for d in probability_diffs)},
        "arms": {arm: {
            "first_probe_train_reaches_oracle": next((r["step"] for r in rows if r["train"]["correct"] == r["train"]["covered"]), None),
            "best_validation_correct_descriptive_only": max(r["validation"]["correct"] for r in rows),
            "final_outcome_ids": {outcome: [r["question_id"] for r in outcome_rows if r["arm"] == arm and r["outcome"] == outcome]
                                  for outcome in ("regression", "unfixed_hard", "coverage_miss", "fixed")},
            "average_unique_candidates_per_update": sum(r["unique_candidates_forwarded"] for r in exposures[arm]) / len(exposures[arm])
        } for arm, rows in probes.items()},
        "warning": "Descriptive single-seed diagnostics only; no threshold selected, no test read, no causal feature attribution."
    }
    (root / "diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    return diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    analyze(Path(args.run_dir))


if __name__ == "__main__":
    main()
