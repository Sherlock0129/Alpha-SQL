"""Evaluate frozen JEV calibration on a database-disjoint holdout split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from alphasql.algorithm.selection.dual_calibration import (
    DualCalibrator, LogisticHead, brier, selection_report,
)


def _reliability(rows: list[dict[str, Any]], head: str,
                 model: LogisticHead, bins: int = 10) -> dict[str, Any]:
    totals = [[0.0, 0.0, 0.0, 0] for _ in range(bins)]
    for row in rows:
        if head == "coverage":
            examples = [(row["coverage_raw"], row["covered_label"])]
        else:
            examples = [(item["raw"], item["label"])
                        for item in row["candidates"] if item["valid"]]
        if not examples:
            continue
        weight = 1.0 / len(examples)
        for raw, label in examples:
            probability = model.predict(float(raw))
            index = min(bins - 1, int(probability * bins))
            bucket = totals[index]
            bucket[0] += weight
            bucket[1] += weight * probability
            bucket[2] += weight * int(label)
            bucket[3] += 1
    total_weight = sum(bucket[0] for bucket in totals)
    out = []
    ece = 0.0
    for index, (weight, confidence, accuracy, count) in enumerate(totals):
        mean_confidence = confidence / weight if weight else None
        mean_accuracy = accuracy / weight if weight else None
        if weight:
            ece += weight / total_weight * abs(mean_confidence - mean_accuracy)
        out.append({"lower": index / bins, "upper": (index + 1) / bins,
                    "weighted_questions": weight, "candidate_count": count,
                    "mean_confidence": mean_confidence,
                    "observed_accuracy": mean_accuracy})
    return {"ece": ece, "bins": out,
            "weighting": "Each question contributes total weight one"}


def evaluate(rows: list[dict[str, Any]], model: DualCalibrator) -> dict[str, Any]:
    raw = LogisticHead()
    return {
        "questions": len(rows),
        "databases": len({row["db_id"] for row in rows}),
        "coverage": {
            "raw_brier": brier(rows, "coverage", raw),
            "calibrated_brier": brier(rows, "coverage", model.coverage),
            "raw_reliability": _reliability(rows, "coverage", raw),
            "calibrated_reliability": _reliability(rows, "coverage", model.coverage),
        },
        "candidate": {
            "raw_brier": brier(rows, "candidate", raw),
            "calibrated_brier": brier(rows, "candidate", model.candidate),
            "raw_reliability": _reliability(rows, "candidate", raw),
            "calibrated_reliability": _reliability(rows, "candidate", model.candidate),
        },
        "selection": selection_report(rows, model.candidate),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--calibrator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    report = evaluate(rows, DualCalibrator.load(args.calibrator))
    report["labels_sha256"] = hashlib.sha256(args.labels.read_bytes()).hexdigest()
    report["calibrator_sha256"] = hashlib.sha256(args.calibrator.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n", encoding="utf-8")
    print({"questions": report["questions"], "databases": report["databases"],
           "coverage_brier": (report["coverage"]["raw_brier"],
                              report["coverage"]["calibrated_brier"]),
           "candidate_brier": (report["candidate"]["raw_brier"],
                               report["candidate"]["calibrated_brier"]),
           "top1_ex": report["selection"]["calibrated_top1_ex"],
           "output": str(args.output)})


if __name__ == "__main__":
    main()
