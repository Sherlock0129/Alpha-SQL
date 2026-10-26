"""Question-balanced, easy-to-hard calibration of Jev SQL judgements."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def _logit(value: float) -> float:
    value = min(max(float(value), 1e-4), 1 - 1e-4)
    return math.log(value / (1 - value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


@dataclass(frozen=True)
class LogisticHead:
    bias: float = 0.0
    slope: float = 1.0

    def predict(self, raw: float) -> float:
        return _sigmoid(self.bias + self.slope * _logit(raw))

    def to_dict(self) -> dict[str, float]:
        return {"bias": self.bias, "slope": self.slope}


@dataclass(frozen=True)
class DualCalibrator:
    coverage: LogisticHead
    candidate: LogisticHead

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "coverage": self.coverage.to_dict(),
                "candidate": self.candidate.to_dict()}

    @classmethod
    def load(cls, path: str | Path) -> "DualCalibrator":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            raise ValueError("Unknown calibrator schema version")
        return cls(LogisticHead(**data["coverage"]), LogisticHead(**data["candidate"]))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def _examples(row: Mapping[str, Any], head: str) -> list[tuple[float, int]]:
    if head == "coverage":
        return [(float(row["coverage_raw"]), int(row["covered_label"]))]
    return [(float(candidate["raw"]), int(candidate["label"]))
            for candidate in row["candidates"] if candidate["valid"]]


def brier(rows: Sequence[Mapping[str, Any]], head: str, model: LogisticHead) -> float:
    # Each question contributes equally, irrespective of pool size.
    question_losses = []
    for row in rows:
        examples = _examples(row, head)
        if examples:
            question_losses.append(sum((model.predict(raw) - label) ** 2
                                       for raw, label in examples) / len(examples))
    return sum(question_losses) / len(question_losses) if question_losses else float("nan")


def curriculum_fraction(epoch: int, epochs: int, start_fraction: float = 0.3) -> float:
    """Easy question coverage grows to the whole train set by 60% of training."""
    if epochs <= 0 or not 0 < start_fraction <= 1:
        raise ValueError("Invalid curriculum schedule")
    if epochs == 1:
        return 1.0
    return min(1.0, start_fraction + (1 - start_fraction) * epoch / (0.6 * (epochs - 1)))


def curriculum_order(rows: Sequence[Mapping[str, Any]], head: str) -> list[Mapping[str, Any]]:
    """Keep generation coverage and within-pool selection difficulty distinct."""
    if head == "candidate":
        def selection_key(row: Mapping[str, Any]) -> tuple[float, float, str]:
            if not int(row["covered_label"]):
                return (1.0, 1.0, str(row["question_id"]))
            given = row.get("selection_difficulty")
            if given is None:
                positive = [float(item["raw"]) for item in row["candidates"]
                            if item["valid"] and item["label"]]
                negative = [float(item["raw"]) for item in row["candidates"]
                            if item["valid"] and not item["label"]]
                best = max(positive, default=0.0)
                margin = best - max(negative, default=0.0)
                given = max(0.0, 1.0 - best, (1.0 - margin) / 2.0)
            return (0.0, float(given), str(row["question_id"]))
        return sorted(rows, key=selection_key)
    if head != "coverage":
        raise ValueError(f"Unknown curriculum head: {head}")
    # A coverage calibrator needs both positive and negative pools even in its
    # earliest prefix. Order each class by model error, then interleave them.
    buckets = [[], []]
    for row in rows:
        label = int(row["covered_label"])
        buckets[label].append(row)
    queues = []
    for bucket in buckets:
        bucket.sort(key=lambda row: (abs(float(row["coverage_raw"])
                                          - int(row["covered_label"])),
                                     str(row["question_id"])))
        queues.append(deque(bucket))
    ordered = []
    while queues[0] or queues[1]:
        for bucket in queues:
            if bucket:
                ordered.append(bucket.popleft())
    return ordered


def fit_head(
    train: Sequence[Mapping[str, Any]], validation: Sequence[Mapping[str, Any]],
    head: str, epochs: int = 100, learning_rate: float = 0.03, l2: float = 1e-3,
) -> tuple[LogisticHead, dict[str, Any]]:
    if head not in {"coverage", "candidate"} or not train or not validation:
        raise ValueError("Head and nonempty train/validation records are required")
    if epochs < 1 or learning_rate <= 0 or l2 < 0:
        raise ValueError("Invalid optimization settings")
    ordered = curriculum_order(train, head)
    if not any(_examples(row, head) for row in ordered):
        raise ValueError(f"Train has no usable {head} labels")
    bias, slope = 0.0, 1.0
    best = LogisticHead(bias, slope)
    best_brier = brier(validation, head, best)
    if not math.isfinite(best_brier):
        raise ValueError(f"Validation has no usable {head} labels")
    best_epoch = 0
    for epoch in range(epochs):
        count = max(1, math.ceil(len(ordered) * curriculum_fraction(epoch, epochs)))
        for row in ordered[:count]:
            examples = _examples(row, head)
            if not examples:
                continue
            gradient_bias = sum(_sigmoid(bias + slope * _logit(raw)) - label
                                for raw, label in examples) / len(examples)
            gradient_slope = sum((_sigmoid(bias + slope * _logit(raw)) - label)
                                 * _logit(raw) for raw, label in examples) / len(examples)
            bias -= learning_rate * (gradient_bias + l2 * bias)
            slope -= learning_rate * (gradient_slope + l2 * slope)
        candidate = LogisticHead(bias, slope)
        score = brier(validation, head, candidate)
        if score < best_brier:
            best, best_brier, best_epoch = candidate, score, epoch + 1
    return best, {"best_epoch": best_epoch, "validation_brier": best_brier,
                  "raw_validation_brier": brier(validation, head, LogisticHead())}


def fit_dual(
    train: Sequence[Mapping[str, Any]], validation: Sequence[Mapping[str, Any]],
    epochs: int = 100, learning_rate: float = 0.03,
) -> tuple[DualCalibrator, dict[str, Any]]:
    train_ids = {str(row["question_id"]) for row in train}
    validation_ids = {str(row["question_id"]) for row in validation}
    train_dbs = {str(row["db_id"]) for row in train}
    validation_dbs = {str(row["db_id"]) for row in validation}
    if train_ids & validation_ids or train_dbs & validation_dbs:
        raise ValueError("Calibration train and validation must be disjoint by database")
    coverage, coverage_report = fit_head(train, validation, "coverage", epochs, learning_rate)
    candidate, candidate_report = fit_head(train, validation, "candidate", epochs, learning_rate)
    return DualCalibrator(coverage, candidate), {
        "train_questions": len(train), "validation_questions": len(validation),
        "coverage": coverage_report, "candidate": candidate_report,
        "selection": selection_report(validation, candidate),
    }


def selection_report(rows: Sequence[Mapping[str, Any]], model: LogisticHead) -> dict[str, Any]:
    selected_correct = raw_correct = covered = 0
    for row in rows:
        valid = [item for item in row["candidates"] if item["valid"]]
        covered += int(row["covered_label"])
        if not valid:
            continue
        selected_correct += int(max(valid, key=lambda item: model.predict(item["raw"]))["label"])
        raw_correct += int(max(valid, key=lambda item: item["raw"])["label"])
    n = len(rows)
    return {"questions": n, "candidate_coverage": covered / n,
            "calibrated_top1_ex": selected_correct / n,
            "raw_jev_top1_ex": raw_correct / n,
            "conditional_selection_accuracy": selected_correct / covered if covered else None}
