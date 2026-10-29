"""Training-free group Copeland and entropic minimax SQL selection.

The three voting groups are forward, reverse, and critical JEV views.  No
labels, reference SQL, or optimizer are needed for prediction.  Evaluation
uses offline labels from frozen pools and must be treated as exploratory on
previously inspected test databases.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.special import logsumexp

from alphasql.runner.multi_angle_selector import VIEWS, _rule
from alphasql.runner.selection_v2 import _joined, _read, _save


METHODS = ("raw_jev", "old_min", "group_borda", "group_copeland",
           "group_entropic_minimax")
TEMPERATURE = 0.2
RAW_ANCHOR = 0.75


def _groups(item: Mapping[str, Any]) -> np.ndarray:
    view = item["views"]
    names = list(VIEWS)
    return np.asarray([
        np.mean([float(view[name]) for name in names[:3]]),
        np.mean([float(view[name]) for name in names[3:6]]),
        float(view["critical"]),
    ])


def _copeland_scores(items: Sequence[Mapping[str, Any]]) -> dict[int, float]:
    """Three grouped votes per pair; equal votes abstain."""
    if not items:
        return {}
    if len(items) == 1:
        return {items[0]["index"]: 1.0}
    points = {item["index"]: 0.0 for item in items}
    utilities = {item["index"]: _groups(item) for item in items}
    for offset, first in enumerate(items):
        for second in items[offset + 1:]:
            a, b = first["index"], second["index"]
            vote_a = int(np.sum(utilities[a] > utilities[b]))
            vote_b = int(np.sum(utilities[b] > utilities[a]))
            if vote_a > vote_b:
                points[a] += 1.0
            elif vote_b > vote_a:
                points[b] += 1.0
            else:
                points[a] += 0.5
                points[b] += 0.5
    return {index: value / (len(items) - 1) for index, value in points.items()}


def _borda_scores(items: Sequence[Mapping[str, Any]]) -> dict[int, float]:
    """Average three grouped ordinal rankings, giving half-credit for ties."""
    if not items:
        return {}
    if len(items) == 1:
        return {items[0]["index"]: 1.0}
    utilities = {item["index"]: _groups(item) for item in items}
    output = {}
    for item in items:
        index = item["index"]
        points = 0.0
        for other in items:
            if other["index"] == index:
                continue
            points += float(np.sum(utilities[index] > utilities[other["index"]]))
            points += 0.5 * float(np.sum(utilities[index] == utilities[other["index"]]))
        output[index] = points / (3 * (len(items) - 1))
    return output


def _entropic_score(item: Mapping[str, Any]) -> float:
    utilities = _groups(item)
    # min_q sum_g q_g u_g + tau KL(q || uniform_3)
    robust = -TEMPERATURE * (logsumexp(-utilities / TEMPERATURE) - math.log(3))
    score = RAW_ANCHOR * float(item["raw"]) + (1 - RAW_ANCHOR) * float(robust)
    return float(np.clip(score, 0.0, 1.0))


def _candidate_scores(items: Sequence[Mapping[str, Any]], method: str) -> dict[int, float]:
    if method == "group_borda":
        return _borda_scores(items)
    if method == "group_copeland":
        return _copeland_scores(items)
    output = {}
    for item in items:
        if method == "raw_jev":
            value = float(item["raw"])
        elif method == "old_min":
            value = _rule(np.asarray([float(item["raw"])] +
                                     [float(item["views"][name]) for name in VIEWS]))
        elif method == "group_entropic_minimax":
            value = _entropic_score(item)
        else:
            raise ValueError(method)
        output[item["index"]] = value
    return output


def select(pool: Mapping[str, Any], method: str) -> tuple[Mapping[str, Any] | None, float | None]:
    items = pool["items"]
    if not items:
        return None, None
    scores = _candidate_scores(items, method)
    chosen = max(items, key=lambda item: (scores[item["index"]],
                                          float(item["raw"]) if method in
                                          {"group_borda", "group_copeland"} else 0.0,
                                          -item["index"]))
    return chosen, scores[chosen["index"]]


def _evaluate(pools: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for pool in pools:
        valid = pool["items"]
        covered = any(item["label"] == 1 for item in valid)
        mixed = covered and any(item["label"] == 0 for item in valid)
        record = {"db_id": pool["db_id"], "question_id": pool["question_id"],
                  "covered": covered, "mixed": mixed, "methods": {}}
        for method in METHODS:
            chosen, score = select(pool, method)
            record["methods"][method] = {"index": chosen["index"] if chosen else None,
                                         "score": score,
                                         "correct": int(chosen["label"]) if chosen else 0}
        rows.append(record)
    report = {"questions": len(rows), "covered": sum(r["covered"] for r in rows),
              "mixed": sum(r["mixed"] for r in rows), "methods": {},
              "seven_view_usage": {
                  "calls": sum(pool["view_usage"].get("calls", 0) for pool in pools),
                  "input_tokens": sum(pool["view_usage"].get("input_tokens", 0)
                                      for pool in pools),
                  "output_tokens": sum(pool["view_usage"].get("output_tokens", 0)
                                       for pool in pools)},
              "aggregation_extra_jev_calls": 0}
    for method in METHODS:
        result = {"all_top1_correct": sum(r["methods"][method]["correct"] for r in rows),
                  "covered_top1_correct": sum(r["covered"] and r["methods"][method]["correct"]
                                              for r in rows),
                  "mixed_top1_correct": sum(r["mixed"] and r["methods"][method]["correct"]
                                            for r in rows),
                  "rescues_vs_raw": sum(r["methods"][method]["correct"] and
                                        not r["methods"]["raw_jev"]["correct"] for r in rows),
                  "regressions_vs_raw": sum(r["methods"]["raw_jev"]["correct"] and
                                            not r["methods"][method]["correct"] for r in rows)}
        report["methods"][method] = result
    report["by_database"] = {
        db: {"questions": sum(r["db_id"] == db for r in rows),
             "covered": sum(r["db_id"] == db and r["covered"] for r in rows),
             "mixed": sum(r["db_id"] == db and r["mixed"] for r in rows),
             "top1_correct": {method: sum(r["db_id"] == db and r["methods"][method]["correct"]
                                          for r in rows) for method in METHODS}}
        for db in sorted({r["db_id"] for r in rows})}
    return report, rows


def evaluate(args: argparse.Namespace) -> None:
    pools = _joined(_read(args.input), args.view_dir, args.execution_dir)
    if any(item["label"] is None for pool in pools for item in pool["items"]):
        raise ValueError("Offline evaluation requires labels")
    report, rows = _evaluate(pools)
    report["split"] = args.split
    report["interpretation"] = "Exploratory on previously inspected test databases" if args.split == "test" else "Offline ablation"
    _save(args.output, report)
    Path(args.output).with_suffix(".jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8")
    print(json.dumps({"split": args.split, "questions": report["questions"],
                      "covered": report["covered"], "mixed": report["mixed"],
                      "methods": report["methods"]}, ensure_ascii=False))


def predict(args: argparse.Namespace) -> None:
    pools = _joined(_read(args.input), args.view_dir, args.execution_dir)
    rows = []
    for pool in pools:
        decisions = {}
        for method in ("group_borda", "group_copeland", "group_entropic_minimax"):
            chosen, score = select(pool, method)
            decisions[method] = {"candidate_index": chosen["index"] if chosen else None,
                                 "sql": chosen["sql"] if chosen else None,
                                 "ranking_score": score,
                                 "score_is_calibrated_probability": False}
        rows.append({"db_id": pool["db_id"], "question_id": pool["question_id"],
                     "pool_sha256": pool["pool_sha256"], "decisions": decisions})
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                            for row in rows), encoding="utf-8")
    print(json.dumps({"questions": len(rows), "with_candidate": sum(
        row["decisions"]["group_borda"]["candidate_index"] is not None for row in rows)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--input", required=True)
    e.add_argument("--view-dir", required=True)
    e.add_argument("--execution-dir", required=True)
    e.add_argument("--split", required=True,
                   choices=("train", "selection_validation", "validation", "test"))
    e.add_argument("--output", required=True)
    e.set_defaults(func=evaluate)
    p = sub.add_parser("predict")
    p.add_argument("--input", required=True)
    p.add_argument("--view-dir", required=True)
    p.add_argument("--execution-dir", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=predict)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
