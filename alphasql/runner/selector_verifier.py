"""Train a SQL-content verifier and fit a validation-only selective stop policy.

The train command never opens the test split. The policy command uses the
database-disjoint validation split and writes a risk-coverage SVG and JSON.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from alphasql.algorithm.selection.content_verifier import (
    ContentVerifier, candidate_records, content_features,
)
from alphasql.algorithm.selection.selective_stop import StopPolicy, fit_stop_policy


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _check_disjoint(train: Sequence[Mapping[str, Any]],
                    validation: Sequence[Mapping[str, Any]]) -> None:
    a, b = {row["db_id"] for row in train}, {row["db_id"] for row in validation}
    if a & b:
        raise ValueError(f"Train and validation databases overlap: {sorted(a & b)}")


def _score_rows(rows: Sequence[Mapping[str, Any]], model: ContentVerifier) -> list[dict[str, Any]]:
    scored = copy.deepcopy(rows)
    records = []
    positions = []
    for row in scored:
        for state in row["rounds"]:
            for item in state["candidates"]:
                item["verifier_probability"] = 0.0
                if not item["execution_rechecked_valid"]:
                    continue
                records.append({"features": content_features(
                    row["question"], row["evidence"], row["full_schema"],
                    item["sql"], item["execution"])})
                positions.append(item)
    for item, score in zip(positions, model.predict(records)):
        item["verifier_probability"] = float(score)
    return scored


def _top1(rows: Sequence[Mapping[str, Any]], score_key: str, *, first: bool) -> dict[str, Any]:
    correct = 0
    covered = 0
    for row in rows:
        state = row["rounds"][0 if first else -1]
        covered += int(state["covered_label"])
        valid = [item for item in state["candidates"] if item["execution_rechecked_valid"]]
        if valid:
            best = max(valid, key=lambda item: float(item[score_key]))
            correct += int(best["execution_label"])
    return {"questions": len(rows), "covered": covered, "selected_correct": correct,
            "selected_accuracy": correct / len(rows) if rows else None}


def _svg(curves: Mapping[str, Sequence[Mapping[str, Any]]], policies: Mapping[str, Any]) -> str:
    width, height, left, top, plot_w, plot_h = 760, 510, 80, 40, 620, 380
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<path d="M {left} {top} V {top+plot_h} H {left+plot_w}" stroke="#333" fill="none"/>']
    for i in range(6):
        x = left + i * plot_w / 5
        y = top + plot_h - i * plot_h / 5
        parts += [f'<path d="M {x:.1f} {top+plot_h} v 5 M {left-5} {y:.1f} h 5" stroke="#333"/>',
                  f'<text x="{x:.1f}" y="{top+plot_h+23}" text-anchor="middle" font-size="12">{i/5:.1f}</text>',
                  f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-size="12">{i/5:.1f}</text>']
    parts += [f'<text x="{left+plot_w/2}" y="{height-35}" text-anchor="middle" font-size="15">Acceptance fraction</text>',
              f'<text transform="translate(24 {top+plot_h/2}) rotate(-90)" text-anchor="middle" font-size="15">Wrong fraction among accepted</text>']
    colors = {"jev_raw": "#3569b0", "content_verifier": "#d36a30"}
    for index, (name, curve) in enumerate(curves.items()):
        points = sorted((item["acceptance_fraction"], item["error_fraction"])
                        for item in curve if item["accepted"])
        color = colors.get(name, "#444")
        path = " ".join(f'{left+x*plot_w:.1f},{top+plot_h-y*plot_h:.1f}' for x, y in points)
        parts.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        chosen = policies[name]
        selected = next((item for item in curve if item["threshold"] == chosen.threshold), None)
        if selected and selected["accepted"]:
            x, y = selected["acceptance_fraction"], selected["error_fraction"]
            parts.append(f'<circle cx="{left+x*plot_w:.1f}" cy="{top+plot_h-y*plot_h:.1f}" r="5" fill="{color}"/>')
        parts.append(f'<text x="{left+12}" y="{top+20+index*22}" fill="{color}" font-size="13">{html.escape(name)}</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def train(args: argparse.Namespace) -> None:
    training, validation = read_rows(args.train), read_rows(args.validation)
    _check_disjoint(training, validation)
    model, report = ContentVerifier.fit(candidate_records(training),
                                        pair_weight=args.pair_weight,
                                        regularization=args.regularization)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "content_verifier.json")
    scored = _score_rows(validation, model)
    report["training_databases"] = sorted({row["db_id"] for row in training})
    report["validation_databases"] = sorted({row["db_id"] for row in validation})
    report["validation_first_round"] = {
        "jev_raw": _top1(validation, "candidate_raw", first=True),
        "content_verifier": _top1(scored, "verifier_probability", first=True)}
    report["validation_final_pool"] = {
        "jev_raw": _top1(validation, "candidate_raw", first=False),
        "content_verifier": _top1(scored, "verifier_probability", first=False)}
    report["label_caveat"] = "Single-database execution equivalence is a proxy label"
    (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def fit_policy(args: argparse.Namespace) -> None:
    validation = read_rows(args.validation)
    if not validation:
        raise ValueError("Validation rows are empty")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sources = {"jev_raw": (validation, "candidate_raw", None)}
    if args.verifier_model:
        model = ContentVerifier.load(args.verifier_model)
        digest = hashlib.sha256(Path(args.verifier_model).read_bytes()).hexdigest()
        sources["content_verifier"] = (_score_rows(validation, model),
                                       "verifier_probability", digest)
    curves, policies = {}, {}
    for name, (rows, score_key, digest) in sources.items():
        policy, curve = fit_stop_policy(rows, score_key=score_key,
                                        max_empirical_risk=args.max_error,
                                        confidence=args.confidence,
                                        min_accepted=args.min_accepted,
                                        criterion=args.criterion)
        if digest:
            policy = policy.with_model_digest(digest)
        policy.save(output / f"stop_policy_{name}.json")
        curves[name], policies[name] = curve, policy
    (output / "risk_coverage.json").write_text(json.dumps(curves, indent=2) + "\n", encoding="utf-8")
    (output / "risk_coverage.svg").write_text(_svg(curves, policies), encoding="utf-8")
    print(json.dumps({name: {"threshold": policy.threshold,
                             "accepted": policy.accepted_questions,
                             "validation_questions": policy.validation_questions}
                      for name, policy in policies.items()}, indent=2))


def evaluate_policy(args: argparse.Namespace) -> None:
    """Read a frozen policy once; never choose a threshold from this split."""
    policy = StopPolicy.load(args.policy)
    rows = read_rows(args.test)
    overlap = {row["db_id"] for row in rows} & set(policy.validation_databases)
    if overlap:
        raise ValueError(f"Test and validation databases overlap: {sorted(overlap)}")
    if policy.score_key == "verifier_probability":
        if not args.verifier_model:
            raise ValueError("Verifier policy requires --verifier-model")
        digest = hashlib.sha256(Path(args.verifier_model).read_bytes()).hexdigest()
        if digest != policy.model_sha256:
            raise ValueError("Verifier model differs from frozen policy")
        rows = _score_rows(rows, ContentVerifier.load(args.verifier_model))
    accepted, wrong, review, continue_count = 0, 0, 0, 0
    for row in rows:
        state = row["rounds"][0]
        valid = [item for item in state["candidates"] if item["execution_rechecked_valid"]]
        if not valid:
            continue_count += 1
            continue
        best = max(valid, key=lambda item: float(item[policy.score_key]))
        pool = float(state["jev"]["pool_covered"])
        if policy.accepts(float(best[policy.score_key]), pool):
            accepted += 1
            wrong += 1 - int(best["execution_label"])
        elif policy.pool_review_below is not None and pool < policy.pool_review_below:
            review += 1
        else:
            continue_count += 1
    report = {"questions": len(rows), "accepted": accepted, "wrong": wrong,
              "acceptance_fraction": accepted / len(rows),
              "error_fraction": wrong / accepted if accepted else None,
              "adjudicate": review, "continue": continue_count,
              "first_round_top1": _top1(rows, policy.score_key, first=True),
              "validation_threshold_unchanged": policy.threshold,
              "label_caveat": "Single-database execution equivalence is a proxy label"}
    path = Path(args.output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--train", required=True)
    training.add_argument("--validation", required=True)
    training.add_argument("--output-dir", required=True)
    training.add_argument("--pair-weight", type=float, default=1.0)
    training.add_argument("--regularization", type=float, default=.1)
    training.set_defaults(func=train)
    policy = sub.add_parser("fit-policy")
    policy.add_argument("--validation", required=True)
    policy.add_argument("--verifier-model")
    policy.add_argument("--output-dir", required=True)
    policy.add_argument("--max-error", type=float, default=.1)
    policy.add_argument("--confidence", type=float, default=.95)
    policy.add_argument("--min-accepted", type=int, default=1)
    policy.add_argument("--criterion", choices=("empirical", "wilson"), default="empirical")
    policy.set_defaults(func=fit_policy)
    evaluation = sub.add_parser("evaluate-policy")
    evaluation.add_argument("--test", required=True)
    evaluation.add_argument("--policy", required=True)
    evaluation.add_argument("--verifier-model")
    evaluation.add_argument("--output-path", required=True)
    evaluation.set_defaults(func=evaluate_policy)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
