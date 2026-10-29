"""Seven-view-first SQL ranking on frozen pools.

This is an exploratory follow-up to selection_v2.  It never requests JEV or
reads reference SQL: labels are loaded only for fitting/evaluation.  The
previously inspected 300-question test split must not be described as a new
independent holdout for this follow-up.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from alphasql.runner.multi_angle_selector import VIEWS, _rule
from alphasql.runner.selection_v2 import _joined, _read, _save, _thresholds


METHODS = ("raw_jev", "old_min", "mean", "trimmed_mean", "median", "softmin_02",
           "group_simplex", "seven_simplex")
VIEW_METHODS = ("mean", "trimmed_mean", "median", "softmin_02",
                "group_simplex", "seven_simplex")
BLEND_METHODS = ("mean", "trimmed_mean", "median", "softmin_02")


def _vector(item: Mapping[str, Any], family: str) -> np.ndarray:
    raw = float(item["raw"])
    views = [float(item["views"][name]) for name in VIEWS]
    if family == "group_simplex":
        return np.asarray([raw, np.mean(views[:3]), np.mean(views[3:6]), views[6]])
    if family == "seven_simplex":
        return np.asarray([raw, *views])
    raise ValueError(family)


def _fit_simplex(pools: Sequence[Mapping[str, Any]], family: str,
                 regularization: float) -> dict[str, Any]:
    """Nonnegative, sum-to-one weights with per-question pair balancing."""
    differences = []
    pair_weights = []
    mixed = 0
    for pool in pools:
        positives = [item for item in pool["items"] if item["label"] == 1]
        negatives = [item for item in pool["items"] if item["label"] == 0]
        if not positives or not negatives:
            continue
        mixed += 1
        weight = 1.0 / (len(positives) * len(negatives))
        for positive in positives:
            for negative in negatives:
                differences.append(_vector(positive, family) - _vector(negative, family))
                pair_weights.append(weight)
    if not differences:
        raise ValueError("No within-question positive/negative candidate pairs")
    diff = np.stack(differences)
    weights = np.asarray(pair_weights) / mixed
    prior = np.zeros(diff.shape[1]); prior[0] = 1.0

    def objective(coef: np.ndarray) -> tuple[float, np.ndarray]:
        margin = diff @ coef
        loss = np.logaddexp(0.0, -margin)
        sigmoid_negative = 1.0 / (1.0 + np.exp(margin))
        value = float(weights @ loss + regularization * np.sum((coef - prior) ** 2) / 2)
        gradient = -(diff.T @ (weights * sigmoid_negative)) + regularization * (coef - prior)
        return value, gradient

    result = minimize(objective, prior, jac=True, method="SLSQP",
                      bounds=[(0.0, 1.0)] * len(prior),
                      constraints={"type": "eq", "fun": lambda w: float(w.sum() - 1.0),
                                   "jac": lambda w: np.ones_like(w)},
                      options={"maxiter": 300, "ftol": 1e-11})
    if not result.success:
        raise ValueError(f"Simplex fit failed: {result.message}")
    coef = np.maximum(result.x, 0.0)
    coef /= coef.sum()
    return {"family": family, "weights": coef.tolist(),
            "regularization": regularization,
            "mixed_training_pools": mixed, "training_pairs": len(diff)}


def _score(item: Mapping[str, Any], method: str,
           models: Mapping[str, Mapping[str, Any]]) -> float:
    raw = float(item["raw"])
    values = np.asarray([float(item["views"][name]) for name in VIEWS])
    if method == "raw_jev":
        result = raw
    elif method == "old_min":
        result = _rule(np.asarray([raw, *values]))
    elif method in BLEND_METHODS:
        alpha = float(models.get("blend", {}).get(method, 0.5))
        if method == "mean":
            aggregate = float(values.mean())
        elif method == "trimmed_mean":
            aggregate = float(np.sort(values)[1:-1].mean())
        elif method == "median":
            aggregate = float(np.median(values))
        else:
            aggregate = float(-0.2 * (logsumexp(-values / 0.2) - np.log(len(values))))
        result = (1 - alpha) * raw + alpha * aggregate
    elif method in {"group_simplex", "seven_simplex"}:
        result = float(_vector(item, method) @ np.asarray(models[method]["weights"]))
    else:
        raise ValueError(method)
    # All ingredients are JEV scores in [0,1], so this is a bounded ranking
    # score.  It is still not a calibrated probability of SQL correctness.
    return float(np.clip(result, 0.0, 1.0))


def _decisions(pools: Sequence[Mapping[str, Any]], models: Mapping[str, Any],
               thresholds: Mapping[str, float | None] | None = None) -> tuple[dict, list[dict]]:
    records = []
    for pool in pools:
        covered = any(item["label"] == 1 for item in pool["items"])
        record = {"db_id": pool["db_id"], "question_id": pool["question_id"],
                  "covered": covered, "methods": {}}
        for method in METHODS:
            chosen = (max(pool["items"], key=lambda item: (_score(item, method, models),
                                                           -item["index"]))
                      if pool["items"] else None)
            record["methods"][method] = {
                "index": chosen["index"] if chosen else None,
                "score": _score(chosen, method, models) if chosen else None,
                "correct": int(chosen["label"]) if chosen else 0,
            }
        records.append(record)
    report = {"questions": len(records), "covered": sum(r["covered"] for r in records),
              "methods": {}}
    for method in METHODS:
        correct = sum(r["methods"][method]["correct"] for r in records)
        rescue = sum(r["methods"][method]["correct"] and
                     not r["methods"]["raw_jev"]["correct"] for r in records)
        regression = sum(r["methods"]["raw_jev"]["correct"] and
                         not r["methods"][method]["correct"] for r in records)
        entry = {"top1_correct": correct, "covered_top1_correct": correct,
                 "rescues_vs_raw": rescue, "regressions_vs_raw": regression}
        if thresholds is not None:
            cutoff = thresholds[method]
            accepted = [r for r in records if cutoff is not None and
                        r["methods"][method]["score"] is not None and
                        r["methods"][method]["score"] >= cutoff]
            wrong = sum(not r["methods"][method]["correct"] for r in accepted)
            entry["stop"] = {"threshold": cutoff, "accepted": len(accepted),
                             "wrong": wrong,
                             "error_fraction": wrong / len(accepted) if accepted else None}
        report["methods"][method] = entry
    return report, records


def fit(args: argparse.Namespace) -> None:
    train = _joined(_read(args.train), args.view_dir, args.execution_dir)
    select_val = _joined(_read(args.selection_validation), args.view_dir, args.execution_dir)
    stop_val = _joined(_read(args.stop_validation), args.view_dir, args.execution_dir)
    dbs = [{pool["db_id"] for pool in group} for group in (train, select_val, stop_val)]
    if dbs[0] & dbs[1] or dbs[0] & dbs[2] or dbs[1] & dbs[2]:
        raise ValueError("Train, ranking validation and stop validation databases must be disjoint")
    if any(item["label"] is None for group in (train, select_val, stop_val)
           for pool in group for item in pool["items"]):
        raise ValueError("Offline labels are required for fitting")

    models = {}
    tuning = {}
    models["blend"] = {}
    for method in BLEND_METHODS:
        trials = []
        for alpha in (0.1, 0.25, 0.5, 0.75, 0.9):
            trial_models = {**models, "blend": {**models["blend"], method: alpha}}
            top1 = sum(bool(pool["items"]) and
                       max(pool["items"], key=lambda item: (_score(item, method, trial_models),
                                                            -item["index"]))["label"] == 1
                       for pool in select_val)
            trials.append({"alpha": alpha, "top1": top1})
        best = max(trials, key=lambda trial: (trial["top1"], -trial["alpha"]))
        models["blend"][method] = best["alpha"]
        tuning[method] = trials
    for family in ("group_simplex", "seven_simplex"):
        trials = []
        for regularization in (0.0, 0.01, 0.1, 1.0):
            model = _fit_simplex(train, family, regularization)
            trial_models = {**models, family: model}
            # Other simplex method is absent; score only this method here.
            top1 = sum(bool(pool["items"]) and
                       max(pool["items"], key=lambda item: (_score(item, family, trial_models),
                                                            -item["index"]))["label"] == 1
                       for pool in select_val)
            trials.append({"regularization": regularization, "top1": top1,
                           "weights": model["weights"], "model": model})
        best = max(trials, key=lambda t: (t["top1"], t["regularization"]))
        models[family] = best["model"]
        tuning[family] = [{k: v for k, v in trial.items() if k != "model"} for trial in trials]

    select_report, _ = _decisions(select_val, models)
    primary = max(VIEW_METHODS,
                  key=lambda name: (select_report["methods"][name]["top1_correct"],
                                    -VIEW_METHODS.index(name)))
    stop_report, stop_records = _decisions(stop_val, models)
    thresholds, curves = _thresholds(stop_records, METHODS, args.max_error)
    output = Path(args.output_dir)
    record = {"version": 1, "primary_method": primary, "models": models,
              "thresholds": thresholds,
              "training_databases": sorted(dbs[0]),
              "selection_validation_databases": sorted(dbs[1]),
              "stop_validation_databases": sorted(dbs[2]),
              "max_empirical_error": args.max_error,
              "caveat": "The prior v2 study already inspected the new 300-question test labels"}
    _save(output / "model.json", record)
    _save(output / "tuning.json", tuning)
    _save(output / "selection_validation.json", select_report)
    _save(output / "stop_validation.json", stop_report)
    _save(output / "risk_coverage.json", curves)
    print(json.dumps({"train": len(train), "selection_validation": len(select_val),
                      "stop_validation": len(stop_val), "primary": primary,
                      "selection_top1": {name: entry["top1_correct"]
                                         for name, entry in select_report["methods"].items()},
                      "thresholds": thresholds}, ensure_ascii=False))


def evaluate(args: argparse.Namespace) -> None:
    model_path = Path(args.model)
    model = json.loads(model_path.read_text(encoding="utf-8"))
    pools = _joined(_read(args.input), args.view_dir, args.execution_dir)
    databases = {pool["db_id"] for pool in pools}
    previous = set(model["training_databases"] + model["selection_validation_databases"] +
                   model["stop_validation_databases"])
    if databases & previous:
        raise ValueError("Evaluation databases overlap model fitting or threshold databases")
    report, records = _decisions(pools, model["models"], model["thresholds"])
    report["primary_method"] = model["primary_method"]
    report["databases"] = sorted(databases)
    report["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    report["interpretation"] = "Exploratory re-analysis; this test split was already inspected"
    _save(args.output, report)
    Path(args.output).with_suffix(".jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


def predict(args: argparse.Namespace) -> None:
    """Emit final SQL using the chosen seven-view ranker without reading labels."""
    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    primary = model["primary_method"]
    pools = _joined(_read(args.input), args.view_dir, args.execution_dir)
    records = []
    for pool in pools:
        ordered = sorted(pool["items"],
                         key=lambda item: (-_score(item, primary, model["models"]),
                                           item["index"]))
        chosen = ordered[0] if ordered else None
        records.append({"db_id": pool["db_id"], "question_id": pool["question_id"],
                        "pool_sha256": pool["pool_sha256"],
                        "method": primary, "candidate_index": chosen["index"] if chosen else None,
                        "final_sql": chosen["sql"] if chosen else None,
                        "ranking_score": _score(chosen, primary, model["models"])
                                         if chosen else None,
                        "score_is_calibrated_probability": False,
                        "candidate_scores": [{"index": item["index"],
                                              "score": _score(item, primary, model["models"])}
                                             for item in ordered]})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                              for row in records), encoding="utf-8")
    print(json.dumps({"questions": len(records), "method": primary,
                      "with_sql": sum(row["final_sql"] is not None for row in records)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fitting = sub.add_parser("fit")
    fitting.add_argument("--train", required=True)
    fitting.add_argument("--selection-validation", required=True)
    fitting.add_argument("--stop-validation", required=True)
    fitting.add_argument("--view-dir", required=True)
    fitting.add_argument("--execution-dir", required=True)
    fitting.add_argument("--output-dir", required=True)
    fitting.add_argument("--max-error", type=float, default=0.1)
    fitting.set_defaults(func=fit)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--input", required=True)
    evaluation.add_argument("--view-dir", required=True)
    evaluation.add_argument("--execution-dir", required=True)
    evaluation.add_argument("--model", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.set_defaults(func=evaluate)
    prediction = sub.add_parser("predict")
    prediction.add_argument("--input", required=True)
    prediction.add_argument("--view-dir", required=True)
    prediction.add_argument("--execution-dir", required=True)
    prediction.add_argument("--model", required=True)
    prediction.add_argument("--output", required=True)
    prediction.set_defaults(func=predict)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
