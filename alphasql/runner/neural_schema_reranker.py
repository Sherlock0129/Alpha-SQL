"""Train and evaluate a GPU-capable pairwise Schema-aware candidate reranker."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from alphasql.algorithm.selection.neural_reranker import NeuralPairwiseReranker
from alphasql.algorithm.selection.schema_signal import selection_metrics, vectorize
from alphasql.runner.schema_signal_experiment import build_records, split_records


def aligned_features(train_records, evaluation_records, groups):
    train_x, train_y, names = vectorize(train_records, groups)
    evaluation_x = np.asarray([
        [float(record["features"].get(name, 0.0)) for name in names]
        for record in evaluation_records
    ], dtype=np.float32)
    return train_x, train_y, evaluation_x, names


def fuse_with_self_consistency(records, neural_scores, alpha):
    """Use the neural model as a residual on top of the existing selector."""
    neural_scores = np.clip(np.asarray(neural_scores, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    neural_logits = np.log(neural_scores / (1.0 - neural_scores))
    baseline_scores = np.asarray([
        float(record["features"].get("exec_result_group_size", 0.0))
        for record in records
    ])
    return baseline_scores + float(alpha) * neural_logits


def tune_fusion_alpha(validation_records, validation_scores, alpha_grid):
    """Select the least aggressive residual weight among equal validation EX."""
    trials = []
    for alpha in alpha_grid:
        metrics = selection_metrics(
            validation_records,
            fuse_with_self_consistency(validation_records, validation_scores, alpha),
        )
        trials.append({"alpha": float(alpha), **metrics})
    best = max(trials, key=lambda row: (row["model_ex"], -row["alpha"]))
    return best["alpha"], trials


def confidence_gated_scores(records, neural_scores, threshold):
    """Keep self-consistency unless the neural alternative wins confidently."""
    grouped = defaultdict(list)
    for index, (record, score) in enumerate(zip(records, neural_scores)):
        grouped[str(record["question_id"])].append((index, record, float(score)))
    selected_scores = np.zeros(len(records), dtype=np.float64)
    for candidates in grouped.values():
        baseline = max(candidates, key=lambda item: (
            item[1]["features"].get("exec_result_group_size", 0),
            -item[1].get("candidate_index", 0),
        ))
        neural = max(candidates, key=lambda item: (
            item[2], -item[1].get("candidate_index", 0)
        ))
        margin = neural[2] - baseline[2]
        selected = neural if neural[0] != baseline[0] and margin >= threshold else baseline
        selected_scores[selected[0]] = 1.0
    return selected_scores


def tune_confidence_threshold(validation_records, validation_scores, threshold_grid):
    trials = []
    for threshold in threshold_grid:
        metrics = selection_metrics(
            validation_records,
            confidence_gated_scores(validation_records, validation_scores, threshold),
        )
        trials.append({"threshold": float(threshold), **metrics})
    # Prefer the more conservative (larger) gate when validation EX ties.
    best = max(trials, key=lambda row: (row["model_ex"], row["threshold"]))
    return best["threshold"], trials


def run_model(name, groups, train_records, validation_records, evaluation_records, args, output_dir):
    train_x, train_y, validation_x, names = aligned_features(train_records, validation_records, groups)
    evaluation_x = np.asarray([
        [float(record["features"].get(feature, 0.0)) for feature in names]
        for record in evaluation_records
    ], dtype=np.float32)
    model = NeuralPairwiseReranker(
        hidden_sizes=tuple(args.hidden_sizes),
        dropout=args.dropout,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        pointwise_weight=args.pointwise_weight,
        seed=args.seed,
        device=args.device,
        patience=args.patience,
    ).fit(
        train_x,
        train_y,
        train_records,
        validation_values=validation_x,
        validation_records=validation_records,
    )
    validation_scores = model.predict_proba(validation_x)
    raw_scores = model.predict_proba(evaluation_x)
    alpha, fusion_trials = tune_fusion_alpha(
        validation_records, validation_scores, args.fusion_alpha_grid
    )
    threshold, gate_trials = tune_confidence_threshold(
        validation_records, validation_scores, args.confidence_threshold_grid
    )
    fusion_validation_ex = max(row["model_ex"] for row in fusion_trials)
    gate_validation_ex = max(row["model_ex"] for row in gate_trials)
    if gate_validation_ex > fusion_validation_ex:
        decision_policy = "confidence_gate"
        scores = confidence_gated_scores(evaluation_records, raw_scores, threshold)
    else:
        decision_policy = "residual_fusion"
        scores = fuse_with_self_consistency(evaluation_records, raw_scores, alpha)
    model.save(output_dir / f"{name}.pt", names)
    return {
        "feature_groups": groups,
        "feature_count": len(names),
        "training_pairs": model.pair_count,
        "best_epoch": model.best_epoch,
        "device": str(model.device),
        "training_history": model.training_history,
        "raw_neural_metrics": selection_metrics(evaluation_records, raw_scores),
        "fusion_alpha": alpha,
        "fusion_validation_trials": fusion_trials,
        "confidence_threshold": threshold,
        "confidence_gate_validation_trials": gate_trials,
        "decision_policy": decision_policy,
        **selection_metrics(evaluation_records, scores),
    }, scores


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    original_train_records = build_records(
        args.train_results_dir,
        args.train_data_path,
        args.db_root_dir,
        include_gold_positive=args.include_gold_positive,
    )
    evaluation_records = build_records(
        args.eval_results_dir, args.eval_data_path, args.eval_db_root_dir
    )
    train_records, validation_records, validation_key = split_records(
        original_train_records, args.validation_fraction, args.seed, args.validation_split_unit
    )
    if args.hard_negatives_per_question:
        training_question_ids = {str(record["question_id"]) for record in train_records}
        augmented_records = build_records(
            args.train_results_dir,
            args.train_data_path,
            args.db_root_dir,
            hard_negatives_per_question=args.hard_negatives_per_question,
        )
        train_records = [
            record for record in augmented_records
            if str(record["question_id"]) in training_question_ids
        ]
    report = {
        "train_candidates": len(train_records),
        "validation_candidates": len(validation_records),
        "evaluation_candidates": len(evaluation_records),
        "validation_split": validation_key,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "synthetic_hard_negatives": sum(
            record.get("candidate_source") == "synthetic_hard_negative" for record in train_records
        ),
        "models": {},
    }
    predictions = []
    for name, groups in {
        "neural_sql_execution": ["sql", "execution"],
        "neural_schema_aware": ["sql", "schema", "execution"],
        "neural_schema_aware_mcts": ["sql", "schema", "execution", "mcts"],
    }.items():
        metrics, scores = run_model(
            name, groups, train_records, validation_records, evaluation_records, args, output_dir
        )
        report["models"][name] = metrics
        if name == "neural_schema_aware":
            predictions = [
                {
                    "question_id": record["question_id"],
                    "candidate_index": record["candidate_index"],
                    "label": record["label"],
                    "score": float(score),
                    "sql": record["sql"],
                }
                for record, score in zip(evaluation_records, scores)
            ]
    schema = report["models"]["neural_schema_aware"]
    sql = report["models"]["neural_sql_execution"]
    report["schema_signal"] = {
        "selection_ex_gain": schema["model_ex"] - sql["model_ex"],
        "gap_closure_gain": schema["gap_closure"] - sql["gap_closure"],
        "is_positive": schema["model_ex"] > sql["model_ex"],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-results-dir", required=True)
    parser.add_argument("--train-data-path", required=True)
    parser.add_argument("--db-root-dir", required=True)
    parser.add_argument("--eval-results-dir", required=True)
    parser.add_argument("--eval-data-path", required=True)
    parser.add_argument("--eval-db-root-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=(128, 64))
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pointwise-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-negatives-per-question", type=int, default=0)
    parser.add_argument(
        "--include-gold-positive", action=argparse.BooleanOptionalAction, default=False,
        help="Add train-only gold SQL positives; evaluation candidates are never modified.",
    )
    parser.add_argument(
        "--fusion-alpha-grid",
        type=float,
        nargs="+",
        default=(0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
        help="Validation-tuned weights for the neural residual over self-consistency.",
    )
    parser.add_argument(
        "--confidence-threshold-grid", type=float, nargs="+",
        default=(0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0),
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--validation-split-unit", choices=("database", "question"), default="database"
    )
    parser.add_argument("--patience", type=int, default=40)
    main(parser.parse_args())
