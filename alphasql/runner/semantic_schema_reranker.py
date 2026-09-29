"""Train a frozen-cross-encoder candidate reranker without running MCTS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from alphasql.algorithm.selection.neural_reranker import NeuralPairwiseReranker
from alphasql.algorithm.selection.schema_signal import selection_metrics, vectorize
from alphasql.runner.neural_schema_reranker import (
    confidence_gated_scores,
    fuse_with_self_consistency,
    tune_confidence_threshold,
    tune_fusion_alpha,
)
from alphasql.runner.schema_signal_experiment import build_records, split_records


def semantic_pairs(records, schema_aware: bool):
    questions, candidates = [], []
    for record in records:
        question = record.get("question", "")
        evidence = record.get("evidence", "")
        questions.append(question + (f" Evidence: {evidence}" if evidence else ""))
        candidate = f"SQL: {record['sql']}"
        if schema_aware:
            candidate += f" Candidate schema: {record.get('candidate_schema', '')}"
            candidate += f" Linked schema: {record.get('linked_schema', '')}"
        candidates.append(candidate)
    return questions, candidates


def encode_records(tokenizer, encoder, records, schema_aware, device, batch_size, max_length):
    questions, candidates = semantic_pairs(records, schema_aware)
    rows = []
    encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = tokenizer(
                questions[start:start + batch_size],
                candidates[start:start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {name: value.to(device) for name, value in batch.items()}
            output = encoder(**batch, output_hidden_states=True)
            hidden = output.hidden_states[-1][:, 0]
            hidden = torch.nn.functional.normalize(hidden.float(), dim=1)
            # This checkpoint was already trained as an MS MARCO cross-encoder.
            # Preserve its relevance prior instead of relearning it from the
            # very small Text-to-SQL candidate pool.
            relevance = output.logits.float().reshape(-1, 1)
            rows.append(torch.cat((hidden, relevance), dim=1).cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float32)


def numeric_features(train_records, other_records, groups):
    train_x, _, names = vectorize(train_records, groups)
    other_x = np.asarray([
        [float(record["features"].get(name, 0.0)) for name in names]
        for record in other_records
    ], dtype=np.float32)
    return train_x.astype(np.float32), other_x, names


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_train = build_records(
        args.train_results_dir, args.train_data_path, args.db_root_dir,
        include_gold_positive=args.include_gold_positive,
    )
    evaluation = build_records(
        args.eval_results_dir, args.eval_data_path, args.eval_db_root_dir
    )
    train, validation, validation_key = split_records(
        all_train, args.validation_fraction, args.seed, args.validation_split_unit
    )

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.encoder, cache_dir=args.model_cache_dir, local_files_only=True
    )
    encoder = AutoModelForSequenceClassification.from_pretrained(
        args.encoder, cache_dir=args.model_cache_dir, local_files_only=True
    ).to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    combined_records = train + validation + evaluation
    report = {
        "encoder": args.encoder,
        "encoder_frozen": True,
        "device": str(device),
        "train_candidates": len(train),
        "validation_candidates": len(validation),
        "evaluation_candidates": len(evaluation),
        "validation_split": validation_key,
        "models": {},
    }
    predictions = []
    layouts = {
        "semantic_sql_execution": (False, ["sql", "execution"]),
        "semantic_schema_aware": (True, ["sql", "schema", "execution"]),
    }
    for name, (schema_aware, groups) in layouts.items():
        embeddings = encode_records(
            tokenizer, encoder, combined_records, schema_aware, device,
            args.encoder_batch_size, args.max_length,
        )
        train_end = len(train)
        validation_end = train_end + len(validation)
        train_numeric, validation_numeric, feature_names = numeric_features(
            train, validation, groups
        )
        _, evaluation_numeric, _ = numeric_features(train, evaluation, groups)
        train_x = np.concatenate((train_numeric, embeddings[:train_end]), axis=1)
        validation_x = np.concatenate(
            (validation_numeric, embeddings[train_end:validation_end]), axis=1
        )
        evaluation_x = np.concatenate(
            (evaluation_numeric, embeddings[validation_end:]), axis=1
        )
        labels = np.asarray([record["label"] for record in train], dtype=np.float32)
        model = NeuralPairwiseReranker(
            hidden_sizes=tuple(args.hidden_sizes), dropout=args.dropout,
            epochs=args.epochs, learning_rate=args.learning_rate,
            weight_decay=args.weight_decay, batch_size=args.batch_size,
            pointwise_weight=args.pointwise_weight, seed=args.seed,
            device=str(device), patience=args.patience,
        ).fit(
            train_x, labels, train,
            validation_values=validation_x, validation_records=validation,
        )
        validation_scores = model.predict_proba(validation_x)
        raw_scores = model.predict_proba(evaluation_x)
        alpha, trials = tune_fusion_alpha(
            validation, validation_scores, args.fusion_alpha_grid
        )
        threshold, gate_trials = tune_confidence_threshold(
            validation, validation_scores, args.confidence_threshold_grid
        )
        if max(row["model_ex"] for row in gate_trials) > max(
            row["model_ex"] for row in trials
        ):
            decision_policy = "confidence_gate"
            scores = confidence_gated_scores(evaluation, raw_scores, threshold)
        else:
            decision_policy = "residual_fusion"
            scores = fuse_with_self_consistency(evaluation, raw_scores, alpha)
        names = feature_names + [f"semantic_{index}" for index in range(embeddings.shape[1])]
        model.save(output_dir / f"{name}.pt", names)
        report["models"][name] = {
            "numeric_feature_groups": groups,
            "feature_count": len(names),
            "training_pairs": model.pair_count,
            "best_epoch": model.best_epoch,
            "training_history": model.training_history,
            "raw_neural_metrics": selection_metrics(evaluation, raw_scores),
            "fusion_alpha": alpha,
            "fusion_validation_trials": trials,
            "confidence_threshold": threshold,
            "confidence_gate_validation_trials": gate_trials,
            "decision_policy": decision_policy,
            **selection_metrics(evaluation, scores),
        }
        if schema_aware:
            predictions = [
                {
                    "question_id": record["question_id"],
                    "candidate_index": record["candidate_index"],
                    "label": record["label"],
                    "score": float(score),
                    "raw_neural_score": float(raw),
                    "sql": record["sql"],
                }
                for record, score, raw in zip(evaluation, scores, raw_scores)
            ]

    schema = report["models"]["semantic_schema_aware"]
    sql = report["models"]["semantic_sql_execution"]
    report["schema_signal"] = {
        "selection_ex_gain": schema["model_ex"] - sql["model_ex"],
        "raw_selection_ex_gain": (
            schema["raw_neural_metrics"]["model_ex"]
            - sql["raw_neural_metrics"]["model_ex"]
        ),
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
    parser.add_argument("--encoder", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--model-cache-dir", default="data/models/hf/hub")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--encoder-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=(64,))
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pointwise-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-gold-positive", action=argparse.BooleanOptionalAction, default=False,
        help="Add train-only gold SQL positives; evaluation candidates are never modified.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--validation-split-unit", choices=("database", "question"), default="database"
    )
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--fusion-alpha-grid", type=float, nargs="+",
        default=(0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
    )
    parser.add_argument(
        "--confidence-threshold-grid", type=float, nargs="+",
        default=(0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0),
    )
    main(parser.parse_args())
