"""Train a hard-case pairwise SQL reranker and evaluate a frozen 100-question split."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
from sqlglot import exp, parse_one
import torch

from alphasql.algorithm.selection.neural_reranker import NeuralPairwiseReranker
from alphasql.algorithm.selection.schema_signal import extract_sql_usage, selection_metrics
from alphasql.database.database_manager import DatabaseManager
from alphasql.runner.neural_schema_reranker import (
    cluster_support_gated_scores,
    confidence_gated_scores,
    fuse_with_self_consistency,
    tune_confidence_threshold,
    tune_fusion_alpha,
    tune_cluster_support_gate,
)
from alphasql.runner.schema_signal_experiment import build_records


SELF_CONSISTENCY_FEATURES = {
    "exec_result_group_size",
    "exec_result_group_share",
    "exec_result_group_rank",
    "exec_result_group_rank_inverse",
    "exec_result_group_ratio_to_top",
    "exec_self_consistency_winner",
    "mcts_consistency_score",
}


def group_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, list]:
    grouped: Dict[str, list] = defaultdict(list)
    for record in records:
        grouped[str(record["question_id"])].append(record)
    return dict(grouped)


def self_consistency_candidate(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return max(records, key=lambda record: (
        float(record["features"].get("exec_result_group_size", 0.0)),
        -int(record.get("candidate_index", 0)),
    ))


def question_kind(records: Sequence[Mapping[str, Any]]) -> str:
    if not any(record["label"] for record in records):
        return "coverage_miss"
    return "baseline_correct" if self_consistency_candidate(records)["label"] else "hard"


def _stratified_take(
    question_ids: Sequence[str],
    count: int,
    db_by_question: Mapping[str, str],
    rng: random.Random,
) -> list[str]:
    pools: Dict[str, list[str]] = defaultdict(list)
    for question_id in question_ids:
        pools[db_by_question[question_id]].append(question_id)
    for pool in pools.values():
        rng.shuffle(pool)
    total = len(question_ids)
    exact = {database: count * len(pool) / total for database, pool in pools.items()}
    quota = {database: min(len(pools[database]), int(value)) for database, value in exact.items()}
    remaining = count - sum(quota.values())
    tie_break = {database: rng.random() for database in pools}
    order = sorted(
        pools,
        key=lambda database: (
            exact[database] - int(exact[database]), tie_break[database]
        ),
        reverse=True,
    )
    while remaining:
        progressed = False
        for database in order:
            if quota[database] < len(pools[database]):
                quota[database] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise ValueError("Cannot satisfy requested stratified sample size")
    selected = [
        question_id
        for database in sorted(pools)
        for question_id in pools[database][:quota[database]]
    ]
    rng.shuffle(selected)
    return selected


def prepare_split(
    records: Sequence[Mapping[str, Any]],
    seed: int,
    test_size: int = 100,
    validation_size: int = 40,
    test_hard_count: int = 5,
    validation_hard_count: int = 3,
) -> Dict[str, Any]:
    grouped = group_records(records)
    if len(grouped) < test_size + validation_size + 1:
        raise ValueError("Candidate pool is too small for the requested split")
    kinds = {question_id: question_kind(items) for question_id, items in grouped.items()}
    hard = sorted(question_id for question_id, kind in kinds.items() if kind == "hard")
    if len(hard) < test_hard_count + validation_hard_count + 1:
        raise ValueError("Not enough hard cases to populate train/validation/test")
    db_by_question = {
        question_id: str(items[0]["db_id"]) for question_id, items in grouped.items()
    }
    rng = random.Random(seed)
    rng.shuffle(hard)
    test_hard = hard[:test_hard_count]
    validation_hard = hard[test_hard_count:test_hard_count + validation_hard_count]
    reserved_hard = set(test_hard + validation_hard)
    nonhard = sorted(question_id for question_id in grouped if question_id not in set(hard))
    test_nonhard = _stratified_take(
        nonhard, test_size - len(test_hard), db_by_question, rng
    )
    used = reserved_hard | set(test_nonhard)
    remaining_nonhard = [question_id for question_id in nonhard if question_id not in used]
    validation_nonhard = _stratified_take(
        remaining_nonhard,
        validation_size - len(validation_hard),
        db_by_question,
        rng,
    )
    test_ids = sorted(test_hard + test_nonhard, key=int)
    validation_ids = sorted(validation_hard + validation_nonhard, key=int)
    train_ids = sorted(
        set(grouped) - set(test_ids) - set(validation_ids), key=int
    )
    return {
        "seed": seed,
        "split_unit": "question_id_stratified_by_database_and_hard_status",
        "candidate_pool_questions": len(grouped),
        "candidate_pool_hard_questions": len(hard),
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "test_ids": test_ids,
        "train_hard_ids": sorted(
            [question_id for question_id in train_ids if kinds[question_id] == "hard"], key=int
        ),
        "validation_hard_ids": sorted(validation_hard, key=int),
        "test_hard_ids": sorted(test_hard, key=int),
        "counts": {
            "train": len(train_ids),
            "validation": len(validation_ids),
            "test": len(test_ids),
            "train_hard": sum(kinds[value] == "hard" for value in train_ids),
            "validation_hard": len(validation_hard),
            "test_hard": len(test_hard),
        },
        "database_counts": {
            split: dict(sorted(Counter(db_by_question[value] for value in ids).items()))
            for split, ids in (
                ("train", train_ids), ("validation", validation_ids), ("test", test_ids)
            )
        },
    }


def _feature_names(records: Sequence[Mapping[str, Any]], include_self_consistency: bool) -> list[str]:
    names = sorted({name for record in records for name in record["features"]})
    if not include_self_consistency:
        names = [name for name in names if name not in SELF_CONSISTENCY_FEATURES]
    return names


def _matrix(records: Sequence[Mapping[str, Any]], names: Sequence[str]) -> np.ndarray:
    return np.asarray([
        [float(record["features"].get(name, 0.0)) for name in names]
        for record in records
    ], dtype=np.float32)


def _paired_bootstrap(
    records: Sequence[Mapping[str, Any]], scores: Sequence[float], repeats: int = 10000
) -> Dict[str, float]:
    grouped = group_records(records)
    by_question = []
    offset = 0
    for question_records in grouped.values():
        question_scores = scores[offset:offset + len(question_records)]
        baseline = int(self_consistency_candidate(question_records)["label"])
        selected = max(
            zip(question_records, question_scores),
            key=lambda item: (item[1], -int(item[0].get("candidate_index", 0))),
        )[0]
        by_question.append(int(selected["label"]) - baseline)
        offset += len(question_records)
    rng = random.Random(42)
    samples = sorted(
        sum(by_question[rng.randrange(len(by_question))] for _ in by_question) / len(by_question)
        for _ in range(repeats)
    )
    return {
        "delta": sum(by_question) / len(by_question),
        "ci95_low": samples[int(0.025 * repeats)],
        "ci95_high": samples[int(0.975 * repeats)],
    }


def train_ensemble(
    name: str,
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
    evaluation_records: Sequence[Mapping[str, Any]],
    include_self_consistency: bool,
    seeds: Sequence[int],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[Dict[str, Any], np.ndarray]:
    names = _feature_names(train_records, include_self_consistency)
    train_x = _matrix(train_records, names)
    validation_x = _matrix(validation_records, names)
    evaluation_x = _matrix(evaluation_records, names)
    labels = np.asarray([record["label"] for record in train_records], dtype=np.float32)
    validation_predictions = []
    evaluation_predictions = []
    model_rows = []
    for seed in seeds:
        model = NeuralPairwiseReranker(
            hidden_sizes=tuple(args.hidden_sizes),
            dropout=args.dropout,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            pointwise_weight=args.pointwise_weight,
            seed=seed,
            device=args.device,
            patience=args.patience,
        ).fit(
            train_x,
            labels,
            train_records,
            validation_values=validation_x,
            validation_records=validation_records,
        )
        validation_predictions.append(model.predict_proba(validation_x))
        evaluation_predictions.append(model.predict_proba(evaluation_x))
        model.save(output_dir / f"{name}_seed{seed}.pt", names)
        model_rows.append({
            "seed": seed,
            "best_epoch": model.best_epoch,
            "training_pairs": model.pair_count,
            "device": str(model.device),
        })
    validation_scores = np.mean(validation_predictions, axis=0)
    raw_scores = np.mean(evaluation_predictions, axis=0)
    alpha, fusion_trials = tune_fusion_alpha(
        validation_records, validation_scores, args.fusion_alpha_grid
    )
    threshold, gate_trials = tune_confidence_threshold(
        validation_records, validation_scores, args.confidence_threshold_grid
    )
    cluster_ratio, cluster_threshold, cluster_trials = tune_cluster_support_gate(
        validation_records,
        validation_scores,
        args.confidence_threshold_grid,
        args.cluster_ratio_grid,
    )
    fusion_scores = fuse_with_self_consistency(evaluation_records, raw_scores, alpha)
    gate_scores = confidence_gated_scores(evaluation_records, raw_scores, threshold)
    cluster_scores = cluster_support_gated_scores(
        evaluation_records, raw_scores, cluster_threshold, cluster_ratio
    )
    fusion_validation = max(row["model_ex"] for row in fusion_trials)
    gate_validation = max(row["model_ex"] for row in gate_trials)
    cluster_validation = max(row["model_ex"] for row in cluster_trials)
    # A cluster-preserving gate is the safest tie breaker when validation EX
    # ties.  It cannot discard self-consistency support without validation
    # explicitly selecting a ratio below 1.0.
    _, policy, final_scores = max(
        (
            (fusion_validation, "residual_fusion", fusion_scores),
            (gate_validation + 1e-9, "confidence_gate", gate_scores),
            (cluster_validation + 2e-9, "cluster_support_gate", cluster_scores),
        ),
        key=lambda item: item[0],
    )
    return ({
        "training_questions": len(group_records(train_records)),
        "training_candidates": len(train_records),
        "feature_count": len(names),
        "feature_names": names,
        "includes_self_consistency_features": include_self_consistency,
        "ensemble": model_rows,
        "raw_neural_metrics": selection_metrics(evaluation_records, raw_scores),
        "fusion_alpha": alpha,
        "fusion_validation_trials": fusion_trials,
        "confidence_threshold": threshold,
        "confidence_gate_validation_trials": gate_trials,
        "cluster_support_min_ratio": cluster_ratio,
        "cluster_support_threshold": cluster_threshold,
        "cluster_support_validation_trials": cluster_trials,
        "decision_policy": policy,
        **selection_metrics(evaluation_records, final_scores),
    }, np.asarray(final_scores), raw_scores)


def _shape(sql: str, schema) -> Dict[str, Any] | None:
    try:
        tree = parse_one(sql, read="sqlite")
    except Exception:
        return None
    usage = extract_sql_usage(sql, schema)
    select = next(tree.find_all(exp.Select), None)
    projections = []
    if select is not None:
        for expression in select.expressions:
            projections.append((
                tuple(sorted(column.name.lower() for column in expression.find_all(exp.Column))),
                tuple(sorted(type(node).__name__ for node in expression.find_all(exp.AggFunc))),
                bool(expression.find(exp.Star)),
            ))
    where = tree.find(exp.Where)
    filters = () if where is None else tuple(sorted(
        (
            type(node).__name__,
            tuple(sorted(column.name.lower() for column in node.find_all(exp.Column))),
            tuple(literal.this for literal in node.find_all(exp.Literal)),
        )
        for node in where.walk()
        if isinstance(node, exp.Predicate)
    ))
    return {
        "tables": tuple(sorted(usage.tables)),
        "projections": tuple(projections),
        "filters": filters,
        "joins": len(list(tree.find_all(exp.Join))),
        "group": bool(tree.find(exp.Group)),
        "having": bool(tree.find(exp.Having)),
        "order": bool(tree.find(exp.Order)),
        "limit": bool(tree.find(exp.Limit)),
        "distinct": bool(tree.find(exp.Distinct)),
        "set_operation": any(tree.find(kind) is not None for kind in (exp.Union, exp.Intersect, exp.Except)),
    }


def diagnose_sql(gold_sql: str, candidate: Mapping[str, Any], schema) -> list[str]:
    if not candidate["features"].get("exec_success", 0.0):
        return ["execution_error"]
    gold, predicted = _shape(gold_sql, schema), _shape(candidate["sql"], schema)
    if gold is None or predicted is None:
        return ["parse_error"]
    errors = []
    if gold["projections"] != predicted["projections"]:
        errors.append("projection")
    if gold["tables"] != predicted["tables"] or gold["joins"] != predicted["joins"]:
        errors.append("table_or_join")
    if gold["filters"] != predicted["filters"]:
        errors.append("filter")
    for name in ("group", "having", "order", "limit", "distinct", "set_operation"):
        if gold[name] != predicted[name]:
            errors.append(name)
    return errors or ["result_or_semantic_mismatch"]


def _select(records: Sequence[Mapping[str, Any]], scores: Sequence[float]) -> Mapping[str, Any]:
    return max(
        zip(records, scores),
        key=lambda item: (float(item[1]), -int(item[0].get("candidate_index", 0))),
    )[0]


def detailed_analysis(
    evaluation_records: Sequence[Mapping[str, Any]],
    final_scores: Sequence[float],
    raw_scores: Sequence[float],
    dataset_path: str,
    db_root: str,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    dataset_rows = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    dataset = {
        str(row.get("question_id", index)): row for index, row in enumerate(dataset_rows)
    }
    grouped = group_records(evaluation_records)
    rows = []
    offset = 0
    for question_id, candidates in grouped.items():
        count = len(candidates)
        policy_values = final_scores[offset:offset + count]
        raw_values = raw_scores[offset:offset + count]
        offset += count
        baseline = self_consistency_candidate(candidates)
        selected = _select(candidates, policy_values)
        selected_position = candidates.index(selected)
        kind = question_kind(candidates)
        baseline_correct = bool(baseline["label"])
        model_correct = bool(selected["label"])
        if not any(record["label"] for record in candidates):
            outcome = "coverage_miss"
        elif not baseline_correct and model_correct:
            outcome = "fixed_by_reranker"
        elif baseline_correct and not model_correct:
            outcome = "regression"
        elif baseline_correct and model_correct:
            outcome = "both_correct"
        else:
            outcome = "unfixed_selection_error"
        item = dataset[question_id]
        schema = DatabaseManager.get_database_schema(item["db_id"], db_root)
        gold_sql = item.get("SQL") or item.get("sql")
        rows.append({
            "question_id": question_id,
            "db_id": item["db_id"],
            "question": item.get("question", ""),
            "evidence": item.get("evidence", ""),
            "question_kind": kind,
            "outcome": outcome,
            "candidate_count": count,
            "positive_candidate_count": sum(record["label"] for record in candidates),
            "baseline_correct": baseline_correct,
            "model_correct": model_correct,
            "model_changed_selection": baseline is not selected,
            "baseline_group_size": baseline["features"].get("exec_result_group_size", 0),
            "selected_group_size": selected["features"].get("exec_result_group_size", 0),
            "selected_raw_neural_score": float(raw_values[selected_position]),
            "neural_margin_over_baseline": float(
                raw_values[selected_position] - raw_values[candidates.index(baseline)]
            ),
            "gold_sql": gold_sql,
            "baseline_sql": baseline["sql"],
            "selected_sql": selected["sql"],
            "baseline_error_types": "|".join(diagnose_sql(gold_sql, baseline, schema)) if not baseline_correct else "",
            "selected_error_types": "|".join(diagnose_sql(gold_sql, selected, schema)) if not model_correct else "",
        })
    errors = [row for row in rows if not row["model_correct"]]
    summary = {
        "outcome_counts": dict(Counter(row["outcome"] for row in rows)),
        "model_error_type_counts": dict(Counter(
            error
            for row in errors
            for error in row["selected_error_types"].split("|") if error
        )),
        "errors_by_database": dict(Counter(row["db_id"] for row in errors)),
        "hard_case_outcomes": dict(Counter(
            row["outcome"] for row in rows if row["question_kind"] == "hard"
        )),
    }
    return rows, summary


def _subset(records: Sequence[Mapping[str, Any]], ids: Iterable[str]) -> list:
    ids = set(ids)
    return [record for record in records if str(record["question_id"]) in ids]


def main(args: argparse.Namespace) -> None:
    records = build_records(args.candidate_dir, args.data_path, args.db_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = Path(args.split_file)
    if args.mode == "prepare":
        split = prepare_split(records, args.seed)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(split["counts"], indent=2))
        return

    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_records = _subset(records, split["train_ids"])
    validation_records = _subset(records, split["validation_ids"])
    evaluation_records = _subset(records, split["test_ids"])
    if len(group_records(evaluation_records)) != 100:
        raise ValueError("Frozen evaluation split must contain exactly 100 questions")
    hard_train_records = _subset(train_records, split["train_hard_ids"])
    covered_train_ids = {
        question_id for question_id, items in group_records(train_records).items()
        if any(record["label"] for record in items)
        and any(not record["label"] for record in items)
    }
    covered_train_records = _subset(train_records, covered_train_ids)
    models = {}
    scores = {}
    raw_scores = {}
    for name, source, include_sc in (
        ("hard_focused_with_sc", hard_train_records, True),
        ("hard_focused_without_sc", hard_train_records, False),
        ("all_covered_with_sc", covered_train_records, True),
    ):
        models[name], scores[name], raw_scores[name] = train_ensemble(
            name, source, validation_records, evaluation_records,
            include_sc, args.seeds, args, output_dir,
        )
    primary = "hard_focused_with_sc"
    hard_evaluation = _subset(evaluation_records, split["test_hard_ids"])
    evaluation_grouped = group_records(evaluation_records)
    hard_positions = []
    position = 0
    for question_id, items in evaluation_grouped.items():
        if question_id in set(split["test_hard_ids"]):
            hard_positions.extend(range(position, position + len(items)))
        position += len(items)
    hard_scores = scores[primary][hard_positions]
    rows, error_summary = detailed_analysis(
        evaluation_records, scores[primary], raw_scores[primary], args.data_path, args.db_root
    )
    outcome_counts = Counter(row["outcome"] for row in rows)
    report = {
        "experiment": "bird_train_hard_case_pairwise_100",
        "candidate_source": "cached direct_schema_pool_300 candidates",
        "split": split,
        "pool": {
            "nominal_questions": len(json.loads(Path(args.data_path).read_text(encoding="utf-8"))),
            "evaluable_questions": len(group_records(records)),
            "candidate_records": len(records),
            "oracle_covered_questions": sum(
                question_kind(items) != "coverage_miss" for items in group_records(records).values()
            ),
            "hard_questions": sum(
                question_kind(items) == "hard" for items in group_records(records).values()
            ),
        },
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(args.device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "models": models,
        "primary_model": primary,
        "primary_paired_bootstrap": _paired_bootstrap(
            evaluation_records, scores[primary]
        ),
        "primary_test_hard_metrics": selection_metrics(hard_evaluation, hard_scores),
        "primary_outcome_counts": dict(outcome_counts),
        "error_summary": error_summary,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = list(rows[0])
    with (output_dir / "per_question.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "errors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row for row in rows if not row["model_correct"])
    primary_metrics = models[primary]
    without_sc = models["hard_focused_without_sc"]
    all_covered = models["all_covered_with_sc"]
    markdown = [
        "# BIRD Train Hard-case Pairwise Reranker: 100-question Experiment", "",
        "The split was frozen before training: 129 train / 40 validation / 100 test questions, "
        "with 9 / 3 / 5 hard cases respectively.", "",
        "| Selector | Test EX | Oracle | Gap closure |", "|---|---:|---:|---:|",
        f"| Self-consistency | {primary_metrics['baseline_ex']:.3f} | {primary_metrics['oracle_ex']:.3f} | 0.000 |",
        f"| Hard-focused pairwise + self-consistency | {primary_metrics['model_ex']:.3f} | {primary_metrics['oracle_ex']:.3f} | {primary_metrics['gap_closure']:.3f} |",
        f"| Hard-focused pairwise without SC features | {without_sc['model_ex']:.3f} | {without_sc['oracle_ex']:.3f} | {without_sc['gap_closure']:.3f} |",
        f"| All-covered pairwise + self-consistency | {all_covered['model_ex']:.3f} | {all_covered['oracle_ex']:.3f} | {all_covered['gap_closure']:.3f} |", "",
        f"Primary paired delta: {report['primary_paired_bootstrap']['delta']:+.3f} "
        f"(95% bootstrap CI {report['primary_paired_bootstrap']['ci95_low']:+.3f} to "
        f"{report['primary_paired_bootstrap']['ci95_high']:+.3f}).", "",
        f"Outcomes: {dict(outcome_counts)}", "",
        f"Error types: {error_summary['model_error_type_counts']}", "",
        "The 300-question cache yielded only 269 executable-gold questions and 17 hard cases. "
        "Consequently, the supervised result is exploratory even though the test set contains exactly 100 questions.",
    ]
    (output_dir / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({
        "primary": primary_metrics,
        "without_sc": without_sc,
        "all_covered": all_covered,
        "outcomes": dict(outcome_counts),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "train"), required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--db-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=(7, 21, 42, 84, 168))
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=(64, 32))
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pointwise-weight", type=float, default=0.20)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fusion-alpha-grid", type=float, nargs="+",
        default=(0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
    )
    parser.add_argument(
        "--confidence-threshold-grid", type=float, nargs="+",
        default=(0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0),
    )
    parser.add_argument(
        "--cluster-ratio-grid", type=float, nargs="+",
        default=(0.5, 0.75, 1.0),
    )
    main(parser.parse_args())
