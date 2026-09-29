"""Run phase-one schema-signal ablations on existing Alpha-SQL candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
from typing import Any, Dict, List, Sequence
import warnings

import numpy as np

from alphasql.algorithm.selection.schema_signal import (
    FEATURE_GROUPS,
    LogisticReranker,
    extract_candidate_features,
    extract_linked_schema,
    extract_sql_usage,
    generate_sql_mutations,
    result_signature,
    selection_metrics,
    vectorize,
)
from alphasql.database.database_manager import DatabaseManager
from alphasql.database.sql_execution import SQLExecutionResultType, cached_execute_sql_with_timeout


def _load_dataset(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    return {str(row.get("question_id", index)): row for index, row in enumerate(rows)}


def _execution_summary(answer) -> Dict[str, Any]:
    success = answer.result_type is SQLExecutionResultType.SUCCESS
    rows = answer.result if success else None
    values = [value for row in (rows or []) for value in row]
    return {
        "success": success,
        "row_count": len(rows or []),
        "column_count": len(answer.result_cols or []),
        "null_ratio": sum(value is None for value in values) / len(values) if values else 0.0,
        "duplicate_ratio": 1.0 - len(set(map(repr, rows or []))) / len(rows) if rows else 0.0,
        "signature": result_signature(rows) if success else None,
    }


def build_records(
    results_dir: str,
    data_path: str,
    db_root_dir: str,
    hard_negatives_per_question: int = 0,
    include_gold_positive: bool = False,
) -> List[Dict[str, Any]]:
    dataset = _load_dataset(data_path)
    all_records = []
    for result_path in sorted(Path(results_dir).glob("*.pkl")):
        question_id = result_path.stem
        if question_id not in dataset:
            continue
        item = dataset[question_id]
        db_id = item["db_id"]
        gold_sql = item.get("SQL") or item.get("sql")
        if not gold_sql:
            raise ValueError(f"Question {question_id} has no gold SQL")
        db_path = Path(db_root_dir) / db_id / f"{db_id}.sqlite"
        schema = DatabaseManager.get_database_schema(db_id, db_root_dir)
        gold = _execution_summary(cached_execute_sql_with_timeout(str(db_path), gold_sql))
        if not gold["success"]:
            warnings.warn(
                f"Skipping question {question_id}: gold SQL does not execute on {db_id}",
                RuntimeWarning,
            )
            continue
        with open(result_path, "rb") as handle:
            paths = pickle.load(handle)
        # Schema selection is a question-level signal. MCTS paths may skip the
        # action entirely, so using each path's empty value confounds schema
        # agreement with search-route choice. Reuse the modal output of the
        # original linker across every candidate for this question.
        linked_options = []
        for path in paths:
            tables, columns = extract_linked_schema(path)
            if tables:
                linked_options.append((tuple(sorted(tables)), tuple(sorted(columns))))
        if linked_options:
            option_counts = Counter(linked_options)
            global_tables_tuple, global_columns_tuple = max(
                option_counts,
                key=lambda option: (option_counts[option], len(option[0]) + len(option[1])),
            )
            global_linked_tables = set(global_tables_tuple)
            global_linked_columns = set(global_columns_tuple)
        else:
            global_linked_tables, global_linked_columns = set(), set()
        pending = []
        for index, path in enumerate(paths):
            final_node = path[-1]
            sql = getattr(final_node, "final_sql_query", None) or getattr(final_node, "revised_sql_query", None)
            if not sql:
                continue
            execution = _execution_summary(cached_execute_sql_with_timeout(str(db_path), sql))
            pending.append((index, path, final_node, sql, execution))
        counts = Counter(entry[4]["signature"] for entry in pending if entry[4]["success"])
        successful = sum(counts.values())
        positive_seeds = []
        seen_sql = {sql.strip().lower() for _, _, _, sql, _ in pending}
        for index, path, final_node, sql, execution in pending:
            group_size = counts.get(execution["signature"], 0) if execution["success"] else 0
            execution.update({"group_size": group_size, "group_share": group_size / successful if successful else 0.0})
            path_linked_tables, _ = extract_linked_schema(path)
            linked_tables, linked_columns = global_linked_tables, global_linked_columns
            mcts = {
                "consistency_score": getattr(final_node, "consistency_score", 0.0),
                "path_length": len(path),
                "depth": getattr(final_node, "depth", 0),
                "path_has_schema_selection": bool(path_linked_tables),
            }
            label = int(execution["success"] and execution["signature"] == gold["signature"])
            usage = extract_sql_usage(sql, schema)
            record = {
                "question_id": question_id,
                "db_id": db_id,
                "question": item.get("question", ""),
                "evidence": item.get("evidence", ""),
                "candidate_index": index,
                "sql": sql,
                "candidate_schema": "tables: " + ", ".join(sorted(usage.tables))
                + "; columns: "
                + ", ".join(f"{table}.{column}" for table, column in sorted(usage.columns)),
                "linked_schema": "tables: " + ", ".join(sorted(linked_tables))
                + "; columns: "
                + ", ".join(f"{table}.{column}" for table, column in sorted(linked_columns)),
                "label": label,
                "candidate_source": "mcts",
                "features": extract_candidate_features(sql, schema, linked_tables, linked_columns, execution, mcts),
            }
            all_records.append(record)
            if label:
                positive_seeds.append((sql, linked_tables, linked_columns, mcts))

        if include_gold_positive and gold_sql.strip().lower() not in seen_sql:
            gold_execution = dict(gold)
            gold_group_size = counts.get(gold["signature"], 0)
            gold_execution.update({
                "group_size": gold_group_size,
                "group_share": gold_group_size / successful if successful else 0.0,
            })
            gold_mcts = {
                "consistency_score": 0.0,
                "path_length": 0,
                "depth": 0,
                "path_has_schema_selection": bool(linked_tables),
            }
            usage = extract_sql_usage(gold_sql, schema)
            all_records.append({
                "question_id": question_id,
                "db_id": db_id,
                "question": item.get("question", ""),
                "evidence": item.get("evidence", ""),
                "candidate_index": len(pending),
                "sql": gold_sql,
                "candidate_schema": "tables: " + ", ".join(sorted(usage.tables))
                + "; columns: "
                + ", ".join(
                    f"{table}.{column}" for table, column in sorted(usage.columns)
                ),
                "linked_schema": "tables: " + ", ".join(sorted(linked_tables))
                + "; columns: "
                + ", ".join(
                    f"{table}.{column}" for table, column in sorted(linked_columns)
                ),
                "label": 1,
                "candidate_source": "gold_train_positive",
                "features": extract_candidate_features(
                    gold_sql, schema, linked_tables, linked_columns,
                    gold_execution, gold_mcts,
                ),
            })
            positive_seeds.append((gold_sql, linked_tables, linked_columns, gold_mcts))

        added = 0
        for seed_sql, linked_tables, linked_columns, mcts in positive_seeds:
            for mutated_sql in generate_sql_mutations(seed_sql, hard_negatives_per_question * 2):
                key = mutated_sql.strip().lower()
                if key in seen_sql:
                    continue
                seen_sql.add(key)
                execution = _execution_summary(cached_execute_sql_with_timeout(str(db_path), mutated_sql))
                if not execution["success"] or execution["signature"] == gold["signature"]:
                    continue
                execution.update({"group_size": 1, "group_share": 0.0})
                all_records.append({
                    "question_id": question_id,
                    "db_id": db_id,
                    "question": item.get("question", ""),
                    "evidence": item.get("evidence", ""),
                    "candidate_index": len(pending) + added,
                    "sql": mutated_sql,
                    "candidate_schema": "",
                    "linked_schema": "tables: " + ", ".join(sorted(linked_tables))
                    + "; columns: "
                    + ", ".join(
                        f"{table}.{column}" for table, column in sorted(linked_columns)
                    ),
                    "label": 0,
                    "candidate_source": "synthetic_hard_negative",
                    "features": extract_candidate_features(
                        mutated_sql, schema, linked_tables, linked_columns, execution, mcts
                    ),
                })
                added += 1
                if added >= hard_negatives_per_question:
                    break
            if added >= hard_negatives_per_question:
                break
    return all_records


def split_records(records: Sequence[Dict[str, Any]], fraction: float, seed: int, split_unit: str):
    if not records:
        raise ValueError("No candidate records were built; check result paths and question IDs")
    key = "db_id" if split_unit == "database" else "question_id"
    group_ids = sorted({str(record[key]) for record in records})
    if len(group_ids) < 2 and split_unit == "database":
        key = "question_id"
        group_ids = sorted({str(record[key]) for record in records})
    if len(group_ids) < 2:
        raise ValueError("At least two questions are required when no separate evaluation set is provided")
    rng = np.random.default_rng(seed)
    rng.shuffle(group_ids)
    evaluation_count = min(len(group_ids) - 1, max(1, round(len(group_ids) * fraction)))
    evaluation_ids = set(group_ids[:evaluation_count])
    return (
        [record for record in records if str(record[key]) not in evaluation_ids],
        [record for record in records if str(record[key]) in evaluation_ids],
        key,
    )


def train_ablation(train_records, evaluation_records, groups, seed):
    train_x, train_y, names = vectorize(train_records, groups)
    evaluation_x, _, evaluation_names = vectorize(evaluation_records, groups)
    if names != evaluation_names:
        # Re-vectorize evaluation against the training vocabulary.
        evaluation_x = np.asarray([[float(record["features"].get(name, 0.0)) for name in names] for record in evaluation_records])
    model = LogisticReranker(seed=seed).fit(train_x, train_y)
    scores = model.predict_proba(evaluation_x)
    return model, names, scores, selection_metrics(evaluation_records, scores)


def _coefficient_summary(model, feature_names, limit=10):
    weighted = sorted(zip(feature_names, model.weights[:-1]), key=lambda item: item[1])
    return {
        "top_negative": [{"feature": name, "coefficient": float(value)} for name, value in weighted[:limit]],
        "top_positive": [{"feature": name, "coefficient": float(value)} for name, value in reversed(weighted[-limit:])],
    }


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_records = build_records(args.train_results_dir, args.train_data_path, args.db_root_dir)
    if args.eval_results_dir:
        evaluation_records = build_records(
            args.eval_results_dir,
            args.eval_data_path,
            args.eval_db_root_dir or args.db_root_dir,
        )
        split_key = "explicit_evaluation_set"
    else:
        train_records, evaluation_records, split_key = split_records(
            train_records, args.validation_fraction, args.seed, args.split_unit
        )

    ablations = {
        "execution_only": ["execution"],
        "sql_execution": ["sql", "execution"],
        "schema_aware": ["sql", "schema", "execution"],
        "schema_aware_mcts": list(FEATURE_GROUPS),
    }
    report = {
        "train_candidates": len(train_records),
        "evaluation_candidates": len(evaluation_records),
        "split_key": split_key,
        "ablations": {},
    }
    predictions = []
    for name, groups in ablations.items():
        model, feature_names, scores, metrics = train_ablation(train_records, evaluation_records, groups, args.seed)
        report["ablations"][name] = {
            "feature_groups": groups,
            "feature_count": len(feature_names),
            **metrics,
            "standardized_coefficients": _coefficient_summary(model, feature_names),
        }
        if name == "schema_aware":
            np.savez(output_dir / "schema_aware_logistic.npz", weights=model.weights, mean=model.scaler.mean, scale=model.scaler.scale, feature_names=np.asarray(feature_names))
            predictions = [{"question_id": record["question_id"], "candidate_index": record["candidate_index"], "label": record["label"], "score": float(score), "sql": record["sql"]} for record, score in zip(evaluation_records, scores)]

    schema_metrics = report["ablations"]["schema_aware"]
    sql_metrics = report["ablations"]["sql_execution"]
    report["schema_signal"] = {
        "selection_ex_gain": schema_metrics["model_ex"] - sql_metrics["model_ex"],
        "gap_closure_gain": schema_metrics["gap_closure"] - sql_metrics["gap_closure"],
        "is_positive": schema_metrics["model_ex"] > sql_metrics["model_ex"],
        "interpretation": "Schema signal is useful only if the gain is stable across held-out splits or a separate evaluation set.",
    }

    with open(output_dir / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    with open(output_dir / "predictions.jsonl", "w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-results-dir", required=True)
    parser.add_argument("--train-data-path", required=True)
    parser.add_argument("--db-root-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-results-dir")
    parser.add_argument("--eval-data-path")
    parser.add_argument("--eval-db-root-dir")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-unit", choices=("database", "question"), default="database")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if bool(args.eval_results_dir) != bool(args.eval_data_path):
        parser.error("--eval-results-dir and --eval-data-path must be provided together")
    main(args)
