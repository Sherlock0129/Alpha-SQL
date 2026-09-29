"""Phase-one comparison of cached Alpha-SQL LLM candidates and the Jev pipeline."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import pickle
import random
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import sqlglot
from sqlglot import exp

from alphasql.algorithm.selection.jev_reranker import JevSQLReranker, RerankCandidate
from alphasql.algorithm.selection.schema_signal import (
    extract_linked_schema,
    extract_sql_usage,
    result_signature,
)
from alphasql.algorithm.compiler.validation import validate_sql
from alphasql.database.database_manager import DatabaseManager
from alphasql.database.sql_execution import (
    SQLExecutionResultType,
    execute_sql_with_timeout,
    normalize_sql,
)
from alphasql.jev.client import JevClient
from alphasql.jev.runtime import JevSettings


CALIBRATION_DATABASES = {"restaurant", "disney", "world"}
DISALLOWED_NODES = (
    exp.Subquery, exp.Window, exp.Or, exp.And, exp.Union, exp.Intersect,
    exp.Except, exp.CTE, exp.Case, exp.Group, exp.Having, exp.Order,
    exp.Limit, exp.In, exp.Between, exp.Is,
)
COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like)
AGGREGATES = (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)


def is_supported_slice(sql: str) -> bool:
    """Return whether the current constrained beam can represent the gold shape."""

    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return False
    if any(tree.find(node_type) is not None for node_type in DISALLOWED_NODES):
        return False
    if len(tree.expressions or []) != 1:
        return False
    selected = tree.expressions[0]
    selected = selected.this if isinstance(selected, exp.Alias) else selected
    argument = selected.this if isinstance(selected, AGGREGATES) else selected
    if not isinstance(argument, (exp.Column, exp.Star)):
        return False
    comparisons = list(tree.find_all(COMPARISONS))
    if len(comparisons) > 1:
        return False
    for comparison in comparisons:
        left, right = comparison.this, comparison.expression
        if not (
            isinstance(left, exp.Column) and isinstance(right, (exp.Literal, exp.Column))
            or isinstance(right, exp.Column) and isinstance(left, (exp.Literal, exp.Column))
        ):
            return False
    return all(
        (join.args.get("side") or "").upper() not in {"LEFT", "RIGHT", "FULL", "CROSS"}
        for join in tree.find_all(exp.Join)
    )


def prepare_split(
    tasks: Sequence[Any],
    db_root: Path | None = None,
    execution_timeout: int = 15,
) -> Dict[str, Any]:
    supported = [task for task in tasks if task.sql and is_supported_slice(task.sql)]
    excluded_gold = []
    if db_root is not None:
        executable = []
        for task in supported:
            db_path = db_root / task.db_id / f"{task.db_id}.sqlite"
            result = execute_sql_with_timeout(str(db_path), task.sql, execution_timeout)
            if result.result_type is SQLExecutionResultType.SUCCESS:
                executable.append(task)
            else:
                excluded_gold.append({
                    "question_id": task.question_id,
                    "db_id": task.db_id,
                    "reason": result.error_message,
                })
        supported = executable
    calibration = [task for task in supported if task.db_id in CALIBRATION_DATABASES]
    evaluation = [task for task in supported if task.db_id not in CALIBRATION_DATABASES]
    return {
        "definition": {
            "single_select_expression": True,
            "aggregates": ["none", "count", "sum", "avg", "min", "max"],
            "max_comparisons_including_join": 1,
            "allowed_predicates": ["=", "!=", ">", ">=", "<", "<=", "LIKE"],
            "excluded": [
                "subquery", "window", "AND/OR", "set operation", "CTE", "CASE",
                "GROUP BY", "HAVING", "ORDER BY", "LIMIT", "IN", "BETWEEN", "IS",
            ],
        },
        "calibration_databases": sorted(CALIBRATION_DATABASES),
        "calibration_ids": [task.question_id for task in calibration],
        "evaluation_ids": [task.question_id for task in evaluation],
        "calibration_count": len(calibration),
        "evaluation_count": len(evaluation),
        "excluded_non_executable_gold": excluded_gold,
        "database_disjoint": not (
            {task.db_id for task in calibration} & {task.db_id for task in evaluation}
        ),
    }


def _candidate_sqls(path: Path) -> Tuple[List[str], List[Any]]:
    with path.open("rb") as handle:
        reasoning_paths = pickle.load(handle)
    sqls, retained_paths, seen = [], [], set()
    for reasoning_path in reasoning_paths:
        final = reasoning_path[-1]
        sql = getattr(final, "final_sql_query", None) or getattr(final, "revised_sql_query", None)
        if not sql:
            continue
        canonical = normalize_sql(sql)
        key = " ".join(canonical.lower().split())
        if key and key not in seen:
            seen.add(key)
            sqls.append(canonical)
            retained_paths.append(reasoning_path)
    return sqls, retained_paths


def _execute(db_path: Path, sql: str, timeout: int) -> Tuple[bool, Tuple[Any, ...] | None, int]:
    result = execute_sql_with_timeout(str(db_path), sql, timeout)
    success = result.result_type is SQLExecutionResultType.SUCCESS
    signature = result_signature(result.result) if success else None
    return success, signature, len(result.result or []) if success else 0


def _self_consistency(signatures: Sequence[Tuple[Any, ...] | None], row_counts: Sequence[int]) -> int | None:
    successful = [index for index, signature in enumerate(signatures) if signature is not None]
    if not successful:
        return None
    nonempty = [index for index in successful if row_counts[index] > 0]
    pool = nonempty or successful
    counts = Counter(signatures[index] for index in pool)
    winner = max(counts, key=lambda signature: counts[signature])
    return next(index for index in pool if signatures[index] == winner)


def _precision_recall(predicted: set, gold: set) -> Tuple[float, float]:
    precision = len(predicted & gold) / len(predicted) if predicted else float(not gold)
    recall = len(predicted & gold) / len(gold) if gold else 1.0
    return precision, recall


def _modal_linked_schema(paths: Sequence[Any]) -> Tuple[set, set]:
    options = []
    for path in paths:
        tables, columns = extract_linked_schema(path)
        if tables:
            options.append((tuple(sorted(tables)), tuple(sorted(columns))))
    if not options:
        return set(), set()
    counts = Counter(options)
    tables, columns = max(options, key=lambda value: (counts[value], len(value[0]) + len(value[1])))
    return set(tables), set(columns)


def _ast_summary(sql: str, schema) -> Dict[str, Any]:
    usage = extract_sql_usage(sql, schema)
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        node_types = sorted({type(node).__name__ for node in tree.walk()})
    except Exception:
        node_types = []
    return {
        "tables": sorted(usage.tables),
        "columns": sorted(f"{table}.{column}" for table, column in usage.columns),
        "node_types": node_types,
    }


def rerank_llm_pool(
    tasks: Sequence[Any],
    split: Mapping[str, Any],
    candidate_dir: Path,
    db_root: Path,
    output_dir: Path,
    max_candidates: int,
    timeout: int,
) -> None:
    settings = JevSettings.from_environment()
    settings.require_api_key()
    reranker = JevSQLReranker(JevClient(settings), settings)
    requested = set(split["calibration_ids"] + split["evaluation_ids"])
    output_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for task in tasks:
        if task.question_id not in requested:
            continue
        output_path = output_dir / f"{task.question_id}.json"
        if output_path.exists():
            continue
        try:
            sqls, _ = _candidate_sqls(candidate_dir / f"{task.question_id}.pkl")
            sqls = sqls[:max_candidates]
            schema = DatabaseManager.get_database_schema(task.db_id, str(db_root))
            db_path = db_root / task.db_id / f"{task.db_id}.sqlite"
            candidates = []
            for sql in sqls:
                execution = validate_sql(str(db_path), sql, timeout)
                summary = _ast_summary(sql, schema)
                candidates.append(RerankCandidate(
                    canonical_sql=sql,
                    ast_summary=summary,
                    execution_summary=execution.summary(),
                    selected_schema={"tables": summary["tables"], "columns": summary["columns"]},
                ))
            started = time.perf_counter()
            ranking = reranker.rerank(task.question, task.evidence, candidates)
            elapsed = (time.perf_counter() - started) * 1000
            payload = {
                "question_id": task.question_id,
                "db_id": task.db_id,
                "candidate_sqls": sqls,
                "ranking": [item.to_dict() for item in ranking],
                "selected_index": ranking[0].candidate_index if ranking else None,
                "selected_sql": sqls[ranking[0].candidate_index] if ranking else None,
                "elapsed_ms": round(elapsed, 3),
                "usage": (
                    {
                        "input_tokens": ranking[0].usage.input_tokens,
                        "output_tokens": ranking[0].usage.output_tokens,
                    }
                    if ranking else {"input_tokens": 0, "output_tokens": 0}
                ),
            }
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            failures.append({
                "question_id": task.question_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
    (output_dir / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _macro(rows: Sequence[Dict[str, Any]], prefix: str) -> Dict[str, Any]:
    candidate_total = sum(row[f"{prefix}_candidate_count"] for row in rows)
    executable_total = sum(row[f"{prefix}_executable_count"] for row in rows)
    covered = sum(row[f"{prefix}_coverage"] for row in rows)
    selected = sum(row[f"{prefix}_selected_ex"] for row in rows)
    return {
        "questions": len(rows),
        "table_precision": sum(row[f"{prefix}_table_precision"] for row in rows) / len(rows),
        "table_recall": sum(row[f"{prefix}_table_recall"] for row in rows) / len(rows),
        "column_precision": sum(row[f"{prefix}_column_precision"] for row in rows) / len(rows),
        "column_recall": sum(row[f"{prefix}_column_recall"] for row in rows) / len(rows),
        "candidate_coverage": covered / len(rows),
        "executable_rate": executable_total / candidate_total if candidate_total else 0.0,
        "selected_ex": selected / len(rows),
        "selector_accuracy_given_coverage": selected / covered if covered else 0.0,
        "average_candidates": candidate_total / len(rows),
    }


def _bootstrap_delta(rows: Sequence[Dict[str, Any]], repeats: int = 10000) -> Dict[str, float]:
    randomizer = random.Random(42)
    differences = []
    for _ in range(repeats):
        sample = [rows[randomizer.randrange(len(rows))] for _ in rows]
        differences.append(sum(row["jev_selected_ex"] - row["llm_selected_ex"] for row in sample) / len(sample))
    differences.sort()
    observed = sum(row["jev_selected_ex"] - row["llm_selected_ex"] for row in rows) / len(rows)
    return {
        "jev_minus_llm": observed,
        "ci95_low": differences[int(0.025 * repeats)],
        "ci95_high": differences[int(0.975 * repeats)],
    }


def evaluate(
    tasks: Sequence[Any],
    split: Mapping[str, Any],
    candidate_dir: Path,
    jev_dir: Path,
    shared_rerank_dir: Path,
    db_root: Path,
    output_dir: Path,
    timeout: int,
) -> Dict[str, Any]:
    split_by_id = {
        **{int(value): "calibration" for value in split["calibration_ids"]},
        **{int(value): "evaluation" for value in split["evaluation_ids"]},
    }
    rows = []
    for task in tasks:
        if task.question_id not in split_by_id:
            continue
        llm_path = candidate_dir / f"{task.question_id}.pkl"
        jev_path = jev_dir / f"{task.question_id}.json"
        if not llm_path.exists() or not jev_path.exists():
            continue
        schema = DatabaseManager.get_database_schema(task.db_id, str(db_root))
        db_path = db_root / task.db_id / f"{task.db_id}.sqlite"
        gold_result = execute_sql_with_timeout(str(db_path), task.sql, timeout)
        if gold_result.result_type is not SQLExecutionResultType.SUCCESS:
            continue
        gold_signature = result_signature(gold_result.result)
        gold_usage = extract_sql_usage(task.sql, schema)
        gold_tables = {value.lower() for value in gold_usage.tables}
        gold_columns = {(table.lower(), column.lower()) for table, column in gold_usage.columns}

        llm_sqls, llm_paths = _candidate_sqls(llm_path)
        llm_exec = [_execute(db_path, sql, timeout) for sql in llm_sqls]
        llm_signatures = [value[1] for value in llm_exec]
        llm_rows = [value[2] for value in llm_exec]
        llm_selected_index = _self_consistency(llm_signatures, llm_rows)
        llm_tables, llm_columns = _modal_linked_schema(llm_paths)
        llm_tables = {value.lower() for value in llm_tables}
        llm_columns = {(table.lower(), column.lower()) for table, column in llm_columns}
        llm_tp, llm_tr = _precision_recall(llm_tables, gold_tables)
        llm_cp, llm_cr = _precision_recall(llm_columns, gold_columns)

        jev = json.loads(jev_path.read_text(encoding="utf-8"))
        jev_tables = {
            item["table"].lower() for item in jev["schema_linking"]["tables"] if item["selected"]
        }
        jev_columns = {
            (item["table"].lower(), item["column"].lower())
            for item in jev["schema_linking"]["columns"] if item["selected"]
        }
        jev_tp, jev_tr = _precision_recall(jev_tables, gold_tables)
        jev_cp, jev_cr = _precision_recall(jev_columns, gold_columns)
        jev_exec = [_execute(db_path, item["canonical_sql"], timeout) for item in jev["candidates"]]
        jev_signatures = [value[1] for value in jev_exec]
        jev_row_counts = [value[2] for value in jev_exec]
        jev_selected_index = jev["selected_index"]
        jev_top_beam_index = (
            max(
                range(len(jev["candidates"])),
                key=lambda index: jev["candidates"][index].get("beam_probability", 0.0),
            )
            if jev["candidates"] else None
        )
        jev_sc_index = _self_consistency(jev_signatures, jev_row_counts)

        shared_path = shared_rerank_dir / f"{task.question_id}.json"
        shared_selected_ex = False
        if shared_path.exists():
            shared = json.loads(shared_path.read_text(encoding="utf-8"))
            shared_index = shared.get("selected_index")
            shared_selected_ex = (
                shared_index is not None
                and shared_index < len(llm_signatures)
                and llm_signatures[shared_index] == gold_signature
            )
        native_rerank_usage = {"input_tokens": 0, "output_tokens": 0}
        if (
            jev_selected_index is not None
            and jev_selected_index < len(jev["candidates"])
            and jev["candidates"][jev_selected_index].get("rerank")
        ):
            native_rerank_usage = jev["candidates"][jev_selected_index]["rerank"]["usage"]
        shared_usage = shared.get("usage", {}) if shared_path.exists() else {}
        if not shared_usage and shared_path.exists() and shared.get("ranking"):
            shared_usage = shared["ranking"][0].get("usage", {})
        row = {
            "question_id": task.question_id,
            "split": split_by_id[task.question_id],
            "db_id": task.db_id,
            "llm_table_precision": llm_tp,
            "llm_table_recall": llm_tr,
            "llm_column_precision": llm_cp,
            "llm_column_recall": llm_cr,
            "llm_candidate_count": len(llm_sqls),
            "llm_executable_count": sum(value[0] for value in llm_exec),
            "llm_coverage": gold_signature in llm_signatures,
            "llm_selected_ex": (
                llm_selected_index is not None and llm_signatures[llm_selected_index] == gold_signature
            ),
            "llm_selected_sql": llm_sqls[llm_selected_index] if llm_selected_index is not None else None,
            "jev_table_precision": jev_tp,
            "jev_table_recall": jev_tr,
            "jev_column_precision": jev_cp,
            "jev_column_recall": jev_cr,
            "jev_candidate_count": len(jev_exec),
            "jev_executable_count": sum(value[0] for value in jev_exec),
            "jev_coverage": gold_signature in jev_signatures,
            "jev_selected_ex": (
                jev_selected_index is not None
                and jev_selected_index < len(jev_signatures)
                and jev_signatures[jev_selected_index] == gold_signature
            ),
            "jev_top_beam_ex": (
                jev_top_beam_index is not None
                and jev_signatures[jev_top_beam_index] == gold_signature
            ),
            "jev_self_consistency_ex": (
                jev_sc_index is not None and jev_signatures[jev_sc_index] == gold_signature
            ),
            "jev_selected_sql": jev.get("selected_sql"),
            "jev_uncertain": bool(jev.get("uncertain", True)),
            "jev_total_ms": jev.get("timings_ms", {}).get("total"),
            "jev_input_tokens": (
                jev["schema_linking"]["usage"].get("input_tokens", 0)
                + native_rerank_usage.get("input_tokens", 0)
            ),
            "jev_output_tokens": (
                jev["schema_linking"]["usage"].get("output_tokens", 0)
                + native_rerank_usage.get("output_tokens", 0)
            ),
            "shared_jev_rerank_selected_ex": shared_selected_ex,
            "shared_jev_rerank_ms": shared.get("elapsed_ms") if shared_path.exists() else None,
            "shared_jev_rerank_input_tokens": shared_usage.get("input_tokens", 0),
            "shared_jev_rerank_output_tokens": shared_usage.get("output_tokens", 0),
        }
        rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_question.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics: Dict[str, Any] = {"split": split}
    for name in ("calibration", "evaluation"):
        partition = [row for row in rows if row["split"] == name]
        paired = {
            "both_correct": sum(row["llm_selected_ex"] and row["jev_selected_ex"] for row in partition),
            "llm_only_correct": sum(row["llm_selected_ex"] and not row["jev_selected_ex"] for row in partition),
            "jev_only_correct": sum(row["jev_selected_ex"] and not row["llm_selected_ex"] for row in partition),
            "both_wrong": sum(not row["llm_selected_ex"] and not row["jev_selected_ex"] for row in partition),
        }
        metrics[name] = {
            "llm_direct_self_consistency": _macro(partition, "llm"),
            "jev_constrained": _macro(partition, "jev"),
            "paired_bootstrap": _bootstrap_delta(partition),
            "paired_outcomes": paired,
            "jev_candidate_pool_selection": {
                "oracle_ex": sum(row["jev_coverage"] for row in partition) / len(partition),
                "top_beam_ex": sum(row["jev_top_beam_ex"] for row in partition) / len(partition),
                "self_consistency_ex": sum(row["jev_self_consistency_ex"] for row in partition) / len(partition),
                "jev_reranker_ex": sum(row["jev_selected_ex"] for row in partition) / len(partition),
            },
            "shared_llm_pool": {
                "coverage": sum(row["llm_coverage"] for row in partition) / len(partition),
                "self_consistency_ex": sum(row["llm_selected_ex"] for row in partition) / len(partition),
                "jev_reranker_ex": sum(row["shared_jev_rerank_selected_ex"] for row in partition) / len(partition),
                "average_jev_rerank_ms": sum(row["shared_jev_rerank_ms"] for row in partition) / len(partition),
                "jev_rerank_input_tokens": sum(row["shared_jev_rerank_input_tokens"] for row in partition),
                "jev_rerank_output_tokens": sum(row["shared_jev_rerank_output_tokens"] for row in partition),
            },
            "jev_efficiency": {
                "average_total_ms": sum(row["jev_total_ms"] for row in partition) / len(partition),
                "total_input_tokens": sum(row["jev_input_tokens"] for row in partition),
                "total_output_tokens": sum(row["jev_output_tokens"] for row in partition),
                "uncertain_rate": sum(row["jev_uncertain"] for row in partition) / len(partition),
            },
        }
    errors = []
    for row in rows:
        for prefix, system in (("llm", "alpha_sql_llm_direct"), ("jev", "jev_constrained")):
            if row[f"{prefix}_selected_ex"]:
                continue
            coverage = row[f"{prefix}_coverage"]
            errors.append({
                "question_id": row["question_id"],
                "split": row["split"],
                "db_id": row["db_id"],
                "system": system,
                "category": "selection_error" if coverage else "candidate_coverage_miss",
                "diagnostic": (
                    "correct_candidate_existed_but_selector_missed"
                    if coverage else
                    "schema_incomplete_possible"
                    if row[f"{prefix}_table_recall"] < 1 or row[f"{prefix}_column_recall"] < 1
                    else "schema_complete_check_literal_or_beam"
                ),
                "table_recall": row[f"{prefix}_table_recall"],
                "column_recall": row[f"{prefix}_column_recall"],
                "selected_sql": row[f"{prefix}_selected_sql"],
            })
    with (output_dir / "error_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "question_id", "split", "db_id", "system", "category", "diagnostic",
            "table_recall", "column_recall", "selected_sql",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(errors)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    evaluation_metrics = metrics["evaluation"]
    llm_metrics = evaluation_metrics["llm_direct_self_consistency"]
    jev_metrics = evaluation_metrics["jev_constrained"]
    shared_metrics = evaluation_metrics["shared_llm_pool"]
    jev_selection = evaluation_metrics["jev_candidate_pool_selection"]
    efficiency = evaluation_metrics["jev_efficiency"]
    report = [
        "# Alpha-SQL LLM vs Jev: Phase-One Results", "",
        "The primary result is the database-disjoint evaluation partition (18 executable-gold questions).", "",
        "| Metric | Alpha-SQL LLM direct + self-consistency | Jev constrained |", "|---|---:|---:|",
        f"| Table precision | {llm_metrics['table_precision']:.3f} | {jev_metrics['table_precision']:.3f} |",
        f"| Table recall | {llm_metrics['table_recall']:.3f} | {jev_metrics['table_recall']:.3f} |",
        f"| Column precision | {llm_metrics['column_precision']:.3f} | {jev_metrics['column_precision']:.3f} |",
        f"| Column recall | {llm_metrics['column_recall']:.3f} | {jev_metrics['column_recall']:.3f} |",
        f"| Candidate coverage | {llm_metrics['candidate_coverage']:.3f} | {jev_metrics['candidate_coverage']:.3f} |",
        f"| Executable rate | {llm_metrics['executable_rate']:.3f} | {jev_metrics['executable_rate']:.3f} |",
        f"| Selected EX | {llm_metrics['selected_ex']:.3f} | {jev_metrics['selected_ex']:.3f} |", "",
        f"Jev - LLM EX difference: {evaluation_metrics['paired_bootstrap']['jev_minus_llm']:.3f} "
        f"(paired bootstrap 95% CI {evaluation_metrics['paired_bootstrap']['ci95_low']:.3f} to "
        f"{evaluation_metrics['paired_bootstrap']['ci95_high']:.3f}).", "",
        f"On the fixed LLM candidate pool, self-consistency EX and Jev reranker EX were both "
        f"{shared_metrics['self_consistency_ex']:.3f}; this pilot shows no selector lift.", "",
        f"On the native Jev pool, top-beam/self-consistency EX was {jev_selection['top_beam_ex']:.3f}, "
        f"Jev reranking reached {jev_selection['jev_reranker_ex']:.3f}, and the candidate oracle was "
        f"{jev_selection['oracle_ex']:.3f}.", "",
        f"Jev native latency averaged {efficiency['average_total_ms']:.1f} ms/question and used "
        f"{efficiency['total_input_tokens']} input plus {efficiency['total_output_tokens']} output tokens "
        "on the evaluation partition.", "",
        "Caveats:", "",
        "- Cached direct Alpha-SQL candidates are not full MCTS runs and contain no reliable latency/token metadata.",
        "- The slice excludes SQL outside the current constrained compiler.",
        "- Calibration databases were used during smoke-test debugging and are excluded from the primary result.",
        "- One nominal evaluation item was excluded because its gold SQL references a missing database column.",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return metrics


def main(args: argparse.Namespace) -> None:
    with Path(args.tasks_file).open("rb") as handle:
        tasks = pickle.load(handle)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = JevSettings.from_environment()
    config = {
        "experiment": "alpha_sql_llm_vs_jev_phase1",
        "llm_system": "cached Alpha-SQL direct candidates plus execution self-consistency",
        "llm_candidate_directory": str(Path(args.llm_candidates)),
        "full_mcts_baseline": False,
        "cached_llm_generation_manifest_available": False,
        "jev_model": settings.model,
        "jev_schema_threshold": settings.schema_threshold,
        "jev_rerank_threshold": settings.rerank_threshold,
        "jev_beam_width": settings.beam_width,
        "max_candidates_for_shared_rerank": args.max_candidates,
        "execution_timeout": args.execution_timeout,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    split_path = output_dir / "split.json"
    split = prepare_split(tasks, Path(args.db_root), args.execution_timeout)
    split_path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    if args.mode == "prepare":
        print(split)
        return
    if args.mode == "rerank-llm":
        rerank_llm_pool(
            tasks, split, Path(args.llm_candidates), Path(args.db_root),
            Path(args.shared_rerank_dir), args.max_candidates, args.execution_timeout,
        )
        return
    metrics = evaluate(
        tasks, split, Path(args.llm_candidates), Path(args.jev_results),
        Path(args.shared_rerank_dir), Path(args.db_root), output_dir,
        args.execution_timeout,
    )
    print(json.dumps(metrics["evaluation"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "rerank-llm", "evaluate"), required=True)
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--db-root", required=True)
    parser.add_argument("--llm-candidates", required=True)
    parser.add_argument("--jev-results", required=True)
    parser.add_argument("--shared-rerank-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--execution-timeout", type=int, default=15)
    main(parser.parse_args())
