"""Inference-visible execution and structure evidence for fixed SQL pools.

The reference SQL is never read here. Full result sets are hashed, not stored in
the selector trace, and execution agreement is treated as a feature, not a label.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import sqlglot
from sqlglot import exp

from alphasql.algorithm.compiler.validation import validate_sql
from alphasql.algorithm.selection.schema_signal import result_signature


_ORDER_WORDS = re.compile(r"\b(order|sorted|rank|top|bottom|first|last|highest|lowest)\b", re.I)
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")


def pool_identity(row: Mapping[str, Any]) -> str:
    state = row["rounds"][0]
    data = {"db_id": row["db_id"], "question_id": row["question_id"],
            "sql": [item["sql"] for item in state["candidates"]]}
    return hashlib.sha256(json.dumps(data, sort_keys=True,
                                      ensure_ascii=False).encode()).hexdigest()


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, default=str,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _structure(sql: str, question: str, evidence: str) -> dict[str, float]:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return {}
    text = f"{question} {evidence}".lower()
    literal_numbers = set(_NUMBER.findall(sql))
    asked_numbers = set(_NUMBER.findall(text))
    joins = list(tree.find_all(exp.Join))
    tables = list(tree.find_all(exp.Table))
    columns = list(tree.find_all(exp.Column))
    has_order = tree.find(exp.Order) is not None
    has_limit = tree.find(exp.Limit) is not None
    has_group = tree.find(exp.Group) is not None
    has_having = tree.find(exp.Having) is not None
    has_where = tree.find(exp.Where) is not None
    return {
        "ast_tables": min(len(tables), 12) / 12,
        "ast_columns": min(len(columns), 40) / 40,
        "ast_joins": min(len(joins), 12) / 12,
        "ast_where": float(has_where),
        "ast_having": float(has_having),
        "ast_group": float(has_group),
        "ast_order": float(has_order),
        "ast_limit": float(has_limit),
        "ast_distinct": float(tree.find(exp.Distinct) is not None),
        "ast_count": float(tree.find(exp.Count) is not None),
        "ast_avg": float(tree.find(exp.Avg) is not None),
        "ast_sum": float(tree.find(exp.Sum) is not None),
        "ast_subquery": float(tree.find(exp.Subquery) is not None),
        "question_order_mismatch": float(bool(_ORDER_WORDS.search(question)) and not has_order),
        "question_number_missing": float(bool(asked_numbers - literal_numbers)),
        "question_number_coverage": len(asked_numbers & literal_numbers) / max(1, len(asked_numbers)),
        "question_count_mismatch": float(bool(re.search(r"\b(how many|number of|count)\b", text))
                                         and tree.find(exp.Count) is None),
        "question_avg_mismatch": float(bool(re.search(r"\b(average|mean)\b", text))
                                       and tree.find(exp.Avg) is None),
    }


def enrich_pool(row: Mapping[str, Any], db_root: str | Path, timeout: int = 60) -> dict[str, Any]:
    """Re-execute candidates once and derive result clusters without gold SQL."""
    db_id = str(row["db_id"])
    db_path = Path(db_root) / db_id / f"{db_id}.sqlite"
    if not db_path.is_file():
        raise FileNotFoundError(db_path)
    state = row["rounds"][0]
    candidates = state["candidates"]
    ordered = bool(_ORDER_WORDS.search(row["question"]))
    items: list[dict[str, Any]] = []
    clusters: dict[str, list[int]] = defaultdict(list)
    for i, candidate in enumerate(candidates):
        features = _structure(candidate["sql"], row["question"], row.get("evidence") or "")
        result = validate_sql(str(db_path), candidate["sql"], timeout)
        if candidate["execution"]["valid"] != result.valid:
            raise ValueError(f"Execution validity changed for {db_id}/{row['question_id']}/{i}")
        record: dict[str, Any] = {"index": i, "valid": result.valid,
                                  "route": candidate.get("route") or "unknown",
                                  "features": features}
        if result.valid:
            # Include column count so two different empty projections do not agree.
            unordered = _digest((len(result.columns), result_signature(result.rows)))
            ordered_hash = _digest((len(result.columns), result.rows))
            chosen_hash = ordered_hash if ordered else unordered
            record.update(result_hash=chosen_hash,
                          unordered_result_hash=unordered,
                          ordered_result_hash=ordered_hash,
                          row_count=len(result.rows), column_count=len(result.columns))
            features.update({
                "exec_empty": float(not result.rows),
                "exec_rows_log": math.log1p(len(result.rows)) / 12,
                "exec_columns": min(len(result.columns), 20) / 20,
                "exec_sample_has_null": float(any(value is None
                    for sample in result.rows[:3] for value in sample[:5])),
            })
            clusters[chosen_hash].append(i)
        items.append(record)
    valid_count = sum(item["valid"] for item in items)
    route_count = len({item["route"] for item in items if item["valid"]})
    cluster_sizes = sorted((len(indices) for indices in clusters.values()), reverse=True)
    entropy = (-sum((n / valid_count) * math.log(n / valid_count)
                    for n in cluster_sizes) / math.log(max(2, valid_count))
               if valid_count else 0.0)
    for indices in clusters.values():
        routes = len({items[i]["route"] for i in indices})
        raw_scores = [float(candidates[i]["candidate_raw"]) for i in indices]
        for i in indices:
            items[i]["features"].update({
                "cluster_fraction": len(indices) / valid_count,
                "cluster_route_fraction": routes / max(1, route_count),
                "cluster_raw_mean": sum(raw_scores) / len(raw_scores),
                "cluster_raw_max": max(raw_scores),
                "cluster_entropy": entropy,
                "cluster_margin": (cluster_sizes[0] - (cluster_sizes[1] if len(cluster_sizes) > 1 else 0))
                                  / max(1, valid_count),
                "cluster_count": min(len(clusters), 10) / 10,
            })
    return {"db_id": db_id, "question_id": row["question_id"],
            "pool_sha256": pool_identity(row), "order_sensitive_heuristic": ordered,
            "candidates": items,
            "clusters": [{"result_hash": key, "indices": indices,
                          "routes": sorted({items[i]["route"] for i in indices})}
                         for key, indices in clusters.items()],
            "cluster_sizes": cluster_sizes,
            "label_free": True}
