"""Small SQL-content verifier with pointwise and within-question pairwise loss.

This is a sparse, inspectable baseline. It consumes only inference-visible text,
schema and execution summaries; reference SQL and JEV scores are never features.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
import sqlglot
from sqlglot import exp


_WORD = re.compile(r"[a-z]+(?:_[a-z0-9]+)*|\d+(?:\.\d+)?", re.I)
_SQL_WORDS = ("select", "from", "join", "where", "having", "group", "order",
              "limit", "distinct", "count", "sum", "avg", "max", "min",
              "like", "between", "exists", "union", "intersect", "except")
_QUESTION_HINTS = {"count": ("how many", "number of", "count"),
                   "max": ("most", "highest", "maximum", "largest"),
                   "min": ("least", "lowest", "minimum", "smallest"),
                   "avg": ("average", "mean"),
                   "sum": ("total", "sum"),
                   "order": ("top", "bottom", "rank", "sorted", "order"),
                   "distinct": ("different", "distinct", "unique")}


def _tokens(value: Any) -> set[str]:
    words = set()
    for match in _WORD.findall(str(value).lower()):
        words.add(match)
        words.update(piece for piece in match.split("_") if len(piece) > 1)
    return words


def _schema_text(schema: Mapping[str, Any], table_names: set[str]) -> str:
    tables = schema.get("tables", {})
    selected = [value for name, value in tables.items()
                if name.lower() in table_names]
    if not selected:
        selected = list(tables.values())[:8]
    parts = []
    for table in selected[:8]:
        for name, column in list(table.get("columns", {}).items())[:80]:
            parts.extend((name, str(column.get("description") or "")))
    return " ".join(parts)


def content_features(question: str, evidence: str, schema: Mapping[str, Any],
                     sql: str, execution: Mapping[str, Any]) -> dict[str, float]:
    """Bounded lexical, SQL-structure, schema-alignment and result features."""
    q_text = f"{question} {evidence}".lower()
    q = _tokens(q_text)
    sql_tokens = _tokens(sql)
    try:
        ast = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        ast = None
    tables = {node.name.lower() for node in ast.find_all(exp.Table)} if ast else set()
    columns = {node.name.lower() for node in ast.find_all(exp.Column)} if ast else set()
    identifiers = _tokens(" ".join(tables | columns))
    schema_tokens = _tokens(_schema_text(schema, tables))
    execution_tokens = _tokens(" ".join(str(value)
                              for row in execution.get("sample_rows", [])[:3]
                              for value in row[:5]))
    features: dict[str, float] = {"bias": 1.0}
    for word in _SQL_WORDS:
        if word in sql_tokens:
            features[f"sql:{word}"] = 1.0
    for word in sorted(sql_tokens - set(_SQL_WORDS))[:80]:
        if len(word) > 2:
            features[f"sql_term:{word}"] = 1.0
    for word in sorted(q & identifiers)[:40]:
        features[f"aligned_identifier:{word}"] = 1.0
    for word in sorted(q & schema_tokens)[:40]:
        features[f"aligned_schema:{word}"] = 1.0
    for word in sorted(q & execution_tokens)[:20]:
        features[f"aligned_result:{word}"] = 1.0
    for name, phrases in _QUESTION_HINTS.items():
        asked = any(phrase in q_text for phrase in phrases)
        present = name in sql_tokens
        features[f"hint:{name}:{int(asked)}:{int(present)}"] = 1.0
    numeric_q = {word for word in q if word[0].isdigit()}
    features.update({
        "identifier_overlap": len(q & identifiers) / max(1, len(q)),
        "schema_overlap": len(q & schema_tokens) / max(1, len(q)),
        "sql_overlap": len(q & sql_tokens) / max(1, len(q)),
        "result_overlap": len(q & execution_tokens) / max(1, len(q)),
        "number_coverage": len(numeric_q & sql_tokens) / max(1, len(numeric_q)),
        "number_missing": float(bool(numeric_q - sql_tokens)),
        "n_tables": min(len(tables), 8) / 8,
        "n_columns": min(len(columns), 20) / 20,
        "n_joins": min(sum(1 for _ in ast.find_all(exp.Join)), 8) / 8 if ast else 0.0,
        "sql_length": min(len(sql), 1200) / 1200,
        "exec_valid": float(bool(execution.get("valid"))),
        "exec_empty": float(execution.get("row_count") == 0),
        "exec_rows_log": math.log1p(max(0, int(execution.get("row_count") or 0))) / 12,
        "exec_columns": min(len(execution.get("columns") or []), 20) / 20,
        "exec_has_null": float(any(value is None
                                   for row in execution.get("sample_rows", [])[:3]
                                   for value in row[:5])),
    })
    return features


def candidate_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Use each question's final pool once; repeated round snapshots are excluded."""
    records = []
    for row in rows:
        state = row["rounds"][-1]
        seen = set()
        for item in state["candidates"]:
            if not item["execution_rechecked_valid"]:
                continue
            canonical = item.get("canonical_sql") or item["sql"]
            if canonical in seen:
                continue
            seen.add(canonical)
            records.append({"question_id": str(row["question_id"]),
                            "db_id": row["db_id"], "sql": item["sql"],
                            "label": int(item["execution_label"]),
                            "near_miss": bool(item.get("near_miss")),
                            "features": content_features(row["question"], row["evidence"],
                                                         row["full_schema"], item["sql"],
                                                         item["execution"])})
    return records


def _matrix(records: Sequence[Mapping[str, Any]], vocab: Mapping[str, int]) -> sparse.csr_matrix:
    r, c, v = [], [], []
    for index, item in enumerate(records):
        for name, value in item["features"].items():
            if name in vocab and value:
                r.append(index); c.append(vocab[name]); v.append(float(value))
    return sparse.csr_matrix((v, (r, c)), shape=(len(records), len(vocab)), dtype=np.float64)


def _pairs(records: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped = defaultdict(lambda: {0: [], 1: []})
    for i, item in enumerate(records):
        grouped[item["question_id"]][item["label"]].append(i)
    positive, negative, weights = [], [], []
    for group in grouped.values():
        ps, ns = group[1], group[0]
        if not ps or not ns:
            continue
        for p in ps:
            for n in ns:
                positive.append(p); negative.append(n)
                weights.append((2.0 if records[n]["near_miss"] else 1.0)
                               / (len(ps) * len(ns)))
    return np.asarray(positive, dtype=int), np.asarray(negative, dtype=int), np.asarray(weights)


class ContentVerifier:
    def __init__(self, vocab: Mapping[str, int], weights: Sequence[float]):
        self.vocab = dict(vocab)
        self.weights = np.asarray(weights, dtype=np.float64)

    @classmethod
    def fit(cls, records: Sequence[Mapping[str, Any]], *, pair_weight: float = 1.0,
            regularization: float = 0.1, max_iterations: int = 300) -> tuple["ContentVerifier", dict[str, Any]]:
        if not 0 <= pair_weight or regularization <= 0:
            raise ValueError("Invalid optimization weights")
        counts = Counter(name for item in records for name in item["features"])
        vocab = {name: i for i, name in enumerate(sorted(name for name, count in counts.items()
                                                        if count >= 2 or name == "bias"))}
        x = _matrix(records, vocab)
        y = np.asarray([item["label"] for item in records], dtype=float)
        if len(np.unique(y)) < 2:
            raise ValueError("Both positive and negative candidates are required")
        group_sizes = Counter(item["question_id"] for item in records)
        point_weights = np.asarray([1 / group_sizes[item["question_id"]]
                                    for item in records])
        point_weights /= point_weights.sum()
        p, n, pair_weights = _pairs(records)
        if not len(p):
            raise ValueError("Need correct and incorrect SQL in the same question")
        pair_weights /= pair_weights.sum()
        difference = x[p] - x[n]

        def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
            logits = x @ w
            signed = 2 * y - 1
            point_loss = float(point_weights @ np.logaddexp(0, -signed * logits))
            point_gradient = x.T @ (point_weights * (expit(logits) - y))
            margins = difference @ w
            rank_loss = float(pair_weights @ np.logaddexp(0, -margins))
            rank_gradient = difference.T @ (-pair_weights * expit(-margins))
            penalty_weights = w.copy()
            penalty_weights[vocab["bias"]] = 0
            penalty = regularization * float(penalty_weights @ penalty_weights) / 2
            gradient = point_gradient + pair_weight * rank_gradient
            gradient += regularization * penalty_weights
            return point_loss + pair_weight * rank_loss + penalty, np.asarray(gradient).ravel()

        result = minimize(objective, np.zeros(len(vocab)), jac=True, method="L-BFGS-B",
                          options={"maxiter": max_iterations})
        model = cls(vocab, result.x)
        return model, {"candidates": len(records), "questions": len(group_sizes),
                       "positive": int(y.sum()), "negative": int(len(y) - y.sum()),
                       "pairs": len(p), "features": len(vocab),
                       "objective": float(result.fun), "iterations": int(result.nit),
                       "optimizer_success": bool(result.success)}

    def predict(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return expit(_matrix(records, self.vocab) @ self.weights)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"version": 1, "vocab": self.vocab,
                                          "weights": self.weights.tolist(),
                                          "implementation_sha256": hashlib.sha256(
                                              Path(__file__).read_bytes()).hexdigest()},
                                         ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ContentVerifier":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["version"] != 1:
            raise ValueError("Unsupported verifier model")
        if data.get("implementation_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise ValueError("Verifier feature code differs from the frozen model")
        return cls(data["vocab"], data["weights"])
