"""Lightweight features and models for schema-aware SQL candidate reranking.

This module deliberately does not change schema linking or candidate generation.  It
turns the existing hard schema selection on each MCTS path into candidate-level
signals that can be evaluated with small, auditable models.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from sqlglot import exp, parse_one

from alphasql.database.schema import DatabaseSchema


FEATURE_GROUPS = {
    "sql": "sql_",
    "schema": "schema_",
    "execution": "exec_",
    "mcts": "mcts_",
}


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _schema_lookup(database_schema: DatabaseSchema):
    tables = {name.lower(): name for name in database_schema.tables}
    columns = {
        table.lower(): {column.lower(): column for column in table_schema.columns}
        for table, table_schema in database_schema.tables.items()
    }
    return tables, columns


@dataclass
class SQLUsage:
    tables: Set[str]
    columns: Set[Tuple[str, str]]
    unresolved_columns: int
    ast: Optional[exp.Expression]


def extract_sql_usage(sql: str, database_schema: DatabaseSchema) -> SQLUsage:
    """Extract canonical table/column names, resolving aliases where possible."""
    try:
        ast = parse_one(sql, dialect="sqlite")
    except Exception:
        return SQLUsage(set(), set(), 0, None)

    table_names, schema_columns = _schema_lookup(database_schema)
    aliases: Dict[str, str] = {}
    used_tables: Set[str] = set()
    for table in ast.find_all(exp.Table):
        actual = table_names.get(table.name.lower(), table.name.lower())
        actual_lower = actual.lower()
        used_tables.add(actual_lower)
        aliases[table.name.lower()] = actual_lower
        aliases[table.alias_or_name.lower()] = actual_lower

    used_columns: Set[Tuple[str, str]] = set()
    unresolved = 0
    for column in ast.find_all(exp.Column):
        column_lower = column.name.lower()
        if column_lower == "*":
            continue
        if column.table:
            table_lower = aliases.get(column.table.lower(), column.table.lower())
            if column_lower in schema_columns.get(table_lower, {}):
                used_columns.add((table_lower, column_lower))
            else:
                unresolved += 1
            continue
        matches = [
            table for table in used_tables
            if column_lower in schema_columns.get(table, {})
        ]
        if len(matches) == 1:
            used_columns.add((matches[0], column_lower))
        else:
            unresolved += 1
    return SQLUsage(used_tables, used_columns, unresolved, ast)


def generate_sql_mutations(sql: str, limit: int = 8) -> List[str]:
    """Generate nearby structural SQL variants for supervised hard negatives.

    These are proposals only: callers must execute them and discard mutations
    that are invalid or still equivalent to the gold result.
    """
    try:
        original = parse_one(sql, dialect="sqlite")
    except Exception:
        return []
    mutations = []

    def add(tree):
        candidate = tree.sql(dialect="sqlite")
        if candidate != sql and candidate not in mutations:
            mutations.append(candidate)

    for argument in ("where", "having", "order", "limit", "group"):
        if any(select.args.get(argument) is not None for select in original.find_all(exp.Select)):
            tree = original.copy()
            targets = [item for item in tree.find_all(exp.Select) if item.args.get(argument) is not None]
            if targets:
                targets[0].set(argument, None)
                add(tree)
    comparison_swaps = {
        exp.EQ: exp.NEQ, exp.NEQ: exp.EQ,
        exp.GT: exp.LTE, exp.GTE: exp.LT,
        exp.LT: exp.GTE, exp.LTE: exp.GT,
    }
    for comparison_type, replacement_type in comparison_swaps.items():
        count = len(list(original.find_all(comparison_type)))
        for index in range(count):
            tree = original.copy()
            target = list(tree.find_all(comparison_type))[index]
            target.replace(replacement_type(this=target.this.copy(), expression=target.expression.copy()))
            add(tree)
            if len(mutations) >= limit:
                return mutations[:limit]
    for distinct in list(original.find_all(exp.Distinct)):
        tree = original.copy()
        target = next(tree.find_all(exp.Distinct), None)
        if target is not None and target.parent is not None:
            target.parent.set("distinct", None)
            add(tree)
            break
    return mutations[:limit]


def _selected_column_names(value: Any) -> Iterable[str]:
    """Normalize linker values from runtime objects and serialized mappings."""
    columns = getattr(value, "columns", None)
    if isinstance(columns, Mapping):
        return columns.keys()
    if isinstance(value, Mapping):
        nested = value.get("columns")
        if isinstance(nested, Mapping):
            return nested.keys()
        return value.keys()
    if isinstance(value, (str, bytes)):
        return (value,)
    return value or ()


def extract_linked_schema(path: Sequence[Any]) -> Tuple[Set[str], Set[Tuple[str, str]]]:
    """Read the last hard schema selection available on an existing MCTS path."""
    selected: Mapping[str, Sequence[str]] = {}
    for node in reversed(path):
        value = getattr(node, "selected_schema_dict", None)
        if value:
            selected = value
            break
    tables = {str(table).lower() for table in selected}
    columns = {
        (str(table).lower(), str(column).lower())
        for table, value in selected.items()
        for column in _selected_column_names(value)
    }
    return tables, columns


def _foreign_key_edges(database_schema: DatabaseSchema) -> Set[Tuple[Tuple[str, str], Tuple[str, str]]]:
    edges = set()
    for table_name, table in database_schema.tables.items():
        for column_name, column in table.columns.items():
            source = (table_name.lower(), column_name.lower())
            for target_table, target_column in column.foreign_keys:
                target = (target_table.lower(), target_column.lower())
                edges.add(tuple(sorted((source, target))))
    return edges


def _connected_components(tables: Set[str], fk_edges) -> int:
    if not tables:
        return 0
    graph = defaultdict(set)
    for left, right in fk_edges:
        if left[0] in tables and right[0] in tables:
            graph[left[0]].add(right[0])
            graph[right[0]].add(left[0])
    remaining = set(tables)
    components = 0
    while remaining:
        components += 1
        queue = deque([remaining.pop()])
        while queue:
            for neighbor in graph[queue.popleft()]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
    return components


def extract_candidate_features(
    sql: str,
    database_schema: DatabaseSchema,
    linked_tables: Set[str],
    linked_columns: Set[Tuple[str, str]],
    execution: Optional[Mapping[str, Any]] = None,
    mcts: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Create interpretable SQL, schema, execution, and optional MCTS features."""
    usage = extract_sql_usage(sql, database_schema)
    ast = usage.ast
    nodes = list(ast.walk()) if ast is not None else []
    joins = list(ast.find_all(exp.Join)) if ast is not None else []
    predicates = [] if ast is None else list(ast.find_all(exp.Predicate))
    aggregates = [] if ast is None else list(ast.find_all(exp.AggFunc))
    subqueries = [] if ast is None else list(ast.find_all(exp.Subquery))

    used_table_overlap = usage.tables & linked_tables
    used_column_overlap = usage.columns & linked_columns
    fk_edges = _foreign_key_edges(database_schema)
    components = _connected_components(usage.tables, fk_edges)
    join_equalities = [] if ast is None else [
        item for item in ast.find_all(exp.EQ)
        if isinstance(item.this, exp.Column) and isinstance(item.expression, exp.Column)
    ]
    fk_join_count = 0
    aliases = {}
    if ast is not None:
        for table in ast.find_all(exp.Table):
            aliases[table.alias_or_name.lower()] = table.name.lower()
    for equality in join_equalities:
        left = (aliases.get(equality.this.table.lower(), equality.this.table.lower()), equality.this.name.lower())
        right = (aliases.get(equality.expression.table.lower(), equality.expression.table.lower()), equality.expression.name.lower())
        if tuple(sorted((left, right))) in fk_edges:
            fk_join_count += 1

    features = {
        "sql_parse_success": float(ast is not None),
        "sql_char_count_log": math.log1p(len(sql or "")),
        "sql_ast_node_count_log": math.log1p(len(nodes)),
        "sql_table_count": float(len(usage.tables)),
        "sql_column_count": float(len(usage.columns)),
        "sql_unresolved_column_count": float(usage.unresolved_columns),
        "sql_join_count": float(len(joins)),
        "sql_join_equality_count": float(len(join_equalities)),
        "sql_predicate_count": float(len(predicates)),
        "sql_aggregate_count": float(len(aggregates)),
        "sql_subquery_count": float(len(subqueries)),
        "sql_has_where": float(ast is not None and ast.find(exp.Where) is not None),
        "sql_has_group": float(ast is not None and ast.find(exp.Group) is not None),
        "sql_has_having": float(ast is not None and ast.find(exp.Having) is not None),
        "sql_has_order": float(ast is not None and ast.find(exp.Order) is not None),
        "sql_has_limit": float(ast is not None and ast.find(exp.Limit) is not None),
        "sql_has_distinct": float(ast is not None and ast.find(exp.Distinct) is not None),
        "schema_linked_table_count": float(len(linked_tables)),
        "schema_linked_column_count": float(len(linked_columns)),
        "schema_table_coverage": _ratio(len(used_table_overlap), len(usage.tables)),
        "schema_column_coverage": _ratio(len(used_column_overlap), len(usage.columns)),
        "schema_table_precision": _ratio(len(used_table_overlap), len(linked_tables)),
        "schema_column_precision": _ratio(len(used_column_overlap), len(linked_columns)),
        "schema_unlinked_used_table_count": float(len(usage.tables - linked_tables)),
        "schema_unlinked_used_column_count": float(len(usage.columns - linked_columns)),
        "schema_unused_linked_table_count": float(len(linked_tables - usage.tables)),
        "schema_unused_linked_column_count": float(len(linked_columns - usage.columns)),
        "schema_fk_component_count": float(components),
        "schema_fk_connected": float(len(usage.tables) <= 1 or components == 1),
        "schema_fk_join_count": float(fk_join_count),
        "schema_non_fk_join_count": float(max(0, len(join_equalities) - fk_join_count)),
        "schema_fk_join_ratio": _ratio(fk_join_count, len(join_equalities)),
    }

    execution = execution or {}
    row_count = float(execution.get("row_count", 0) or 0)
    features.update({
        "exec_success": float(bool(execution.get("success", False))),
        "exec_nonempty": float(row_count > 0),
        "exec_row_count_log": math.log1p(max(0.0, row_count)),
        "exec_column_count": float(execution.get("column_count", 0) or 0),
        "exec_null_ratio": float(execution.get("null_ratio", 0.0) or 0.0),
        "exec_duplicate_ratio": float(execution.get("duplicate_ratio", 0.0) or 0.0),
        "exec_result_group_size": float(execution.get("group_size", 0) or 0),
        "exec_result_group_share": float(execution.get("group_share", 0.0) or 0.0),
    })

    mcts = mcts or {}
    features.update({
        "mcts_consistency_score": float(mcts.get("consistency_score", 0.0) or 0.0),
        "mcts_path_length": float(mcts.get("path_length", 0) or 0),
        "mcts_depth": float(mcts.get("depth", 0) or 0),
        "mcts_has_schema_selection": float(bool(mcts.get("path_has_schema_selection", linked_tables))),
    })
    return features


def result_signature(rows: Optional[Sequence[Sequence[Any]]]) -> Optional[Tuple[Any, ...]]:
    """Order-insensitive multiset signature; unlike frozenset it preserves duplicates."""
    if rows is None:
        return None
    normalized = [tuple(_normalize_value(value) for value in row) for row in rows]
    return tuple(sorted(Counter(normalized).items(), key=repr))


def _normalize_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, (str, int, bytes, type(None))):
        return value
    return repr(value)


class Standardizer:
    def fit(self, values: np.ndarray):
        self.mean = values.mean(axis=0)
        self.scale = values.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale


class LogisticReranker:
    """Dependency-free, class-balanced logistic regression trained with Adam."""
    def __init__(self, epochs: int = 1000, learning_rate: float = 0.03, l2: float = 1e-3, seed: int = 42):
        self.epochs, self.learning_rate, self.l2, self.seed = epochs, learning_rate, l2, seed

    def fit(self, values: np.ndarray, labels: np.ndarray):
        values = np.asarray(values, dtype=float)
        labels = np.asarray(labels, dtype=float)
        if len(set(labels.tolist())) < 2:
            raise ValueError("Training candidates must include both correct and incorrect SQL labels")
        self.scaler = Standardizer().fit(values)
        x = self.scaler.transform(values)
        x = np.column_stack([x, np.ones(len(x))])
        self.weights = np.zeros(x.shape[1], dtype=float)
        positive_weight = len(labels) / (2.0 * labels.sum())
        negative_weight = len(labels) / (2.0 * (len(labels) - labels.sum()))
        sample_weight = np.where(labels == 1, positive_weight, negative_weight)
        m = np.zeros_like(self.weights)
        v = np.zeros_like(self.weights)
        for step in range(1, self.epochs + 1):
            prediction = 1.0 / (1.0 + np.exp(-np.clip(x @ self.weights, -30, 30)))
            gradient = x.T @ ((prediction - labels) * sample_weight) / len(labels)
            gradient[:-1] += self.l2 * self.weights[:-1]
            m = 0.9 * m + 0.1 * gradient
            v = 0.999 * v + 0.001 * gradient * gradient
            self.weights -= self.learning_rate * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
        return self

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        x = self.scaler.transform(np.asarray(values, dtype=float))
        x = np.column_stack([x, np.ones(len(x))])
        return 1.0 / (1.0 + np.exp(-np.clip(x @ self.weights, -30, 30)))


def vectorize(records: Sequence[Mapping[str, Any]], groups: Iterable[str]):
    prefixes = tuple(FEATURE_GROUPS[group] for group in groups)
    names = sorted({name for record in records for name in record["features"] if name.startswith(prefixes)})
    values = np.asarray([[float(record["features"].get(name, 0.0)) for name in names] for record in records])
    labels = np.asarray([int(record["label"]) for record in records])
    return values, labels, names


def selection_metrics(records: Sequence[Mapping[str, Any]], scores: Sequence[float]) -> Dict[str, float]:
    grouped = defaultdict(list)
    for record, score in zip(records, scores):
        grouped[str(record["question_id"])].append((record, float(score)))
    questions = len(grouped)
    oracle = model = baseline = 0
    for candidates in grouped.values():
        oracle += int(any(item[0]["label"] for item in candidates))
        model += int(max(candidates, key=lambda item: (item[1], -item[0].get("candidate_index", 0)))[0]["label"])
        baseline += int(max(candidates, key=lambda item: (
            item[0]["features"].get("exec_result_group_size", 0),
            -item[0].get("candidate_index", 0),
        ))[0]["label"])
    oracle_ex = _ratio(oracle, questions)
    baseline_ex = _ratio(baseline, questions)
    model_ex = _ratio(model, questions)
    denominator = oracle_ex - baseline_ex
    return {
        "question_count": questions,
        "oracle_ex": oracle_ex,
        "baseline_ex": baseline_ex,
        "model_ex": model_ex,
        "gap_closure": _ratio(model_ex - baseline_ex, denominator) if denominator > 0 else 0.0,
        "solvable_selection_accuracy": _ratio(model, oracle),
    }
