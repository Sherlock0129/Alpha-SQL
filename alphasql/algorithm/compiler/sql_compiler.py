"""Compile a validated ``QueryPlan`` into canonical SQLite SQL."""

from __future__ import annotations

import math
from typing import Any

import sqlglot

from alphasql.algorithm.compiler.query_plan import (
    Aggregate,
    Comparison,
    DateFunction,
    Expression,
    Predicate,
    QueryPlan,
    SchemaGraph,
)


class SQLCompilationError(ValueError):
    pass


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise SQLCompilationError("Non-finite numeric literals are not supported")
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


class SQLCompiler:
    def __init__(self, schema_graph: SchemaGraph) -> None:
        self.graph = schema_graph

    def _expression(self, expression: Expression) -> str:
        if expression.column is None:
            base = "*"
        else:
            ref = self.graph.canonical_column(expression.column)
            base = f"{quote_identifier(ref.table)}.{quote_identifier(ref.column)}"
        if expression.date_function:
            args = []
            if expression.date_function == DateFunction.STRFTIME:
                args.append(sql_literal(expression.date_format))
            args.append(base)
            args.extend(sql_literal(modifier) for modifier in expression.date_modifiers)
            base = f"{expression.date_function.value.upper()}({', '.join(args)})"
        if expression.aggregate != Aggregate.NONE:
            base = f"{expression.aggregate.value.upper()}({base})"
        return base

    def _predicate(self, predicate: Predicate) -> str:
        left = self._expression(predicate.left)
        operator = predicate.operator.value
        if predicate.operator in (Comparison.IS_NULL, Comparison.IS_NOT_NULL):
            return f"{left} {operator}"
        if predicate.right_column is not None:
            right = self.graph.canonical_column(predicate.right_column)
            return f"{left} {operator} {quote_identifier(right.table)}.{quote_identifier(right.column)}"
        if predicate.operator == Comparison.IN:
            values = ", ".join(sql_literal(value) for value in predicate.value)
            return f"{left} IN ({values})"
        if predicate.operator == Comparison.BETWEEN:
            return f"{left} BETWEEN {sql_literal(predicate.value[0])} AND {sql_literal(predicate.value[1])}"
        return f"{left} {operator} {sql_literal(predicate.value)}"

    def compile(self, plan: QueryPlan) -> str:
        try:
            self.graph.validate_plan(plan)
            select = []
            for item in plan.select:
                value = self._expression(item.expression)
                if item.alias:
                    value += f" AS {quote_identifier(item.alias)}"
                select.append(value)
            sql = "SELECT " + ("DISTINCT " if plan.distinct else "") + ", ".join(select)
            from_table = self.graph.canonical_table(plan.from_table)
            sql += f" FROM {quote_identifier(from_table)}"
            for join in plan.joins:
                table = self.graph.canonical_table(join.table)
                left = self.graph.canonical_column(join.left)
                right = self.graph.canonical_column(join.right)
                sql += (
                    f" INNER JOIN {quote_identifier(table)} ON "
                    f"{quote_identifier(left.table)}.{quote_identifier(left.column)} = "
                    f"{quote_identifier(right.table)}.{quote_identifier(right.column)}"
                )
            if plan.where:
                sql += " WHERE " + " AND ".join(self._predicate(item) for item in plan.where)
            if plan.group_by:
                sql += " GROUP BY " + ", ".join(self._expression(item) for item in plan.group_by)
            if plan.having:
                sql += " HAVING " + " AND ".join(self._predicate(item) for item in plan.having)
            if plan.order_by:
                sql += " ORDER BY " + ", ".join(
                    f"{self._expression(item.expression)} {item.direction.value}" for item in plan.order_by
                )
            if plan.limit is not None:
                sql += f" LIMIT {int(plan.limit)}"
            parsed = sqlglot.parse_one(sql, read="sqlite")
            canonical = parsed.sql(dialect="sqlite", identify=True, pretty=False, comments=False)
            # Reparse the emitted dialect to catch renderer mistakes.
            sqlglot.parse_one(canonical, read="sqlite")
            return canonical
        except SQLCompilationError:
            raise
        except Exception as exc:
            raise SQLCompilationError(f"Cannot compile constrained query plan: {exc}") from exc
