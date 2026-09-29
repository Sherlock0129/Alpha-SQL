"""Serializable SQL query plan and schema-graph validation."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from alphasql.database.schema import DatabaseSchema


class Aggregate(str, Enum):
    NONE = "none"
    COUNT = "count"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"


class Comparison(str, Enum):
    EQ = "="
    NE = "!="
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    LIKE = "LIKE"
    IN = "IN"
    BETWEEN = "BETWEEN"
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"


class OrderDirection(str, Enum):
    ASC = "ASC"
    DESC = "DESC"


class DateFunction(str, Enum):
    DATE = "date"
    DATETIME = "datetime"
    JULIANDAY = "julianday"
    STRFTIME = "strftime"


@dataclass(frozen=True)
class ColumnRef:
    table: str
    column: str

    @classmethod
    def from_value(cls, value: Dict[str, str] | "ColumnRef") -> "ColumnRef":
        return value if isinstance(value, cls) else cls(**value)


@dataclass(frozen=True)
class Expression:
    column: Optional[ColumnRef] = None
    aggregate: Aggregate = Aggregate.NONE
    date_function: Optional[DateFunction] = None
    date_format: Optional[str] = None
    date_modifiers: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SelectItem:
    expression: Expression
    alias: Optional[str] = None


@dataclass(frozen=True)
class JoinSpec:
    table: str
    left: ColumnRef
    right: ColumnRef


@dataclass(frozen=True)
class Predicate:
    left: Expression
    operator: Comparison
    value: Any = None
    right_column: Optional[ColumnRef] = None


@dataclass(frozen=True)
class OrderSpec:
    expression: Expression
    direction: OrderDirection = OrderDirection.ASC


@dataclass
class QueryPlan:
    from_table: str
    select: List[SelectItem]
    distinct: bool = False
    joins: List[JoinSpec] = field(default_factory=list)
    where: List[Predicate] = field(default_factory=list)
    group_by: List[Expression] = field(default_factory=list)
    having: List[Predicate] = field(default_factory=list)
    order_by: List[OrderSpec] = field(default_factory=list)
    limit: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value
            if hasattr(value, "__dataclass_fields__"):
                return {key: convert(item) for key, item in asdict(value).items()}
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(item) for item in value]
            return value
        return convert(self)


@dataclass(frozen=True)
class ForeignKeyEdge:
    source: ColumnRef
    target: ColumnRef

    @property
    def tables(self) -> frozenset[str]:
        return frozenset((self.source.table, self.target.table))

    def to_dict(self) -> Dict[str, Any]:
        return {"source": asdict(self.source), "target": asdict(self.target)}


class SchemaValidationError(ValueError):
    pass


class SchemaGraph:
    """Case-preserving lookup and FK graph derived from ``DatabaseSchema``."""

    def __init__(self, schema: DatabaseSchema) -> None:
        self.schema = schema
        self._tables = {name.lower(): name for name in schema.tables}
        self._columns: Dict[str, Dict[str, str]] = {}
        for table_key, table in schema.tables.items():
            table_name = table.table_name or table_key
            self._tables.setdefault(table_name.lower(), table_key)
            self._columns[table_key.lower()] = {
                column_name.lower(): column_name for column_name in table.columns
            }
            self._columns[table_key.lower()].update({
                (column.original_column_name or column_name).lower():
                (column.original_column_name or column_name)
                for column_name, column in table.columns.items()
            })
            self._columns[table_name.lower()] = self._columns[table_key.lower()]
        edges: List[ForeignKeyEdge] = []
        seen = set()
        for table_key, table in schema.tables.items():
            source_table = table.table_name or table_key
            for column_key, column in table.columns.items():
                source_column = column.original_column_name or column_key
                for target_table, target_column in column.foreign_keys:
                    edge = ForeignKeyEdge(
                        ColumnRef(source_table, source_column),
                        ColumnRef(target_table, target_column),
                    )
                    key = (
                        edge.source.table.lower(), edge.source.column.lower(),
                        edge.target.table.lower(), edge.target.column.lower(),
                    )
                    if key not in seen:
                        edges.append(edge)
                        seen.add(key)
        self.edges = edges

    def canonical_table(self, table: str) -> str:
        key = self._tables.get(table.lower())
        if key is None:
            raise SchemaValidationError(f"Unknown table: {table}")
        return self.schema.tables[key].table_name or key

    def canonical_column(self, ref: ColumnRef) -> ColumnRef:
        table = self.canonical_table(ref.table)
        columns = self._columns.get(table.lower(), {})
        column = columns.get(ref.column.lower())
        if column is None:
            raise SchemaValidationError(f"Unknown column: {table}.{ref.column}")
        return ColumnRef(table, column)

    def is_fk_pair(self, left: ColumnRef, right: ColumnRef) -> bool:
        left = self.canonical_column(left)
        right = self.canonical_column(right)
        left_key = (left.table.lower(), left.column.lower())
        right_key = (right.table.lower(), right.column.lower())
        return any(
            {(edge.source.table.lower(), edge.source.column.lower()),
             (edge.target.table.lower(), edge.target.column.lower())}
            == {left_key, right_key}
            for edge in self.edges
        )

    def primary_keys(self, tables: Iterable[str]) -> List[ColumnRef]:
        output = []
        for table_name in tables:
            canonical = self.canonical_table(table_name)
            table = next(
                value for key, value in self.schema.tables.items()
                if key.lower() == canonical.lower() or value.table_name.lower() == canonical.lower()
            )
            output.extend(
                ColumnRef(canonical, column.original_column_name or key)
                for key, column in table.columns.items() if column.primary_key
            )
        return output

    def paths(self, start: str, end: str, max_hops: int = 4) -> List[List[ForeignKeyEdge]]:
        start = self.canonical_table(start)
        end = self.canonical_table(end)
        if start.lower() == end.lower():
            return [[]]
        queue = deque([(start, [], frozenset((start.lower(),)))])
        paths: List[List[ForeignKeyEdge]] = []
        while queue:
            table, path, visited = queue.popleft()
            if len(path) >= max_hops:
                continue
            for edge in self.edges:
                if table.lower() == edge.source.table.lower():
                    next_table = edge.target.table
                elif table.lower() == edge.target.table.lower():
                    next_table = edge.source.table
                else:
                    continue
                if next_table.lower() in visited:
                    continue
                next_path = path + [edge]
                if next_table.lower() == end.lower():
                    paths.append(next_path)
                else:
                    queue.append((next_table, next_path, visited | {next_table.lower()}))
        return paths

    def validate_plan(self, plan: QueryPlan) -> None:
        present = {self.canonical_table(plan.from_table).lower()}
        for join in plan.joins:
            join_table = self.canonical_table(join.table)
            left = self.canonical_column(join.left)
            right = self.canonical_column(join.right)
            if not self.is_fk_pair(left, right):
                raise SchemaValidationError(
                    f"Join is not a declared foreign-key edge: {left.table}.{left.column} = "
                    f"{right.table}.{right.column}"
                )
            endpoints = {left.table.lower(), right.table.lower()}
            if join_table.lower() not in endpoints or not endpoints.intersection(present):
                raise SchemaValidationError(f"Join table {join_table} is not connected to the current plan")
            present.add(join_table.lower())
        if not plan.select:
            raise SchemaValidationError("A query plan needs at least one SELECT item")
        expressions: List[Expression] = [item.expression for item in plan.select]
        expressions += [predicate.left for predicate in plan.where + plan.having]
        expressions += plan.group_by
        expressions += [order.expression for order in plan.order_by]
        for expression in expressions:
            if expression.column is not None:
                canonical = self.canonical_column(expression.column)
                if canonical.table.lower() not in present:
                    raise SchemaValidationError(
                        f"Column {canonical.table}.{canonical.column} uses a table absent from FROM/JOIN"
                    )
            if expression.date_function == DateFunction.STRFTIME and not expression.date_format:
                raise SchemaValidationError("strftime expressions require date_format")
        for predicate in plan.where + plan.having:
            if predicate.right_column is not None:
                right = self.canonical_column(predicate.right_column)
                if right.table.lower() not in present:
                    raise SchemaValidationError(
                        f"Column {right.table}.{right.column} uses a table absent from FROM/JOIN"
                    )
            if predicate.operator == Comparison.BETWEEN and (
                not isinstance(predicate.value, (list, tuple)) or len(predicate.value) != 2
            ):
                raise SchemaValidationError("BETWEEN requires exactly two values")
            if predicate.operator == Comparison.IN and not isinstance(predicate.value, (list, tuple)):
                raise SchemaValidationError("IN requires a value list")
        if plan.limit is not None and plan.limit <= 0:
            raise SchemaValidationError("LIMIT must be positive")
