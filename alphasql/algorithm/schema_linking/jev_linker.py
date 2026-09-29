"""Schema-aware linking through batched, typed Jev decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from alphasql.algorithm.compiler.query_plan import ColumnRef, ForeignKeyEdge, SchemaGraph
from alphasql.database.schema import DatabaseSchema
from alphasql.jev.client import JevClient
from alphasql.jev.models import ChoiceQuestion, ChoiceResult, NoulQuestion, NoulResult, Usage
from alphasql.jev.runtime import JevSettings


@dataclass(frozen=True)
class ScoredTable:
    table: str
    probability: float
    confidence: float
    selected: bool


@dataclass(frozen=True)
class ScoredColumn:
    table: str
    column: str
    probability: float
    confidence: float
    selected: bool
    added_by_constraint: bool = False


@dataclass(frozen=True)
class CandidateJoinPath:
    edges: Tuple[ForeignKeyEdge, ...]
    probability: float
    confidence: float
    selected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edges": [edge.to_dict() for edge in self.edges],
            "probability": self.probability,
            "confidence": self.confidence,
            "selected": self.selected,
        }


@dataclass(frozen=True)
class LiteralCandidate:
    table: str
    column: str
    value: Any
    probability: float
    confidence: float
    selected: bool


@dataclass(frozen=True)
class IntentDecision:
    choice: str
    probabilities: Dict[str, float]
    confidence: float


@dataclass
class SchemaLinkingResult:
    tables: List[ScoredTable]
    columns: List[ScoredColumn]
    join_paths: List[CandidateJoinPath]
    literals: List[LiteralCandidate]
    decisions: Dict[str, IntentDecision]
    usage: Usage = field(default_factory=Usage)

    @property
    def selected_tables(self) -> List[str]:
        return [item.table for item in self.tables if item.selected]

    @property
    def selected_columns(self) -> List[ColumnRef]:
        return [ColumnRef(item.table, item.column) for item in self.columns if item.selected]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tables": [asdict(item) for item in self.tables],
            "columns": [asdict(item) for item in self.columns],
            "join_paths": [item.to_dict() for item in self.join_paths],
            "literals": [asdict(item) for item in self.literals],
            "decisions": {key: asdict(value) for key, value in self.decisions.items()},
            "usage": asdict(self.usage),
        }


class JevSchemaLinker:
    """Ask one atomic Noul per table/column/value and typed Choices for SQL intent."""

    INTENT_CHOICES = {
        "aggregate": {
            "none": "No aggregation is needed.",
            "count": "The answer requires COUNT.",
            "sum": "The answer requires SUM.",
            "avg": "The answer requires AVG.",
            "min": "The answer requires MIN.",
            "max": "The answer requires MAX.",
        },
        "distinct": {
            "no": "Duplicate rows should be preserved.",
            "yes": "Duplicate values must be removed.",
        },
        "order": {
            "none": "No ordering is requested.",
            "asc": "Ascending order is requested.",
            "desc": "Descending order is requested.",
        },
        "comparison": {
            "=": "Equality filtering is requested.",
            "!=": "Inequality filtering is requested.",
            ">": "A strict greater-than filter is requested.",
            ">=": "A greater-than-or-equal filter is requested.",
            "<": "A strict less-than filter is requested.",
            "<=": "A less-than-or-equal filter is requested.",
            "LIKE": "A text pattern or containment filter is requested.",
        },
        "date_function": {
            "none": "No SQLite date transformation is required.",
            "date": "Normalize or compare a calendar date.",
            "datetime": "Normalize or compare date and time.",
            "julianday": "Compute elapsed time using Julian days.",
            "strftime": "Extract or format a date component.",
        },
    }
    INTENT_INSTRUCTIONS = {
        "aggregate": "Which aggregation, if any, is required by the user question?",
        "distinct": "Must duplicate output values be removed to answer the user question?",
        "order": "Which output ordering direction, if any, is explicitly or implicitly required?",
        "comparison": "Which comparison operator best represents the requested filter?",
        "date_function": "Which SQLite date function, if any, is required?",
    }

    def __init__(
        self,
        client: JevClient,
        settings: JevSettings | None = None,
        max_join_hops: int = 4,
        max_join_paths: int = 32,
        max_literal_questions: int = 64,
    ) -> None:
        self.client = client
        self.settings = settings or client.settings
        self.max_join_hops = max_join_hops
        self.max_join_paths = max_join_paths
        self.max_literal_questions = max_literal_questions

    @staticmethod
    def _schema_state(schema: DatabaseSchema) -> List[Dict[str, Any]]:
        output = []
        for table_key, table in schema.tables.items():
            table_name = table.table_name or table_key
            columns = []
            for column_key, column in table.columns.items():
                columns.append({
                    "name": column.original_column_name or column_key,
                    "type": column.column_type,
                    "meaning": column.expanded_column_name,
                    "description": column.column_description,
                    "value_description": column.value_description,
                    "primary_key": column.primary_key,
                    "foreign_keys": list(column.foreign_keys),
                    "examples": list(column.value_examples[:5]),
                })
            output.append({"name": table_name, "columns": columns})
        return output

    def _all_join_paths(self, graph: SchemaGraph) -> List[Tuple[ForeignKeyEdge, ...]]:
        paths: List[Tuple[ForeignKeyEdge, ...]] = []
        seen = set()
        tables = sorted({table.table_name or key for key, table in graph.schema.tables.items()})
        for left, right in combinations(tables, 2):
            for path in graph.paths(left, right, self.max_join_hops):
                signature = tuple(
                    (edge.source.table, edge.source.column, edge.target.table, edge.target.column)
                    for edge in path
                )
                if signature not in seen:
                    seen.add(signature)
                    paths.append(tuple(path))
                    if len(paths) >= self.max_join_paths:
                        return paths
        return paths

    @staticmethod
    def _path_description(path: Sequence[ForeignKeyEdge]) -> str:
        return " then ".join(
            f"{edge.source.table}.{edge.source.column} = {edge.target.table}.{edge.target.column}"
            for edge in path
        )

    @staticmethod
    def _literal_values(
        schema: DatabaseSchema,
        value_examples: Optional[Mapping[str, Mapping[str, Sequence[Any]]]],
    ) -> List[Tuple[str, str, Any]]:
        values: List[Tuple[str, str, Any]] = []
        seen = set()
        columns = [
            (table.table_name or table_key, column.original_column_name or column_key, column)
            for table_key, table in schema.tables.items()
            for column_key, column in table.columns.items()
        ]

        def append(table_name: str, column_name: str, candidates: Iterable[Any]) -> None:
            for value in candidates:
                signature = (table_name, column_name, str(value))
                if signature not in seen and str(value).strip():
                    seen.add(signature)
                    values.append((table_name, column_name, value))

        # This must be global, not merely per-column: a cap of N literals should
        # never be exhausted by generic values from early tables before a later
        # table's exact question match is considered.
        for table_name, column_name, _ in columns:
            append(
                table_name,
                column_name,
                (value_examples or {}).get(table_name, {}).get(column_name, []),
            )
        for table_name, column_name, column in columns:
            append(table_name, column_name, column.value_examples)
        return values

    def link(
        self,
        question: str,
        evidence: str,
        schema: DatabaseSchema,
        value_examples: Optional[Mapping[str, Mapping[str, Sequence[Any]]]] = None,
    ) -> SchemaLinkingResult:
        graph = SchemaGraph(schema)
        questions: Dict[str, Any] = {}
        table_keys: Dict[str, str] = {}
        column_keys: Dict[str, Tuple[str, str]] = {}
        for table_index, (table_key, table) in enumerate(schema.tables.items()):
            table_name = table.table_name or table_key
            key = f"table_{table_index}"
            table_keys[key] = table_name
            questions[key] = NoulQuestion(
                f"Is table '{table_name}' semantically necessary to answer the user question?"
            )
            for column_index, (column_key, column) in enumerate(table.columns.items()):
                column_name = column.original_column_name or column_key
                key = f"column_{table_index}_{column_index}"
                column_keys[key] = (table_name, column_name)
                questions[key] = NoulQuestion(
                    f"Is column '{table_name}.{column_name}' needed for projection, filtering, joining, "
                    "grouping, ordering, or date logic in the answer?"
                )
        join_paths = self._all_join_paths(graph)
        if join_paths:
            criteria = {"no_join": "The query should use only one table and needs no join."}
            criteria.update({
                f"path_{index}": self._path_description(path)
                for index, path in enumerate(join_paths)
            })
            questions["join_path"] = ChoiceQuestion(
                criteria,
                instructions=(
                    "Which single schema join path is necessary to answer the user question? "
                    "Choose no_join when one table is sufficient."
                ),
            )
        for name, criteria in self.INTENT_CHOICES.items():
            questions[f"intent_{name}"] = ChoiceQuestion(
                criteria, instructions=self.INTENT_INSTRUCTIONS[name]
            )
        literal_keys: Dict[str, Tuple[str, str, Any]] = {}
        for index, literal in enumerate(self._literal_values(schema, value_examples)[:self.max_literal_questions]):
            key = f"literal_{index}"
            literal_keys[key] = literal
            table, column, value = literal
            questions[key] = NoulQuestion(
                f"Should the literal value {value!r} from '{table}.{column}' be used in a SQL filter?"
            )
        state = {
            "task": "Make independent typed decisions for a constrained SQLite query plan.",
            "question": question,
            "evidence": evidence or "",
            "schema": self._schema_state(schema),
        }
        response = self.client.ask(state, questions)
        threshold = self.settings.schema_threshold
        tables = []
        for key, table in table_keys.items():
            answer = response.require(key, NoulResult)
            tables.append(ScoredTable(table, answer.noul, answer.confidence, answer.noul >= threshold))
        if tables and not any(item.selected for item in tables):
            winner = max(range(len(tables)), key=lambda index: tables[index].probability)
            tables[winner] = ScoredTable(**{**asdict(tables[winner]), "selected": True})
        selected_tables = {item.table.lower() for item in tables if item.selected}
        columns = []
        for key, (table, column) in column_keys.items():
            answer = response.require(key, NoulResult)
            selected = answer.noul >= threshold and table.lower() in selected_tables
            columns.append(ScoredColumn(table, column, answer.noul, answer.confidence, selected))
        for table in list(selected_tables):
            indices = [i for i, item in enumerate(columns) if item.table.lower() == table]
            if indices and not any(columns[i].selected for i in indices):
                winner = max(indices, key=lambda index: columns[index].probability)
                columns[winner] = ScoredColumn(**{**asdict(columns[winner]), "selected": True})
        scored_paths: List[CandidateJoinPath] = []
        if join_paths:
            answer = response.require("join_path", ChoiceResult)
            scored_paths.append(CandidateJoinPath(
                (), answer.probabilities.get("no_join", 0.0), answer.confidence,
                answer.choice == "no_join",
            ))
            for index, path in enumerate(join_paths):
                probability = answer.probabilities.get(f"path_{index}", 0.0)
                path_tables = {
                    name.lower() for edge in path for name in (edge.source.table, edge.target.table)
                }
                selected = answer.choice == f"path_{index}" or (
                    len(selected_tables) >= 2 and selected_tables.issubset(path_tables)
                )
                scored_paths.append(CandidateJoinPath(path, probability, answer.confidence, selected))
        # Enforce referential completeness after semantic selection. These are soft additions,
        # not a hard rejection of low-scoring semantic columns.
        required_refs = {(ref.table.lower(), ref.column.lower()) for ref in graph.primary_keys(selected_tables)}
        for path in scored_paths:
            if path.selected:
                for edge in path.edges:
                    selected_tables.update((edge.source.table.lower(), edge.target.table.lower()))
                    required_refs.add((edge.source.table.lower(), edge.source.column.lower()))
                    required_refs.add((edge.target.table.lower(), edge.target.column.lower()))
        required_refs.update(
            (ref.table.lower(), ref.column.lower()) for ref in graph.primary_keys(selected_tables)
        )
        updated_tables = []
        for item in tables:
            updated_tables.append(ScoredTable(
                item.table, item.probability, item.confidence,
                item.selected or item.table.lower() in selected_tables,
            ))
        tables = updated_tables
        existing = {(item.table.lower(), item.column.lower()): index for index, item in enumerate(columns)}
        for table, column in required_refs:
            index = existing.get((table, column))
            if index is not None and not columns[index].selected:
                item = columns[index]
                columns[index] = ScoredColumn(
                    item.table, item.column, item.probability, item.confidence, True, True
                )
        literals = []
        for key, (table, column, value) in literal_keys.items():
            answer = response.require(key, NoulResult)
            literals.append(LiteralCandidate(
                table, column, value, answer.noul, answer.confidence, answer.noul >= threshold
            ))
        decisions = {}
        for name in self.INTENT_CHOICES:
            answer = response.require(f"intent_{name}", ChoiceResult)
            decisions[name] = IntentDecision(
                answer.choice, dict(answer.probabilities), answer.confidence
            )
        return SchemaLinkingResult(tables, columns, scored_paths, literals, decisions, response.usage)
