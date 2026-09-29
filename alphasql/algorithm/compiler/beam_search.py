"""Probability-aware beam construction of constrained query plans."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

from alphasql.algorithm.compiler.query_plan import (
    Aggregate,
    ColumnRef,
    Comparison,
    DateFunction,
    Expression,
    JoinSpec,
    OrderDirection,
    OrderSpec,
    Predicate,
    QueryPlan,
    SchemaGraph,
    SelectItem,
)
from alphasql.algorithm.schema_linking.jev_linker import CandidateJoinPath, SchemaLinkingResult


@dataclass(frozen=True)
class DecisionOption:
    value: Any
    probability: float


@dataclass
class BeamState:
    values: Dict[str, Any] = field(default_factory=dict)
    log_probability: float = 0.0
    trace: Dict[str, float] = field(default_factory=dict)

    @property
    def probability(self) -> float:
        return math.exp(self.log_probability)


@dataclass(frozen=True)
class PlanCandidate:
    plan: QueryPlan
    probability: float
    decision_probabilities: Dict[str, float]
    uncertain: bool = False


def expand_beam(
    beam: Sequence[BeamState],
    name: str,
    options: Iterable[DecisionOption],
    width: int,
) -> List[BeamState]:
    """Expand every supplied probability, then prune by joint log probability."""

    expanded: List[BeamState] = []
    for state in beam:
        for option in options:
            probability = max(min(float(option.probability), 1.0), 1e-12)
            expanded.append(BeamState(
                values={**state.values, name: option.value},
                log_probability=state.log_probability + math.log(probability),
                trace={**state.trace, name: float(option.probability)},
            ))
    return sorted(expanded, key=lambda item: item.log_probability, reverse=True)[:width]


class QueryPlanBeamSearch:
    def __init__(self, graph: SchemaGraph, beam_width: int = 8, min_confidence: float = 0.50) -> None:
        if beam_width <= 0:
            raise ValueError("beam_width must be positive")
        self.graph = graph
        self.beam_width = beam_width
        self.min_confidence = min_confidence

    @staticmethod
    def _options(probabilities: Mapping[str, float], transform: Callable[[str], Any]) -> List[DecisionOption]:
        return [DecisionOption(transform(name), probability) for name, probability in probabilities.items()]

    @staticmethod
    def _joins(path: CandidateJoinPath, start_table: str) -> List[JoinSpec]:
        remaining = list(path.edges)
        present = {start_table.lower()}
        joins: List[JoinSpec] = []
        while remaining:
            progress = False
            for index, edge in enumerate(remaining):
                source_present = edge.source.table.lower() in present
                target_present = edge.target.table.lower() in present
                if source_present == target_present:
                    continue
                new_table = edge.target.table if source_present else edge.source.table
                joins.append(JoinSpec(new_table, edge.source, edge.target))
                present.add(new_table.lower())
                remaining.pop(index)
                progress = True
                break
            if not progress:
                raise ValueError("Join path is disconnected from the selected FROM table")
        return joins

    def generate(self, linked: SchemaLinkingResult) -> List[PlanCandidate]:
        semantic_columns = [item for item in linked.columns if item.selected and not item.added_by_constraint]
        columns = semantic_columns or [item for item in linked.columns if item.selected]
        if not columns:
            return []
        projection_options = [
            DecisionOption(ColumnRef(item.table, item.column), max(item.probability, 1e-6))
            for item in columns
        ]
        aggregate = linked.decisions["aggregate"]
        distinct = linked.decisions["distinct"]
        order = linked.decisions["order"]
        comparison = linked.decisions["comparison"]
        date_function = linked.decisions["date_function"]
        join_options = [
            DecisionOption(path, max(path.probability, 1e-6)) for path in linked.join_paths
        ] or [DecisionOption(None, 1.0)]
        selected_literals = [item for item in linked.literals if item.selected]
        no_literal_probability = math.prod(1.0 - item.probability for item in selected_literals)
        literal_options = [DecisionOption(None, max(no_literal_probability, 1e-6))]
        literal_options.extend(
            DecisionOption(item, max(item.probability, 1e-6)) for item in selected_literals
        )
        beam = [BeamState()]
        decisions = (
            ("projection", projection_options),
            ("aggregate", self._options(aggregate.probabilities, Aggregate)),
            ("distinct", self._options(distinct.probabilities, lambda value: value == "yes")),
            ("join", join_options),
            ("literal", literal_options),
            ("comparison", self._options(comparison.probabilities, Comparison)),
            ("date_function", self._options(
                date_function.probabilities,
                lambda value: None if value == "none" else DateFunction(value),
            )),
            ("order", self._options(order.probabilities, lambda value: value)),
        )
        for name, options in decisions:
            beam = expand_beam(beam, name, options, self.beam_width)
        uncertain = any(
            decision.confidence < self.min_confidence for decision in linked.decisions.values()
        )
        candidates: List[PlanCandidate] = []
        seen = set()
        for state in beam:
            projection: ColumnRef = state.values["projection"]
            path: CandidateJoinPath | None = state.values["join"]
            literal = state.values["literal"]
            expression = Expression(
                column=projection,
                aggregate=state.values["aggregate"],
                date_function=state.values["date_function"],
                date_format="%Y-%m-%d" if state.values["date_function"] == DateFunction.STRFTIME else None,
            )
            try:
                joins = self._joins(path, projection.table) if path is not None else []
                where = []
                if literal is not None:
                    where.append(Predicate(
                        Expression(ColumnRef(literal.table, literal.column)),
                        state.values["comparison"],
                        literal.value,
                    ))
                order_by = []
                if state.values["order"] != "none":
                    direction = OrderDirection(state.values["order"].upper())
                    order_by = [OrderSpec(expression, direction)]
                plan = QueryPlan(
                    from_table=projection.table,
                    select=[SelectItem(expression)],
                    distinct=state.values["distinct"],
                    joins=joins,
                    where=where,
                    order_by=order_by,
                )
                self.graph.validate_plan(plan)
                signature = str(plan.to_dict())
                if signature in seen:
                    continue
                seen.add(signature)
                candidates.append(PlanCandidate(plan, state.probability, state.trace, uncertain))
            except (ValueError, KeyError):
                continue
        return candidates
