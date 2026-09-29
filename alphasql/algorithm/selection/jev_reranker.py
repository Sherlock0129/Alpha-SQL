"""Jev-based, schema-aware reranking of executable constrained SQL candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

from alphasql.jev.client import JevClient
from alphasql.jev.models import NoulQuestion, NoulResult, ScoreQuestion, ScoreResult, Usage
from alphasql.jev.runtime import JevSettings


QUALITY_SCALE = (
    "Clearly incorrect or absent.",
    "Mostly incorrect; major requirements are missing.",
    "Partially correct but with a material ambiguity or omission.",
    "Mostly correct with only a minor issue.",
    "Fully correct for the user's request.",
)


@dataclass(frozen=True)
class RerankWeights:
    projection: float = 1.0
    filters: float = 1.0
    join: float = 1.0
    logic: float = 1.0
    minimality: float = 0.5

    def normalized(self) -> Dict[str, float]:
        values = asdict(self)
        total = sum(values.values())
        if total <= 0:
            raise ValueError("At least one rerank weight must be positive")
        return {key: value / total for key, value in values.items()}


@dataclass(frozen=True)
class RerankCandidate:
    canonical_sql: str
    ast_summary: Mapping[str, Any]
    execution_summary: Mapping[str, Any]
    selected_schema: Mapping[str, Any]


@dataclass(frozen=True)
class DimensionScore:
    value: float
    confidence: float
    probabilities: Dict[str, float]


@dataclass
class RerankResult:
    candidate_index: int
    score: float
    dimensions: Dict[str, DimensionScore]
    uncertain: bool
    above_threshold: bool
    usage: Usage = field(default_factory=Usage)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "score": self.score,
            "dimensions": {key: asdict(value) for key, value in self.dimensions.items()},
            "uncertain": self.uncertain,
            "above_threshold": self.above_threshold,
            "usage": asdict(self.usage),
        }


class JevSQLReranker:
    def __init__(
        self,
        client: JevClient,
        settings: JevSettings | None = None,
        weights: RerankWeights | None = None,
    ) -> None:
        self.client = client
        self.settings = settings or client.settings
        self.weights = weights or RerankWeights()

    def rerank(
        self,
        question: str,
        hint: str,
        candidates: Sequence[RerankCandidate],
    ) -> List[RerankResult]:
        if not candidates:
            return []
        questions: Dict[str, Any] = {}
        for index in range(len(candidates)):
            questions[f"c{index}_projection"] = ScoreQuestion(
                QUALITY_SCALE,
                instructions=f"Rate whether candidate {index}'s SELECT projection exactly answers the request.",
            )
            questions[f"c{index}_filters"] = ScoreQuestion(
                QUALITY_SCALE,
                instructions=f"Rate whether candidate {index} covers every requested filter correctly.",
            )
            questions[f"c{index}_join"] = NoulQuestion(
                f"Does candidate {index} use schema-valid joins that match the requested relationships?"
            )
            questions[f"c{index}_logic"] = ScoreQuestion(
                QUALITY_SCALE,
                instructions=(
                    f"Rate candidate {index}'s aggregation, grouping, ordering, limit, and date logic."
                ),
            )
            questions[f"c{index}_minimality"] = NoulQuestion(
                f"Does candidate {index} avoid unnecessary tables/columns while also avoiding obvious omissions?"
            )
        state = {
            "task": "Independently assess each constrained SQLite candidate. Do not generate SQL.",
            "question": question,
            "hint": hint or "",
            "candidate_dimension_instructions": {
                "projection": "Does SELECT return exactly what the user asks for?",
                "filters": "Are all requested filters represented with correct values/operators?",
                "join": "Are joins valid and semantically appropriate?",
                "logic": "Are aggregation, grouping, ordering, limit, and date operations correct?",
                "minimality": "Are there unnecessary schema elements or obvious missing elements?",
            },
            "candidates": [
                {
                    "index": index,
                    "sql": candidate.canonical_sql,
                    "ast": dict(candidate.ast_summary),
                    "selected_schema": dict(candidate.selected_schema),
                    "execution": dict(candidate.execution_summary),
                }
                for index, candidate in enumerate(candidates)
            ],
        }
        response = self.client.ask(state, questions)
        weights = self.weights.normalized()
        results = []
        for index in range(len(candidates)):
            dimensions: Dict[str, DimensionScore] = {}
            for name in ("projection", "filters", "logic"):
                answer = response.require(f"c{index}_{name}", ScoreResult)
                dimensions[name] = DimensionScore(
                    answer.score / (len(QUALITY_SCALE) - 1), answer.confidence,
                    dict(answer.probabilities),
                )
            for name in ("join", "minimality"):
                answer = response.require(f"c{index}_{name}", NoulResult)
                dimensions[name] = DimensionScore(
                    answer.noul, answer.confidence, dict(answer.probabilities)
                )
            score = sum(weights[name] * dimensions[name].value for name in weights)
            uncertain = any(
                value.confidence < self.settings.min_confidence for value in dimensions.values()
            )
            results.append(RerankResult(
                candidate_index=index,
                score=score,
                dimensions=dimensions,
                uncertain=uncertain,
                above_threshold=score >= self.settings.rerank_threshold,
                usage=response.usage,
            ))
        return sorted(results, key=lambda item: item.score, reverse=True)
