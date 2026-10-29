"""Two distinct Jev judgements for a Text-to-SQL candidate pool.

The raw ``noul`` values are model judgements, not calibrated frequencies. A
separate train-only calibration step is needed before treating them as
probabilities of correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any, Mapping, Sequence

from alphasql.jev.client import JevClient
from alphasql.jev.models import NoulQuestion, NoulResult, Usage


@dataclass(frozen=True)
class PoolJudgement:
    coverage_raw: float
    candidate_raw: tuple[float, ...]
    usage: Usage
    diagnostics: Mapping[str, float] = field(default_factory=dict)
    model: str = ""


class JevPoolJudge:
    """Ask about pool coverage and each candidate in the same SQL context."""

    def __init__(self, client: JevClient, max_candidates: int = 32) -> None:
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        self.client = client
        self.max_candidates = max_candidates

    def assess(
        self,
        question: str,
        evidence: str,
        schema: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
    ) -> PoolJudgement:
        if len(candidates) > self.max_candidates:
            raise ValueError("Candidate pool exceeds the configured Jev context limit")
        valid_indices = [i for i, candidate in enumerate(candidates) if candidate["execution"]["valid"]]
        if not valid_indices:
            return PoolJudgement(0.0, (0.0,) * len(candidates), Usage())
        state = {
            "task": "Judge the given SQL candidates. Do not generate or revise SQL.",
            "question": question,
            "evidence": evidence,
            "schema": dict(schema),
            "candidates": [
                {"index": i, "sql": candidates[i]["sql"],
                 "generation_route": candidates[i].get("route"),
                 "linked_tables": candidates[i].get("selected_tables", []),
                 "execution": candidates[i]["execution"]}
                for i in valid_indices
            ],
        }
        questions = {
            "pool_covered": NoulQuestion(
                "Is at least one candidate fully correct for the question, including "
                "projection, filters, joins, aggregation and result multiplicity? "
                "A query merely executing is insufficient."
            )
        }
        questions.update({
            "gap_schema": NoulQuestion(
                "If the candidates are wrong, is missing or incorrect table/column/value "
                "grounding or JOIN path a likely cause?"
            ),
            "gap_projection": NoulQuestion(
                "If the candidates are wrong, is SELECT projection or result multiplicity "
                "a likely cause?"
            ),
            "gap_filters": NoulQuestion(
                "If the candidates are wrong, are predicates, literals or date boundaries "
                "a likely cause?"
            ),
            "gap_logic": NoulQuestion(
                "If the candidates are wrong, is aggregation, grouping, ordering, LIMIT "
                "or SQL structure a likely cause?"
            ),
        })
        questions.update({
            f"candidate_{i}": NoulQuestion(
                f"Is candidate {i} fully correct for the question, including all "
                "requested conditions and result multiplicity?"
            ) for i in valid_indices
        })
        response = self.client.ask(state, questions)
        raw = [0.0] * len(candidates)
        for i in valid_indices:
            raw[i] = response.require(f"candidate_{i}", NoulResult).noul
        return PoolJudgement(
            response.require("pool_covered", NoulResult).noul,
            tuple(raw), response.usage,
            {name: response.require(f"gap_{name}", NoulResult).noul
             for name in ("schema", "projection", "filters", "logic")},
            response.model,
        )

    def adjudicate(self, question: str, evidence: str, schema: Mapping[str, Any],
                   candidates: Sequence[Mapping[str, Any]],
                   scores: Sequence[float]) -> dict[str, Any]:
        """Independent comparison of up to four valid SQLs; recommendation only.

        This action does not establish correctness or silently change Top-1.
        """
        ranked = sorted((i for i, item in enumerate(candidates)
                         if item["execution"]["valid"]),
                        key=lambda i: (-scores[i], i))[:4]
        if not ranked:
            return {"recommended_index": None, "scores": {}, "usage": Usage(),
                    "model": "", "called": False}
        questions = {f"verify_{i}": NoulQuestion(
            f"After independently checking the question, schema, SQL {i} and its "
            "execution summary, is this SQL fully correct? Check all requested "
            "conditions, joins, projection and multiplicity.") for i in ranked}
        for position, i in enumerate(ranked):
            for j in ranked[position + 1:]:
                questions[f"prefer_{i}_{j}"] = NoulQuestion(
                    f"Comparing SQL {i} and SQL {j}, is SQL {i} more likely to "
                    "fully satisfy the question than SQL {j}?")
        response = self.client.ask({
            "task": "Adjudicate existing SQL candidates without generating SQL.",
            "question": question, "evidence": evidence, "schema": dict(schema),
            "candidates": [{"index": i, "sql": candidates[i]["sql"],
                            "execution": candidates[i]["execution"]} for i in ranked],
        }, questions)
        totals = {i: response.require(f"verify_{i}", NoulResult).noul for i in ranked}
        for position, i in enumerate(ranked):
            for j in ranked[position + 1:]:
                preference = response.require(f"prefer_{i}_{j}", NoulResult).noul
                totals[i] += preference
                totals[j] += 1 - preference
        recommended = max(ranked, key=lambda i: (totals[i], -ranked.index(i)))
        return {"recommended_index": recommended,
                "scores": {str(i): totals[i] / len(ranked) for i in ranked},
                "usage": response.usage, "model": response.model, "called": True}
