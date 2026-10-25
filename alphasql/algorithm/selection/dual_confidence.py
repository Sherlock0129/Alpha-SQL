"""Two distinct Jev judgements for a Text-to-SQL candidate pool.

The raw ``noul`` values are model judgements, not calibrated frequencies. A
separate train-only calibration step is needed before treating them as
probabilities of correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from alphasql.jev.client import JevClient
from alphasql.jev.models import NoulQuestion, NoulResult, Usage


@dataclass(frozen=True)
class PoolJudgement:
    coverage_raw: float
    candidate_raw: tuple[float, ...]
    usage: Usage


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
        )
