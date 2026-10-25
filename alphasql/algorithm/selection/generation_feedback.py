"""Turn SQL verification signals into a bounded next generation action."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from alphasql.algorithm.selection.dual_confidence import PoolJudgement


DIMENSIONS = ("schema", "projection", "filters", "logic")


@dataclass(frozen=True)
class GenerationFeedback:
    action: str
    dimension: str
    anchor_sql: str | None
    anchor_schema_tables: tuple[str, ...] = ()
    prior_sqls: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.action not in {"relink", "revise", "diversify"}:
            raise ValueError(f"Unknown generation action: {self.action}")

    def hint(self) -> str:
        """A diagnostic hint, never a replacement for the user's evidence."""
        concern = {
            "schema": "Reconsider table, column, literal-value and JOIN-path grounding from the full schema.",
            "projection": "Recheck SELECT expressions and result multiplicity against the request.",
            "filters": "Recheck every predicate, literal, comparison and date boundary.",
            "logic": "Recheck aggregation, grouping, ordering, LIMIT and SQL structure.",
            "syntax": "Repair the SQL execution or syntax error using the database schema.",
            "uncertain": "Explore a materially different SQL interpretation of the original question.",
        }.get(self.dimension, "Recheck the SQL against the original question.")
        lines = ["Verifier feedback is uncertain; the original question and evidence remain authoritative.",
                 concern]
        if self.anchor_schema_tables:
            lines.append("Previous linked tables: " + ", ".join(self.anchor_schema_tables[:12]))
        if self.anchor_sql:
            lines.append("Previous candidate SQL to inspect: " + self.anchor_sql[:2000])
        if self.action == "diversify" and self.prior_sqls:
            lines.append("Avoid merely repeating these candidates: " +
                         " | ".join(sql[:500] for sql in self.prior_sqls[:3]))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "dimension": self.dimension,
                "anchor_sql": self.anchor_sql,
                "anchor_schema_tables": list(self.anchor_schema_tables)}


def choose_feedback(
    judgement: PoolJudgement,
    candidates: Sequence[Mapping[str, Any]],
    diagnostic_threshold: float = 0.55,
    candidate_scores: Sequence[float] | None = None,
) -> GenerationFeedback:
    """Choose relinking, targeted SQL repair, or diverse regeneration."""
    if not 0 <= diagnostic_threshold <= 1:
        raise ValueError("diagnostic_threshold must be in [0, 1]")
    if not candidates:
        return GenerationFeedback("diversify", "uncertain", None)
    if candidate_scores is not None and len(candidate_scores) != len(candidates):
        raise ValueError("candidate_scores length must match candidates")
    valid = [i for i, row in enumerate(candidates) if row["execution"]["valid"]]
    scores = candidate_scores if candidate_scores is not None else judgement.candidate_raw
    anchor_index = (max(valid, key=lambda i: (scores[i], -i))
                    if valid else 0)
    anchor = candidates[anchor_index]
    anchor_sql = anchor.get("canonical_sql") or anchor["sql"]
    tables = tuple(anchor.get("selected_tables") or ())
    prior = tuple(row.get("canonical_sql") or row["sql"] for row in candidates[-3:])
    if not valid:
        errors = " ".join(str(row["execution"].get("error") or "").lower()
                          for row in candidates)
        dimension = ("schema" if any(phrase in errors for phrase in
                     ("no such table", "no such column", "ambiguous column")) else "syntax")
        action = "relink" if dimension == "schema" else "revise"
    else:
        dimension = max(DIMENSIONS,
                        key=lambda name: (float(judgement.diagnostics.get(name, 0.0)),
                                          -DIMENSIONS.index(name)))
        if float(judgement.diagnostics.get(dimension, 0.0)) < diagnostic_threshold:
            action, dimension = "diversify", "uncertain"
        else:
            action = "relink" if dimension == "schema" else "revise"
    return GenerationFeedback(action, dimension, anchor_sql, tables, prior)
