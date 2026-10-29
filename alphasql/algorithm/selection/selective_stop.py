"""Validation-only selective stopping for a fixed candidate scorer.

Risk is the observed error fraction among accepted first-round questions.
The pointwise Wilson bound is descriptive after threshold search: it is neither
a post-selection nor a distribution-shift guarantee. Gold labels are used only
by the offline fitter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping, Sequence


def _first_round(row: Mapping[str, Any], score_key: str) -> tuple[float, int] | None:
    state = row["rounds"][0]
    valid = [(float(item[score_key]), int(item["execution_label"]))
             for item in state["candidates"] if item["execution_rechecked_valid"]]
    return max(valid, key=lambda pair: pair[0]) if valid else None


def _wilson_upper(errors: int, count: int, confidence: float) -> float | None:
    if not count:
        return None
    z = NormalDist().inv_cdf(confidence)
    p = errors / count
    denominator = 1 + z * z / count
    center = p + z * z / (2 * count)
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count))
    return min(1.0, (center + radius) / denominator)


@dataclass(frozen=True)
class StopPolicy:
    score_key: str
    threshold: float | None
    max_empirical_risk: float
    validation_questions: int
    accepted_questions: int
    validation_databases: tuple[str, ...]
    confidence: float = 0.95
    model_sha256: str | None = None
    pool_review_below: float | None = 0.8
    generator_version: str | None = None
    judge_model: str | None = None
    criterion: str = "empirical"
    selected_error_fraction: float | None = None
    selected_wilson_upper: float | None = None
    version: int = 1

    def with_model_digest(self, digest: str) -> "StopPolicy":
        return replace(self, model_sha256=digest)

    def accepts(self, best_score: float, pool_score: float | None = None) -> bool:
        # A low pool score requests review; it is not an uncovered-pool label.
        return (self.threshold is not None and best_score >= self.threshold
                and (self.pool_review_below is None or
                     pool_score is not None and pool_score >= self.pool_review_below))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "StopPolicy":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["validation_databases"] = tuple(data["validation_databases"])
        policy = cls(**data)
        if policy.version != 1 or policy.criterion not in {"empirical", "wilson"} or policy.score_key not in {
                "candidate_raw", "candidate_probability", "verifier_probability"}:
            raise ValueError("Unsupported stop policy")
        return policy


def fit_stop_policy(rows: Sequence[Mapping[str, Any]], *, score_key: str,
                    max_empirical_risk: float = 0.10, confidence: float = 0.95,
                    min_accepted: int = 1, pool_review_below: float | None = .8,
                    criterion: str = "empirical"
                    ) -> tuple[StopPolicy, list[dict[str, Any]]]:
    if score_key not in {"candidate_raw", "candidate_probability", "verifier_probability"}:
        raise ValueError("Unsupported score_key")
    if not 0 <= max_empirical_risk < 1 or not .5 < confidence < 1 or min_accepted < 1:
        raise ValueError("Invalid risk, confidence or minimum acceptance")
    if pool_review_below is not None and not 0 <= pool_review_below <= 1:
        raise ValueError("Invalid pool review threshold")
    if criterion not in {"empirical", "wilson"}:
        raise ValueError("criterion must be empirical or wilson")
    dbs = [str(row["db_id"]) for row in rows]
    if len(dbs) < 2 or len(set(dbs)) < 2:
        raise ValueError("Use a database-disjoint validation set with at least two databases")
    if len({(str(row["db_id"]), str(row["question_id"])) for row in rows}) != len(rows):
        raise ValueError("Duplicate validation question")
    generator_versions = {row.get("generator", {}).get("version") for row in rows}
    judge_models = {state["jev"].get("model") for row in rows
                    for state in row["rounds"][:1] if state["jev"].get("called")}
    if len(generator_versions) != 1 or len(judge_models) > 1:
        raise ValueError("Stop policy requires one frozen generator and judge version")
    observations = [(_first_round(row, score_key),
                     float(row["rounds"][0]["jev"]["pool_covered"])) for row in rows]
    valid = [item for item, pool in observations if item is not None
             and (pool_review_below is None or pool >= pool_review_below)]
    observed_scores = sorted({score for score, _ in valid})
    # Every achievable acceptance set is represented. Use an interval midpoint
    # rather than the lowest accepted observed score; the latter needlessly
    # rejects unseen scores that fall in an empty validation interval.
    thresholds = ([observed_scores[0]] if observed_scores else []) + [
        (lower + upper) / 2 for lower, upper in zip(observed_scores, observed_scores[1:])]
    curve = [{"threshold": None, "accepted": 0, "acceptance_fraction": 0.0,
              "errors": 0, "error_fraction": None, "wilson_upper": None}]
    for threshold in thresholds:
        selected = [label for score, label in valid if score >= threshold]
        count = len(selected)
        errors = count - sum(selected)
        curve.append({"threshold": threshold, "accepted": count,
                      "acceptance_fraction": count / len(rows), "errors": errors,
                      "error_fraction": errors / count,
                      "wilson_upper": _wilson_upper(errors, count, confidence)})
    risk_key = "error_fraction" if criterion == "empirical" else "wilson_upper"
    eligible = [item for item in curve[1:] if item["accepted"] >= min_accepted
                and item[risk_key] <= max_empirical_risk]
    chosen = max(eligible, key=lambda item: (item["accepted"], -item["threshold"])) if eligible else curve[0]
    policy = StopPolicy(score_key, chosen["threshold"], max_empirical_risk,
                        len(rows), chosen["accepted"], tuple(sorted(set(dbs))), confidence,
                        pool_review_below=pool_review_below,
                        generator_version=next(iter(generator_versions)),
                        judge_model=next(iter(judge_models)) if judge_models else None,
                        criterion=criterion,
                        selected_error_fraction=chosen["error_fraction"],
                        selected_wilson_upper=chosen["wilson_upper"])
    return policy, curve
