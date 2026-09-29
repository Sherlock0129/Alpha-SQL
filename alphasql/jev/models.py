"""Internal, SDK-independent models for typed Jev questions and answers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Sequence, Union


class JevResponseError(ValueError):
    """Raised when System One returns an incomplete or malformed response."""


def _probability(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise JevResponseError(f"{field_name} must be numeric") from exc
    if not 0.0 <= parsed <= 1.0:
        raise JevResponseError(f"{field_name} must be between 0 and 1")
    return parsed


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @classmethod
    def from_value(cls, value: Any) -> "Usage":
        if value is None:
            return cls()
        if isinstance(value, Mapping):
            return cls(
                input_tokens=int(value.get("input_tokens", 0) or 0),
                output_tokens=int(value.get("output_tokens", 0) or 0),
            )
        return cls(
            input_tokens=int(getattr(value, "input_tokens", 0) or 0),
            output_tokens=int(getattr(value, "output_tokens", 0) or 0),
        )


@dataclass(frozen=True)
class ChoiceQuestion:
    """A finite decision. Keys are machine-readable choices; values define criteria."""

    criteria: Mapping[str, str]
    instructions: Any = None

    def __post_init__(self) -> None:
        if len(self.criteria) < 2:
            raise ValueError("Choice questions require at least two choices")


@dataclass(frozen=True)
class ScoreQuestion:
    """An ordinal decision whose criteria are ordered from worst to best."""

    criteria: Sequence[str]
    instructions: Any = None

    def __post_init__(self) -> None:
        if not 2 <= len(self.criteria) <= 10:
            raise ValueError("Score questions require between 2 and 10 criteria")


@dataclass(frozen=True)
class NoulQuestion:
    """An atomic yes/no uncertainty question."""

    instructions: str

    def __post_init__(self) -> None:
        if not self.instructions.strip():
            raise ValueError("Noul instructions cannot be empty")


TypedQuestion = Union[ChoiceQuestion, ScoreQuestion, NoulQuestion]


@dataclass(frozen=True)
class ChoiceResult:
    choice: str
    probabilities: Dict[str, float]
    confidence: float
    usage: Usage = field(default_factory=Usage)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScoreResult:
    score: float
    probabilities: Dict[str, float]
    confidence: float
    usage: Usage = field(default_factory=Usage)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NoulResult:
    noul: float
    probabilities: Dict[str, float]
    confidence: float
    usage: Usage = field(default_factory=Usage)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


TypedResult = Union[ChoiceResult, ScoreResult, NoulResult]


@dataclass(frozen=True)
class JevResponse:
    answers: Dict[str, TypedResult]
    usage: Usage
    model: str = ""

    def require(self, key: str, result_type: type) -> TypedResult:
        if key not in self.answers:
            raise JevResponseError(f"Jev response is missing answer '{key}'")
        answer = self.answers[key]
        if not isinstance(answer, result_type):
            raise JevResponseError(
                f"Answer '{key}' has type {type(answer).__name__}, expected {result_type.__name__}"
            )
        return answer


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    if hasattr(value, "__dict__"):
        return vars(value)
    raise JevResponseError(f"Expected an answer object, got {type(value).__name__}")


def _probabilities(value: Any, labels: Sequence[str] | None = None) -> Dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): _probability(prob, f"probabilities[{key}]") for key, prob in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        names = list(labels or [str(index) for index in range(len(value))])
        if len(names) != len(value):
            names = [str(index) for index in range(len(value))]
        return {name: _probability(prob, f"probabilities[{name}]") for name, prob in zip(names, value)}
    raise JevResponseError("probabilities must be a mapping or sequence")


def parse_typed_result(
    raw: Any,
    question: TypedQuestion,
    usage: Usage | None = None,
) -> TypedResult:
    """Parse an SDK answer while checking it against the question type."""

    payload = _as_mapping(raw)
    answer_usage = Usage.from_value(payload.get("usage")) if payload.get("usage") else (usage or Usage())
    if isinstance(question, ChoiceQuestion):
        choice = str(payload.get("choice", ""))
        if choice not in question.criteria:
            raise JevResponseError(f"Invalid Choice answer '{choice}'")
        probabilities = _probabilities(payload.get("probabilities"), list(question.criteria))
        if not probabilities:
            raise JevResponseError("Choice answer is missing probabilities")
        confidence = _probability(payload.get("confidence", probabilities.get(choice, 0.0)), "confidence")
        return ChoiceResult(choice, probabilities, confidence, answer_usage)
    if isinstance(question, ScoreQuestion):
        if "score" not in payload:
            raise JevResponseError("Score answer is missing score")
        score = float(payload["score"])
        if not 0 <= score <= len(question.criteria) - 1:
            raise JevResponseError("Score answer is outside the configured scale")
        probabilities = _probabilities(payload.get("probabilities"), [str(i) for i in range(len(question.criteria))])
        if not probabilities:
            raise JevResponseError("Score answer is missing probabilities")
        confidence = _probability(payload.get("confidence", max(probabilities.values())), "confidence")
        return ScoreResult(score, probabilities, confidence, answer_usage)
    if "noul" not in payload:
        raise JevResponseError("Noul answer is missing noul")
    noul = _probability(payload["noul"], "noul")
    probabilities = _probabilities(payload.get("probabilities")) or {"false": 1.0 - noul, "true": noul}
    # The SDK exposes only `noul`; confidence is a transparent derived margin.
    confidence = _probability(payload.get("confidence", abs(2.0 * noul - 1.0)), "confidence")
    return NoulResult(noul, probabilities, confidence, answer_usage)
