"""Small adapter around the official TypeSafe System One Python SDK."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

from alphasql.jev.models import (
    ChoiceQuestion,
    JevResponse,
    JevResponseError,
    NoulQuestion,
    ScoreQuestion,
    TypedQuestion,
    Usage,
    parse_typed_result,
)
from alphasql.jev.runtime import JevConfigurationError, JevSettings


class JevAPIError(RuntimeError):
    pass


class JevTimeoutError(JevAPIError):
    pass


class JevClient:
    """Send independent typed questions in one System One request.

    ``transport`` is an injection seam used by unit tests. It accepts state and the
    internal question mapping and must return a raw/SDK-like response object.
    """

    def __init__(
        self,
        settings: JevSettings | None = None,
        transport: Callable[[Any, Mapping[str, TypedQuestion]], Any] | None = None,
    ) -> None:
        self.settings = settings or JevSettings.from_environment()
        self.transport = transport

    def ask(self, state: Any, questions: Mapping[str, TypedQuestion]) -> JevResponse:
        if not questions:
            raise ValueError("At least one typed Jev question is required")
        try:
            raw = self.transport(state, questions) if self.transport else self._sdk_request(state, questions)
            return self._parse_response(raw, questions)
        except (JevConfigurationError, JevResponseError, JevAPIError):
            raise
        except (TimeoutError, ConnectionError) as exc:
            raise JevTimeoutError(f"TypeSafe System One request timed out: {exc}") from exc
        except Exception as exc:
            name = type(exc).__name__.lower()
            if "timeout" in name:
                raise JevTimeoutError(f"TypeSafe System One request timed out: {exc}") from exc
            raise JevAPIError(f"TypeSafe System One request failed: {type(exc).__name__}: {exc}") from exc

    def _sdk_request(self, state: Any, questions: Mapping[str, TypedQuestion]) -> Any:
        api_key = self.settings.require_api_key()
        try:
            from typesafe_sdk import Choice, Noul, RetryPolicy, Score, TypeSafeClient
        except ImportError as exc:
            raise JevConfigurationError(
                "typesafe-sdk is not installed; install requirements.txt before using the Jev runner"
            ) from exc
        sdk_questions: Dict[str, Any] = {}
        for key, question in questions.items():
            if isinstance(question, ChoiceQuestion):
                sdk_questions[key] = Choice(
                    criteria=dict(question.criteria), instructions=question.instructions
                )
            elif isinstance(question, ScoreQuestion):
                sdk_questions[key] = Score(
                    criteria=list(question.criteria), instructions=question.instructions
                )
            elif isinstance(question, NoulQuestion):
                sdk_questions[key] = Noul(instructions=question.instructions)
            else:
                raise TypeError(f"Unsupported question type for '{key}': {type(question).__name__}")
        client = TypeSafeClient(
            api_key=api_key,
            model=self.settings.model,
            retry=RetryPolicy(max_retries=self.settings.max_retries),
            timeout=self.settings.request_timeout,
        )
        if hasattr(client, "__enter__"):
            with client as session:
                return session.system_one(state=state, questions=sdk_questions)
        return client.system_one(state=state, questions=sdk_questions)

    @staticmethod
    def _get(raw: Any, name: str, default: Any = None) -> Any:
        if isinstance(raw, Mapping):
            return raw.get(name, default)
        return getattr(raw, name, default)

    def _parse_response(
        self,
        raw: Any,
        questions: Mapping[str, TypedQuestion],
    ) -> JevResponse:
        if raw is None:
            raise JevResponseError("TypeSafe System One returned an empty response")
        usage = Usage.from_value(self._get(raw, "usage"))
        answers = self._get(raw, "answers")
        if answers is None:
            # SDK responses also expose answers in type-specific mappings.
            answers = {}
            for name in ("choices", "scores", "nouls"):
                group = self._get(raw, name, {}) or {}
                if isinstance(group, Mapping):
                    answers.update(group)
        if not isinstance(answers, Mapping):
            raise JevResponseError("TypeSafe System One response has no answer mapping")
        missing = sorted(set(questions) - set(answers))
        if missing:
            raise JevResponseError(f"TypeSafe System One response omitted: {', '.join(missing)}")
        parsed = {
            key: parse_typed_result(answers[key], question, usage)
            for key, question in questions.items()
        }
        return JevResponse(
            answers=parsed,
            usage=usage,
            model=str(self._get(raw, "model", self.settings.model) or self.settings.model),
        )
