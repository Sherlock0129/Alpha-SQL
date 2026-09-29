"""Configuration for the independent Jev path."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping, Optional

from dotenv import load_dotenv


class JevConfigurationError(RuntimeError):
    pass


def _number(env: Mapping[str, str], name: str, default: float, cast):
    raw = env.get(name, str(default))
    try:
        return cast(raw)
    except (TypeError, ValueError) as exc:
        raise JevConfigurationError(f"{name} must be a valid {cast.__name__}") from exc


@dataclass(frozen=True)
class JevSettings:
    api_key: Optional[str] = None
    model: str = "jev-latest"
    request_timeout: float = 60.0
    max_retries: int = 1
    schema_threshold: float = 0.50
    rerank_threshold: float = 0.55
    beam_width: int = 8
    min_confidence: float = 0.50

    @classmethod
    def from_environment(
        cls,
        env: Mapping[str, str] | None = None,
        env_file: str | Path = ".env.local",
    ) -> "JevSettings":
        if env is None:
            path = Path(env_file)
            if path.exists():
                load_dotenv(path, override=False)
            env = os.environ
        settings = cls(
            api_key=(env.get("TYPESAFE_API_KEY") or "").strip() or None,
            model=(env.get("JEV_MODEL") or "jev-latest").strip(),
            request_timeout=_number(env, "JEV_REQUEST_TIMEOUT", 60.0, float),
            max_retries=_number(env, "JEV_MAX_RETRIES", 1, int),
            schema_threshold=_number(env, "JEV_SCHEMA_THRESHOLD", 0.50, float),
            rerank_threshold=_number(env, "JEV_RERANK_THRESHOLD", 0.55, float),
            beam_width=_number(env, "JEV_BEAM_WIDTH", 8, int),
            min_confidence=_number(env, "JEV_MIN_CONFIDENCE", 0.50, float),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.model:
            raise JevConfigurationError("JEV_MODEL cannot be empty")
        if self.request_timeout <= 0:
            raise JevConfigurationError("JEV_REQUEST_TIMEOUT must be positive")
        if self.max_retries < 0:
            raise JevConfigurationError("JEV_MAX_RETRIES cannot be negative")
        if self.beam_width <= 0:
            raise JevConfigurationError("JEV_BEAM_WIDTH must be positive")
        for name, value in (
            ("JEV_SCHEMA_THRESHOLD", self.schema_threshold),
            ("JEV_RERANK_THRESHOLD", self.rerank_threshold),
            ("JEV_MIN_CONFIDENCE", self.min_confidence),
        ):
            if not 0.0 <= value <= 1.0:
                raise JevConfigurationError(f"{name} must be between 0 and 1")

    def require_api_key(self) -> str:
        if not self.api_key:
            raise JevConfigurationError(
                "TYPESAFE_API_KEY is required for a real Jev request; no network call was made"
            )
        return self.api_key
