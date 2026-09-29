"""Syntax and read-only SQLite validation for compiled candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import sqlglot

from alphasql.database.sql_execution import (
    SQLExecutionResult,
    SQLExecutionResultType,
    execute_sql_with_timeout,
)


@dataclass(frozen=True)
class CandidateExecution:
    valid: bool
    status: str
    columns: List[str]
    rows: List[Any]
    error: str | None = None

    def summary(self, max_rows: int = 3) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "status": self.status,
            "columns": self.columns,
            "row_count": len(self.rows),
            "sample_rows": self.rows[:max_rows],
            "error": self.error,
        }


def is_execution_valid(result: SQLExecutionResult) -> bool:
    """Successful execution is valid even when SQLite returns zero rows."""

    return result.result_type is SQLExecutionResultType.SUCCESS


def validate_sql(db_path: str, sql: str, timeout: int = 60) -> CandidateExecution:
    try:
        sqlglot.parse_one(sql, read="sqlite")
    except Exception as exc:
        return CandidateExecution(False, "syntax_error", [], [], str(exc))
    result = execute_sql_with_timeout(db_path, sql, timeout)
    return CandidateExecution(
        valid=is_execution_valid(result),
        status=result.result_type.value,
        columns=list(result.result_cols or []),
        rows=list(result.result or []),
        error=result.error_message,
    )
