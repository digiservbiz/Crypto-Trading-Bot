"""Structured validation evidence recording.

Evidence records deliberately exclude credentials and account identifiers.
The recorder is append-only JSONL so controlled-validation results can be
collected without changing trading behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ValidationEvidenceError(ValueError):
    pass


_FORBIDDEN_KEYS = {
    "api_key",
    "apikey",
    "secret",
    "secret_key",
    "password",
    "token",
    "private_key",
    "account_id",
}


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): "[REDACTED]" if str(k).lower() in _FORBIDDEN_KEYS else _sanitize(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize(v) for v in value]
    return value


def record_validation_evidence(
    path: str,
    *,
    scenario: str,
    commit_sha: str,
    environment: str,
    expected_result: str,
    observed_result: str,
    status: str,
    timestamp: str,
    operator: str = "",
    evidence: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append one sanitized validation record to a JSONL file."""
    if not scenario.strip():
        raise ValidationEvidenceError("scenario is required")
    if status.upper() not in {"PASS", "FAIL", "BLOCKED", "NOT_RUN"}:
        raise ValidationEvidenceError("status must be PASS, FAIL, BLOCKED, or NOT_RUN")

    record = {
        "scenario": scenario.strip(),
        "commit_sha": commit_sha.strip(),
        "environment": environment.strip(),
        "expected_result": expected_result.strip(),
        "observed_result": observed_result.strip(),
        "status": status.upper(),
        "timestamp": timestamp.strip(),
        "operator": operator.strip(),
        "evidence": evidence.strip(),
        "metadata": _sanitize(metadata or {}),
    }

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
