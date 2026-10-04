"""Summarize append-only validation evidence without exposing secrets.

The report is intentionally read-only: it never changes trading state and it
does not infer PASS from the existence of a scenario row.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

VALID_STATUSES = {"PASS", "FAIL", "BLOCKED", "NOT_RUN"}


def load_validation_evidence(path: str) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"validation record at line {line_number} is not an object")
        records.append(record)
    return records


def summarize_validation_evidence(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {status: 0 for status in VALID_STATUSES}
    scenarios: dict[str, str] = {}

    for record in records:
        scenario = str(record.get("scenario") or "").strip()
        status = str(record.get("status") or "NOT_RUN").upper()
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid validation status: {status}")
        if not scenario:
            raise ValueError("validation record is missing scenario")
        counts[status] += 1
        scenarios[scenario] = status

    return {
        "total_records": len(records),
        "counts": counts,
        "scenarios": scenarios,
        "release_blocked": (
            counts["FAIL"] > 0
            or counts["BLOCKED"] > 0
            or counts["NOT_RUN"] > 0
        ),
    }
