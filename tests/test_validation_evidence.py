from pathlib import Path

import pytest

from scripts.validation_evidence import (
    ValidationEvidenceError,
    record_validation_evidence,
)


def test_record_validation_evidence_appends_jsonl_and_sanitizes_secrets(tmp_path):
    path = tmp_path / "evidence.jsonl"

    record_validation_evidence(
        str(path),
        scenario="duplicate execution",
        commit_sha="abc123",
        environment="paper",
        expected_result="second claim rejected",
        observed_result="second claim rejected",
        status="PASS",
        timestamp="2026-10-04T00:00:00+02:00",
        metadata={
            "exchange": "sandbox",
            "api_key": "do-not-store",
            "nested": {"secret_key": "also-do-not-store"},
        },
    )

    line = path.read_text(encoding="utf-8").strip()
    assert '"api_key": "[REDACTED]"' in line
    assert '"secret_key": "[REDACTED]"' in line
    assert "do-not-store" not in line


def test_record_validation_evidence_rejects_invalid_status(tmp_path):
    with pytest.raises(ValidationEvidenceError):
        record_validation_evidence(
            str(tmp_path / "evidence.jsonl"),
            scenario="test",
            commit_sha="abc",
            environment="paper",
            expected_result="x",
            observed_result="y",
            status="MAYBE",
            timestamp="now",
        )
