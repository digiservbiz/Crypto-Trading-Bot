from scripts.validation_report import summarize_validation_evidence


def test_summary_blocks_release_until_all_scenarios_pass():
    summary = summarize_validation_evidence([
        {"scenario": "full fill", "status": "PASS"},
        {"scenario": "partial fill", "status": "NOT_RUN"},
    ])

    assert summary["counts"]["PASS"] == 1
    assert summary["counts"]["NOT_RUN"] == 1
    assert summary["release_blocked"] is True


def test_summary_is_clear_when_all_recorded_scenarios_pass():
    summary = summarize_validation_evidence([
        {"scenario": "full fill", "status": "PASS"},
        {"scenario": "kill switch", "status": "PASS"},
    ])

    assert summary["release_blocked"] is False
