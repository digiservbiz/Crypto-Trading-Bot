from scripts.execution_lifecycle import classify_execution
from scripts.order_reconciliation import reconcile_order


def test_only_full_fill_can_open_position():
    result = classify_execution(reconcile_order(
        {"id": "1", "status": "closed", "filled": 1, "remaining": 0}, 1
    ))
    assert result.state == "filled"
    assert result.should_open_position is True


def test_partial_fill_is_unresolved():
    result = classify_execution(reconcile_order(
        {"id": "2", "status": "open", "filled": 0.4, "remaining": 0.6}, 1
    ))
    assert result.state == "unresolved"
    assert result.should_open_position is False
    assert result.should_retry is False


def test_rejected_order_is_not_retried_automatically():
    result = classify_execution(reconcile_order(
        {"id": "3", "status": "rejected", "filled": 0, "remaining": 1}, 1
    ))
    assert result.state == "failed"
    assert result.should_retry is False
