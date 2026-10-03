from scripts.order_reconciliation import reconcile_order


def test_close_order_with_partial_fill_remains_unresolved():
    result = reconcile_order(
        {"id": "close-1", "status": "partially_filled", "filled": 0.4, "remaining": 0.6},
        1.0,
    )
    assert result.is_open is True
    assert result.is_fully_filled is False
    assert result.is_terminal_failure is False


def test_close_order_rejected_is_terminal_failure():
    result = reconcile_order(
        {"id": "close-2", "status": "canceled", "filled": 0.0, "remaining": 1.0},
        1.0,
    )
    assert result.is_terminal_failure is True
    assert result.is_fully_filled is False


def test_malformed_nan_fill_is_rejected_by_safe_layer():
    from scripts.safe_order_reconciliation import safe_reconcile_order
    import pytest

    with pytest.raises(ValueError):
        safe_reconcile_order(
            {"id": "bad", "status": "closed", "filled": float("nan"), "remaining": 0},
            1.0,
        )
