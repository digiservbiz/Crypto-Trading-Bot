import pytest

from scripts.order_reconciliation import reconcile_order
from scripts.safe_order_reconciliation import safe_reconcile_order


def test_malformed_filled_is_rejected():
    with pytest.raises(ValueError):
        safe_reconcile_order({"id": "1", "status": "open", "filled": "nan"}, 1)


def test_negative_remaining_is_rejected():
    with pytest.raises(ValueError):
        safe_reconcile_order({"id": "1", "status": "open", "filled": 0, "remaining": -1}, 1)


def test_invalid_requested_amount_is_rejected():
    with pytest.raises(ValueError):
        safe_reconcile_order({"id": "1", "status": "open"}, 0)


def test_existing_reconciler_still_handles_valid_order():
    result = reconcile_order(
        {"id": "1", "status": "closed", "filled": 2, "remaining": 0}, 2
    )
    assert result.is_fully_filled is True
