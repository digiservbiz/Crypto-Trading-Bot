"""Tests for conservative order lifecycle reconciliation."""

import pytest

from scripts.order_reconciliation import reconcile_order


def test_closed_order_is_fully_filled():
    result = reconcile_order(
        {"id": "1", "status": "closed", "filled": 2.0, "remaining": 0.0},
        2.0,
    )
    assert result.is_fully_filled
    assert not result.is_open
    assert not result.is_terminal_failure


def test_open_order_is_not_treated_as_filled():
    result = reconcile_order(
        {"id": "2", "status": "open", "filled": 0.5, "remaining": 1.5},
        2.0,
    )
    assert not result.is_fully_filled
    assert result.is_open
    assert result.remaining_amount == pytest.approx(1.5)


def test_partial_fill_is_not_fully_filled():
    result = reconcile_order(
        {"id": "3", "status": "partially_filled", "filled": 0.75},
        2.0,
    )
    assert not result.is_fully_filled
    assert result.is_open
    assert result.remaining_amount == pytest.approx(1.25)


def test_rejected_order_is_terminal_failure():
    result = reconcile_order(
        {"id": "4", "status": "canceled", "filled": 0.0, "remaining": 2.0},
        2.0,
    )
    assert result.is_terminal_failure
    assert not result.is_fully_filled


def test_unknown_status_is_conservative():
    result = reconcile_order(
        {"id": "5", "status": "unknown", "filled": 0.0},
        2.0,
    )
    assert not result.is_fully_filled
    assert not result.is_terminal_failure
    assert not result.is_open
