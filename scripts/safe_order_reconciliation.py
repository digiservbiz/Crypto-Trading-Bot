"""Stronger normalization for authoritative exchange order snapshots."""

from __future__ import annotations

import math
from typing import Any

from .order_reconciliation import OrderReconciliation


def _finite_nonnegative(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"invalid {field}")
    return number


def safe_reconcile_order(order: dict[str, Any], requested_amount: float) -> OrderReconciliation:
    """Reject malformed broker quantities before they can affect local state."""
    requested = _finite_nonnegative(requested_amount, "requested_amount")
    if requested <= 0:
        raise ValueError("requested_amount must be positive")
    return OrderReconciliation(
        order_id=str(order.get("id") or ""),
        status=str(order.get("status") or "unknown").lower(),
        requested_amount=requested,
        filled_amount=_finite_nonnegative(order.get("filled", 0.0), "filled"),
        remaining_amount=_finite_nonnegative(
            order.get("remaining", max(requested - _finite_nonnegative(order.get("filled", 0.0), "filled"), 0.0)),
            "remaining",
        ),
        is_fully_filled=False,
        is_terminal_failure=False,
        is_open=False,
    )
