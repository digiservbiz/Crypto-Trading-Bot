"""Order lifecycle reconciliation helpers.

The trading loop must not treat a successful submission response as proof that
an order is fully filled. This module normalizes exchange order status and
provides conservative decisions for local state management.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


TERMINAL_FILLED = {"closed", "filled"}
TERMINAL_REJECTED = {"canceled", "cancelled", "rejected", "expired"}
OPEN_STATUSES = {"open", "new", "partially_filled", "partial"}


@dataclass(frozen=True)
class OrderReconciliation:
    order_id: str
    status: str
    requested_amount: float
    filled_amount: float
    remaining_amount: float
    is_fully_filled: bool
    is_terminal_failure: bool
    is_open: bool


def reconcile_order(order: dict[str, Any], requested_amount: float) -> OrderReconciliation:
    """Normalize a CCXT order response without assuming submission == fill."""
    order_id = str(order.get("id") or "")
    status = str(order.get("status") or "unknown").lower()
    filled = float(order.get("filled") or 0.0)
    remaining_raw = order.get("remaining")
    remaining = (
        max(float(remaining_raw), 0.0)
        if remaining_raw is not None
        else max(float(requested_amount) - filled, 0.0)
    )

    fully_filled = status in TERMINAL_FILLED or (
        requested_amount > 0 and filled >= requested_amount and remaining <= 0
    )

    return OrderReconciliation(
        order_id=order_id,
        status=status,
        requested_amount=float(requested_amount),
        filled_amount=filled,
        remaining_amount=remaining,
        is_fully_filled=fully_filled,
        is_terminal_failure=status in TERMINAL_REJECTED,
        is_open=status in OPEN_STATUSES,
    )


def fetch_order_reconciliation(
    exchange: Any,
    order_id: str,
    symbol: str,
    requested_amount: float,
) -> Optional[OrderReconciliation]:
    """Fetch the authoritative order state when the exchange supports it."""
    if not order_id or not hasattr(exchange, "fetch_order"):
        return None
    try:
        order = exchange.fetch_order(order_id, symbol)
    except Exception:
        return None
    if not isinstance(order, dict):
        return None
    return reconcile_order(order, requested_amount)
