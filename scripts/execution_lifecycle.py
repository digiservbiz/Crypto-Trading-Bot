"""Conservative execution lifecycle decisions.

This module keeps broker outcomes separate from strategy state. A submitted
order is only considered safe to record as an entry when reconciliation proves
a positive fill. Unknown/partial outcomes remain unresolved.
"""

from __future__ import annotations

from dataclasses import dataclass

from .order_reconciliation import OrderReconciliation


@dataclass(frozen=True)
class ExecutionOutcome:
    state: str
    filled_amount: float
    order_id: str
    should_open_position: bool
    should_retry: bool


def classify_execution(reconciliation: OrderReconciliation) -> ExecutionOutcome:
    if reconciliation.is_fully_filled:
        return ExecutionOutcome(
            state="filled",
            filled_amount=reconciliation.filled_amount,
            order_id=reconciliation.order_id,
            should_open_position=reconciliation.filled_amount > 0,
            should_retry=False,
        )

    if reconciliation.is_terminal_failure:
        return ExecutionOutcome(
            state="failed",
            filled_amount=reconciliation.filled_amount,
            order_id=reconciliation.order_id,
            should_open_position=False,
            should_retry=False,
        )

    if reconciliation.is_open or reconciliation.filled_amount > 0:
        return ExecutionOutcome(
            state="unresolved",
            filled_amount=reconciliation.filled_amount,
            order_id=reconciliation.order_id,
            should_open_position=False,
            should_retry=False,
        )

    return ExecutionOutcome(
        state="unknown",
        filled_amount=reconciliation.filled_amount,
        order_id=reconciliation.order_id,
        should_open_position=False,
        should_retry=False,
    )
