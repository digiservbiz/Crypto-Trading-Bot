"""Controlled broker execution boundary.

Composes the final safety gate, kill switch, durable idempotency claim, broker
submission, and authoritative reconciliation. Strategy code should provide an
already-approved RiskDecision and must not mutate it after this point.

This adapter deliberately does not retry ambiguous submissions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .execution_safety import ExecutionPolicy, build_trade_intent, validate_trade_intent
from .idempotency import build_execution_key
from .kill_switch import KillSwitch
from .order_reconciliation import fetch_order_reconciliation, reconcile_order
from .persistent_execution_ledger import PersistentExecutionLedger


@dataclass(frozen=True)
class ControlledExecutionResult:
    state: str
    order_id: str
    requested_amount: float
    filled_amount: float
    duplicate: bool = False


class ControlledExecutor:
    def __init__(
        self,
        exchange: Any,
        *,
        policy: ExecutionPolicy = ExecutionPolicy(),
        kill_switch: KillSwitch | None = None,
        ledger: PersistentExecutionLedger | None = None,
    ) -> None:
        self.exchange = exchange
        self.policy = policy
        self.kill_switch = kill_switch or KillSwitch()
        self.ledger = ledger or PersistentExecutionLedger()

    def execute(self, risk_decision, symbol: str, side: str, price: float, balance: float, now: float) -> ControlledExecutionResult:
        self.kill_switch.require_clear()

        intent = build_trade_intent(
            risk_decision, symbol, side, price, balance, now=now
        )
        amount = validate_trade_intent(
            intent, risk_decision, self.policy, now=now
        )

        key = build_execution_key(risk_decision.signal_id, symbol, side)
        if not self.ledger.claim(key, now):
            return ControlledExecutionResult("duplicate", "", amount, 0.0, True)

        try:
            order = self.exchange.create_order(symbol, "market", side, amount)
        except Exception:
            # Keep the claim: a broker exception may have occurred after submission.
            return ControlledExecutionResult("unknown", "", amount, 0.0)

        if not isinstance(order, dict) or not order.get("id"):
            return ControlledExecutionResult("unknown", "", amount, 0.0)

        reconciliation = reconcile_order(order, amount)

        if not reconciliation.is_fully_filled and not reconciliation.is_terminal_failure:
            authoritative = fetch_order_reconciliation(self.exchange, reconciliation.order_id, symbol, amount)
            if authoritative is not None:
                reconciliation = authoritative

        if reconciliation.is_fully_filled:
            return ControlledExecutionResult(
                "filled",
                reconciliation.order_id,
                amount,
                reconciliation.filled_amount,
            )
        if reconciliation.is_terminal_failure:
            self.ledger.release(key)
            return ControlledExecutionResult(
                "failed",
                reconciliation.order_id,
                amount,
                reconciliation.filled_amount,
            )
        return ControlledExecutionResult(
            "unresolved",
            reconciliation.order_id,
            amount,
            reconciliation.filled_amount,
        )
