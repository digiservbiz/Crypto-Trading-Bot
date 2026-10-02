"""Final execution safety gate for live trading.

This module is deliberately independent of the strategy/risk agents.  It
validates the already-approved order immediately before an exchange call.
No research, strategy, or UI component may increase the approved size.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

from scripts.agents.base_agent import RiskDecision


@dataclass(frozen=True)
class TradeIntent:
    """Immutable order intent presented to the final execution gate."""

    signal_id: str
    symbol: str
    side: str
    current_price: float
    balance: float
    size_pct: float
    timestamp: float


@dataclass(frozen=True)
class ExecutionPolicy:
    """Hard limits enforced immediately before exchange submission."""

    max_position_pct: float = 0.10
    max_approval_age_seconds: float = 300.0
    min_price: float = 0.0


class ExecutionSafetyError(ValueError):
    """Raised when an order violates an execution invariant."""


def build_trade_intent(
    risk_decision: RiskDecision,
    symbol: str,
    side: str,
    current_price: float,
    balance: float,
    now: Optional[float] = None,
) -> TradeIntent:
    """Build an immutable intent from an approved risk decision."""
    return TradeIntent(
        signal_id=str(risk_decision.signal_id),
        symbol=symbol,
        side=side.lower(),
        current_price=float(current_price),
        balance=float(balance),
        size_pct=float(risk_decision.adjusted_size_pct),
        timestamp=float(risk_decision.timestamp if now is None else now),
    )


def validate_trade_intent(
    intent: TradeIntent,
    risk_decision: RiskDecision,
    policy: ExecutionPolicy = ExecutionPolicy(),
    now: Optional[float] = None,
) -> float:
    """Validate an approved intent and return the final base-asset amount.

    The returned amount is derived only from the risk-approved size.  Any
    attempt to mutate that size after approval must fail the gate.
    """
    current_time = time.time() if now is None else float(now)

    if not risk_decision.is_approved():
        raise ExecutionSafetyError("risk decision is not approved")

    if not intent.signal_id or intent.signal_id != risk_decision.signal_id:
        raise ExecutionSafetyError("signal_id mismatch")

    if intent.side not in {"buy", "sell"}:
        raise ExecutionSafetyError(f"invalid side: {intent.side}")

    if not intent.symbol or "/" not in intent.symbol:
        raise ExecutionSafetyError("invalid trading symbol")

    if not math.isfinite(intent.current_price) or intent.current_price <= policy.min_price:
        raise ExecutionSafetyError("invalid current price")

    if not math.isfinite(intent.balance) or intent.balance <= 0:
        raise ExecutionSafetyError("invalid balance")

    approved_size = float(risk_decision.adjusted_size_pct)
    intent_size = float(intent.size_pct)

    if not math.isfinite(approved_size) or approved_size <= 0:
        raise ExecutionSafetyError("invalid approved position size")

    if not math.isfinite(intent_size) or abs(intent_size - approved_size) > 1e-12:
        raise ExecutionSafetyError("trade size was changed after risk approval")

    if approved_size > policy.max_position_pct:
        raise ExecutionSafetyError(
            f"approved position size {approved_size:.6f} exceeds hard cap "
            f"{policy.max_position_pct:.6f}"
        )

    age = current_time - float(risk_decision.timestamp)
    if age < 0 or age > policy.max_approval_age_seconds:
        raise ExecutionSafetyError(f"risk approval is stale: age={age:.1f}s")

    amount = (intent.balance * approved_size) / intent.current_price
    if not math.isfinite(amount) or amount <= 0:
        raise ExecutionSafetyError("computed order amount is invalid")

    return amount
