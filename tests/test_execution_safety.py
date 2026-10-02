"""Tests for the final execution safety boundary."""

import pytest

from scripts.agents.base_agent import RiskDecision
from scripts.execution_safety import (
    ExecutionPolicy,
    ExecutionSafetyError,
    TradeIntent,
    build_trade_intent,
    validate_trade_intent,
)


def approved(**overrides):
    values = dict(
        signal_id="sig-1",
        decision="approve",
        adjusted_size_pct=0.05,
        var_95=0.01,
        cvar_95=0.02,
        sharpe_ratio=1.5,
        circuit_breaker_triggered=False,
        circuit_breaker_reason=None,
        timestamp=1000.0,
    )
    values.update(overrides)
    return RiskDecision(**values)


def test_valid_approval_produces_expected_amount():
    decision = approved()
    intent = build_trade_intent(decision, "BTC/USDT", "buy", 100_000, 10_000)
    amount = validate_trade_intent(intent, decision, now=1100)
    assert amount == pytest.approx(0.005)


def test_rejected_risk_decision_is_blocked():
    decision = approved(decision="reject")
    intent = build_trade_intent(decision, "BTC/USDT", "buy", 100_000, 10_000)
    with pytest.raises(ExecutionSafetyError, match="not approved"):
        validate_trade_intent(intent, decision, now=1100)


def test_size_mutation_after_approval_is_blocked():
    decision = approved(adjusted_size_pct=0.05)
    intent = TradeIntent(
        signal_id="sig-1",
        symbol="BTC/USDT",
        side="buy",
        current_price=100_000,
        balance=10_000,
        size_pct=0.075,
        timestamp=1000,
    )
    with pytest.raises(ExecutionSafetyError, match="changed"):
        validate_trade_intent(intent, decision, now=1100)


def test_hard_position_cap_is_enforced():
    decision = approved(adjusted_size_pct=0.11)
    intent = build_trade_intent(decision, "BTC/USDT", "buy", 100_000, 10_000)
    with pytest.raises(ExecutionSafetyError, match="hard cap"):
        validate_trade_intent(
            intent,
            decision,
            policy=ExecutionPolicy(max_position_pct=0.10),
            now=1100,
        )


def test_stale_approval_is_blocked():
    decision = approved(timestamp=1000)
    intent = build_trade_intent(decision, "BTC/USDT", "buy", 100_000, 10_000)
    with pytest.raises(ExecutionSafetyError, match="stale"):
        validate_trade_intent(
            intent,
            decision,
            policy=ExecutionPolicy(max_approval_age_seconds=300),
            now=1301,
        )


def test_signal_mismatch_is_blocked():
    decision = approved(signal_id="approved-signal")
    intent = TradeIntent(
        signal_id="different-signal",
        symbol="BTC/USDT",
        side="buy",
        current_price=100_000,
        balance=10_000,
        size_pct=0.05,
        timestamp=1000,
    )
    with pytest.raises(ExecutionSafetyError, match="signal_id"):
        validate_trade_intent(intent, decision, now=1100)


@pytest.mark.parametrize("side", ["hold", "BUY ", ""])
def test_invalid_side_is_blocked(side):
    decision = approved()
    intent = TradeIntent(
        signal_id="sig-1",
        symbol="BTC/USDT",
        side=side.lower(),
        current_price=100_000,
        balance=10_000,
        size_pct=0.05,
        timestamp=1000,
    )
    with pytest.raises(ExecutionSafetyError, match="invalid side"):
        validate_trade_intent(intent, decision, now=1100)
