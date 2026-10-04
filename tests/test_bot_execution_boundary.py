from scripts.bot import execute_trade
from scripts.agents.base_agent import RiskDecision


class FakeNotifier:
    def __init__(self):
        self.messages = []

    def send_message(self, message):
        self.messages.append(message)


class FakeExecutor:
    def __init__(self, state="filled"):
        self.state = state
        self.calls = []

    def execute(self, risk_decision, symbol, side, price, balance, now):
        self.calls.append((risk_decision, symbol, side, price, balance, now))
        return type(
            "Result",
            (),
            {
                "state": self.state,
                "order_id": "order-1",
                "filled_amount": 0.02,
            },
        )()


def approved():
    return RiskDecision(
        signal_id="sig-1",
        decision="approve",
        adjusted_size_pct=0.05,
        var_95=0.01,
        cvar_95=0.02,
        sharpe_ratio=1.0,
        circuit_breaker_triggered=False,
        circuit_breaker_reason="",
        timestamp=100.0,
    )


def test_live_execute_trade_requires_controlled_executor():
    notifier = FakeNotifier()
    assert execute_trade(
        exchange=None,
        notifier=notifier,
        risk_decision=approved(),
        symbol="BTC/USDT",
        side="buy",
        current_price=50000,
        balance=1000,
        dry_run=False,
    ) is None


def test_live_execute_trade_uses_controlled_executor():
    notifier = FakeNotifier()
    executor = FakeExecutor("filled")
    amount = execute_trade(
        exchange=None,
        notifier=notifier,
        risk_decision=approved(),
        symbol="BTC/USDT",
        side="buy",
        current_price=50000,
        balance=1000,
        dry_run=False,
        controlled_executor=executor,
    )
    assert amount == 0.02
    assert len(executor.calls) == 1


def test_unresolved_execution_does_not_advance_position_amount():
    notifier = FakeNotifier()
    executor = FakeExecutor("unresolved")
    amount = execute_trade(
        exchange=None,
        notifier=notifier,
        risk_decision=approved(),
        symbol="BTC/USDT",
        side="buy",
        current_price=50000,
        balance=1000,
        dry_run=False,
        controlled_executor=executor,
    )
    assert amount is None
