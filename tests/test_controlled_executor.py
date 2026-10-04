from scripts.controlled_executor import ControlledExecutor
from scripts.kill_switch import KillSwitch
from scripts.persistent_execution_ledger import PersistentExecutionLedger
from scripts.agents.base_agent import RiskDecision


class FakeExchange:
    def __init__(self, order):
        self.order = order
        self.calls = 0

    def create_order(self, symbol, order_type, side, amount):
        self.calls += 1
        return self.order


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


def test_controlled_executor_reconciles_full_fill(tmp_path):
    exchange = FakeExchange({"id": "o1", "status": "closed", "filled": 1.0, "remaining": 0.0})
    executor = ControlledExecutor(
        exchange,
        kill_switch=KillSwitch(str(tmp_path / "stop")),
        ledger=PersistentExecutionLedger(str(tmp_path / "ledger.sqlite3")),
    )
    result = executor.execute(approved(), "BTC/USDT", "buy", 50000, 1000, 100)
    assert result.state == "filled"
    assert result.order_id == "o1"
    assert exchange.calls == 1


def test_controlled_executor_blocks_duplicate(tmp_path):
    exchange = FakeExchange({"id": "o1", "status": "closed", "filled": 1.0, "remaining": 0.0})
    executor = ControlledExecutor(
        exchange,
        kill_switch=KillSwitch(str(tmp_path / "stop")),
        ledger=PersistentExecutionLedger(str(tmp_path / "ledger.sqlite3")),
    )
    first = executor.execute(approved(), "BTC/USDT", "buy", 50000, 1000, 100)
    second = executor.execute(approved(), "BTC/USDT", "buy", 50000, 1000, 101)
    assert first.state == "filled"
    assert second.duplicate is True
    assert exchange.calls == 1


def test_controlled_executor_does_not_clear_ambiguous_claim(tmp_path):
    exchange = FakeExchange({"id": "o1", "status": "open", "filled": 0.2, "remaining": 0.8})
    ledger = PersistentExecutionLedger(str(tmp_path / "ledger.sqlite3"))
    executor = ControlledExecutor(
        exchange,
        kill_switch=KillSwitch(str(tmp_path / "stop")),
        ledger=ledger,
    )
    result = executor.execute(approved(), "BTC/USDT", "buy", 50000, 1000, 100)
    assert result.state == "unresolved"
    assert ledger.contains("missing") is False
    assert exchange.calls == 1


def test_controlled_executor_fetches_authoritative_state_for_open_submission(tmp_path):
    class FetchingExchange(FakeExchange):
        def fetch_order(self, order_id, symbol):
            return {"id": order_id, "status": "closed", "filled": 1.0, "remaining": 0.0}

    exchange = FetchingExchange({"id": "o2", "status": "open", "filled": 0.0, "remaining": 1.0})
    executor = ControlledExecutor(
        exchange,
        kill_switch=KillSwitch(str(tmp_path / "stop")),
        ledger=PersistentExecutionLedger(str(tmp_path / "ledger.sqlite3")),
    )
    result = executor.execute(approved(), "BTC/USDT", "buy", 50000, 1000, 100)
    assert result.state == "filled"
    assert result.order_id == "o2"
    assert exchange.calls == 1


def test_controlled_executor_keeps_unresolved_when_fetch_is_unavailable(tmp_path):
    exchange = FakeExchange({"id": "o3", "status": "open", "filled": 0.2, "remaining": 0.8})
    executor = ControlledExecutor(
        exchange,
        kill_switch=KillSwitch(str(tmp_path / "stop")),
        ledger=PersistentExecutionLedger(str(tmp_path / "ledger.sqlite3")),
    )
    result = executor.execute(approved(), "BTC/USDT", "buy", 50000, 1000, 100)
    assert result.state == "unresolved"
    assert result.order_id == "o3"
