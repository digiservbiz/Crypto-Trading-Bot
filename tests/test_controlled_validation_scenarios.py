from scripts.controlled_executor import ControlledExecutor
from scripts.kill_switch import KillSwitch
from scripts.persistent_execution_ledger import PersistentExecutionLedger
from scripts.agents.base_agent import RiskDecision


class Exchange:
    def __init__(self, order=None, raises=False):
        self.order = order
        self.raises = raises
        self.calls = 0

    def create_order(self, symbol, order_type, side, amount):
        self.calls += 1
        if self.raises:
            raise RuntimeError("broker timeout")
        return self.order


def approved():
    return RiskDecision(
        signal_id="scenario-sig",
        decision="approve",
        adjusted_size_pct=0.05,
        var_95=0.01,
        cvar_95=0.02,
        sharpe_ratio=1.0,
        circuit_breaker_triggered=False,
        circuit_breaker_reason="",
        timestamp=100.0,
    )


def executor(tmp_path, exchange):
    return ControlledExecutor(
        exchange,
        kill_switch=KillSwitch(str(tmp_path / "stop")),
        ledger=PersistentExecutionLedger(str(tmp_path / "ledger.sqlite3")),
    )


def test_timeout_remains_unknown_and_key_is_retained(tmp_path):
    exchange = Exchange(raises=True)
    ledger_path = tmp_path / "ledger.sqlite3"
    ex = ControlledExecutor(
        exchange,
        kill_switch=KillSwitch(str(tmp_path / "stop")),
        ledger=PersistentExecutionLedger(str(ledger_path)),
    )

    result = ex.execute(approved(), "BTC/USDT", "buy", 50000, 1000, 100)

    assert result.state == "unknown"
    assert exchange.calls == 1

    # A new process instance must still reject the same execution request.
    ex2 = ControlledExecutor(
        Exchange({"id": "should-not-submit", "status": "closed", "filled": 1.0}),
        kill_switch=KillSwitch(str(tmp_path / "stop")),
        ledger=PersistentExecutionLedger(str(ledger_path)),
    )
    duplicate = ex2.execute(approved(), "BTC/USDT", "buy", 50000, 1000, 101)
    assert duplicate.duplicate is True
    assert ex2.exchange.calls == 0


def test_kill_switch_blocks_before_broker_submission(tmp_path):
    exchange = Exchange({"id": "never", "status": "closed", "filled": 1.0})
    stop = KillSwitch(str(tmp_path / "stop"))
    stop.activate()
    ex = ControlledExecutor(
        exchange,
        kill_switch=stop,
        ledger=PersistentExecutionLedger(str(tmp_path / "ledger.sqlite3")),
    )

    try:
        ex.execute(approved(), "BTC/USDT", "buy", 50000, 1000, 100)
    except RuntimeError as exc:
        assert "kill switch" in str(exc).lower()
    else:
        raise AssertionError("active kill switch must block execution")

    assert exchange.calls == 0


def test_partial_fill_stays_unresolved(tmp_path):
    exchange = Exchange({"id": "partial", "status": "partially_filled", "filled": 0.25, "remaining": 0.75})
    result = executor(tmp_path, exchange).execute(
        approved(), "BTC/USDT", "buy", 50000, 1000, 100
    )
    assert result.state == "unresolved"
    assert result.filled_amount == 0.25
