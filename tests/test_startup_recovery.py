from scripts.startup_recovery import recover_startup_state, require_startup_recovery


class FakeExchange:
    def __init__(self, positions=None, error=False):
        self.positions = positions
        self.error = error

    def fetch_positions(self, symbols):
        if self.error:
            raise RuntimeError("exchange unavailable")
        return self.positions


def test_startup_recovery_unblocks_after_valid_exchange_snapshot():
    state = recover_startup_state(
        FakeExchange([{"symbol": "BTC/USDT", "side": "long", "contracts": 1, "entryPrice": 50000}]),
        ["BTC/USDT"],
    )
    assert state.trading_blocked is False
    assert len(state.positions) == 1
    require_startup_recovery(state)


def test_startup_recovery_blocks_on_exchange_failure():
    state = recover_startup_state(FakeExchange(error=True), ["BTC/USDT"])
    assert state.trading_blocked is True
    try:
        require_startup_recovery(state)
    except RuntimeError as exc:
        assert "startup recovery required" in str(exc)
    else:
        raise AssertionError("failed recovery must block trading")


def test_startup_recovery_blocks_when_endpoint_is_unsupported():
    class Unsupported:
        pass

    state = recover_startup_state(Unsupported(), ["BTC/USDT"])
    assert state.trading_blocked is True
