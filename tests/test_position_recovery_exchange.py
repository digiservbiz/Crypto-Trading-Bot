from scripts.position_recovery import recover_positions


class FakeExchange:
    def __init__(self, positions):
        self.positions = positions

    def fetch_positions(self, symbols):
        return self.positions


def test_recover_positions_normalizes_exchange_state():
    exchange = FakeExchange([
        {"symbol": "BTC/USDT", "side": "long", "contracts": 0.25, "entryPrice": 60000},
        {"symbol": "ETH/USDT", "side": "short", "amount": 2, "entryPrice": 3000},
    ])
    result = recover_positions(exchange, ["BTC/USDT", "ETH/USDT"])
    assert [(p.symbol, p.side, p.amount) for p in result] == [
        ("BTC/USDT", "long", 0.25),
        ("ETH/USDT", "short", 2.0),
    ]


def test_recover_positions_fails_closed_on_exchange_error():
    exchange = FakeExchange(None)
    def broken(_symbols):
        raise RuntimeError("temporary failure")
    exchange.fetch_positions = broken
    assert recover_positions(exchange, ["BTC/USDT"]) == []
