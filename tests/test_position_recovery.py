from scripts.position_recovery import normalize_position


def test_normalizes_long_position():
    position = normalize_position({
        "symbol": "BTC/USDT:USDT",
        "side": "long",
        "contracts": 0.25,
        "entryPrice": 60000,
    })
    assert position is not None
    assert position.symbol == "BTC/USDT:USDT"
    assert position.side == "long"
    assert position.amount == 0.25
    assert position.entry_price == 60000


def test_ignores_zero_position():
    assert normalize_position({"symbol": "BTC/USDT", "side": "long", "contracts": 0}) is None


def test_ignores_invalid_side():
    assert normalize_position({"symbol": "BTC/USDT", "side": "sell", "contracts": 1}) is None


def test_invalid_entry_price_becomes_unknown():
    position = normalize_position({
        "symbol": "BTC/USDT",
        "side": "short",
        "contracts": 1,
        "entryPrice": -1,
    })
    assert position is not None
    assert position.entry_price is None
