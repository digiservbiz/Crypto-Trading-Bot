from scripts.dashboard_markets import market_snapshot


def test_market_snapshot_handles_missing_market_state():
    snap = market_snapshot({}, "SOL/USDT")
    assert snap["symbol"] == "SOL/USDT"
    assert snap["price"] == 0.0
    assert snap["regime"] == "unknown"
    assert snap["side"] is None


def test_market_snapshot_extracts_market_state():
    state = {
        "current_prices": {"SOL/USDT": 150.0},
        "regimes": {"SOL/USDT": {"regime": "bull-trending", "confidence": 0.8}},
        "positions": {"SOL/USDT": {"side": "buy", "entry_price": 145.0, "pnl_pct": 0.03, "size_pct": 0.05}},
    }
    snap = market_snapshot(state, "SOL/USDT")
    assert snap["price"] == 150.0
    assert snap["regime"] == "bull-trending"
    assert snap["confidence"] == 0.8
    assert snap["side"] == "buy"
