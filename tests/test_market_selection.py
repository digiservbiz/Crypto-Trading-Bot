from scripts.market_selection import configured_symbols, normalize_symbols


def test_normalize_symbols_deduplicates_and_normalizes():
    assert normalize_symbols(["btc/usdt", "ETH/USDT", "BTC/USDT"]) == [
        "BTC/USDT", "ETH/USDT"
    ]


def test_normalize_symbols_rejects_invalid():
    try:
        normalize_symbols(["BTC-USDT"])
    except ValueError:
        pass
    else:
        raise AssertionError("invalid symbol must be rejected")


def test_configured_symbols_uses_data_symbols():
    assert configured_symbols({"data": {"symbols": ["sol/usdt"]}}) == ["SOL/USDT"]
