from scripts.exchange_preflight import run_exchange_preflight


class FakeExchange:
    def load_markets(self):
        return {"BTC/USDT": {"active": True, "spot": True}}


def test_preflight_is_read_only_and_reports_market():
    result = run_exchange_preflight(
        FakeExchange(),
        ["BTC/USDT", "ETH/USDT"],
        exchange_name="fake",
        sandbox=True,
    )
    assert result.markets_loaded is True
    assert result.sandbox is True
    assert result.results[0].eligible is True
    assert result.results[1].eligible is False
