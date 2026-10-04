from scripts.market_eligibility import check_market_eligibility


def test_spot_market_must_exist_and_be_active():
    results = check_market_eligibility(
        {
            "BTC/USDT": {"active": True, "spot": True},
            "ETH/USDT": {"active": False, "spot": True},
            "SOL/USDT": {"active": True, "spot": True},
        },
        ["BTC/USDT", "ETH/USDT", "DOGE/USDT"],
    )
    assert [r.eligible for r in results] == [True, False, False]
    assert results[1].reason == "market inactive"
    assert results[2].reason == "market not found"


def test_futures_market_requires_futures_or_swap_flag():
    results = check_market_eligibility(
        {
            "BTC/USDT": {"active": True, "spot": True},
            "BTC/USDT:USDT": {"active": True, "swap": True},
        },
        ["BTC/USDT", "BTC/USDT:USDT"],
        market_mode="futures",
    )
    assert results[0].eligible is False
    assert results[1].eligible is True


def test_invalid_market_mode_rejected():
    try:
        check_market_eligibility({}, ["BTC/USDT"], market_mode="margin")
    except ValueError as exc:
        assert "market_mode" in str(exc)
    else:
        raise AssertionError("invalid market mode must fail")
