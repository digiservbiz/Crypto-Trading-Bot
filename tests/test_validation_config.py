from scripts.validation_config import ValidationConfigError, validate_controlled_validation_config


def test_controlled_validation_requires_sandbox():
    config = {"exchange": {"sandbox": True}, "execution": {"market_mode": "spot"}, "data": {"symbols": ["BTC/USDT"]}}
    validate_controlled_validation_config(config)


def test_controlled_validation_rejects_non_sandbox():
    config = {"exchange": {"sandbox": False}, "execution": {"market_mode": "spot"}, "data": {"symbols": ["BTC/USDT"]}}
    try:
        validate_controlled_validation_config(config)
    except ValidationConfigError as exc:
        assert "sandbox" in str(exc)
    else:
        raise AssertionError("non-sandbox configuration must fail")


def test_controlled_validation_requires_markets():
    config = {"exchange": {"sandbox": True}, "execution": {"market_mode": "spot"}, "data": {"symbols": []}}
    try:
        validate_controlled_validation_config(config)
    except ValidationConfigError as exc:
        assert "symbols" in str(exc)
    else:
        raise AssertionError("empty market universe must fail")
