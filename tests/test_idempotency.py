import pytest

from scripts.idempotency import build_execution_key


def test_same_execution_inputs_produce_same_key():
    assert build_execution_key("sig-123", "BTC/USDT", "buy") == build_execution_key(
        "sig-123", "BTC/USDT", "buy"
    )


def test_different_signal_produces_different_key():
    assert build_execution_key("sig-123", "BTC/USDT", "buy") != build_execution_key(
        "sig-124", "BTC/USDT", "buy"
    )


@pytest.mark.parametrize("side", ["", "hold", "BUY "])
def test_invalid_side_rejected(side):
    with pytest.raises(ValueError):
        build_execution_key("sig-123", "BTC/USDT", side)


def test_empty_signal_rejected():
    with pytest.raises(ValueError):
        build_execution_key("", "BTC/USDT", "buy")
