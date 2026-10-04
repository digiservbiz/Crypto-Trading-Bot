import pytest

from scripts.execution_config import ExecutionConfig, ExecutionConfigError


def test_defaults_are_conservative():
    assert ExecutionConfig().validate().sandbox is True
    assert ExecutionConfig().max_position_pct == 0.10


@pytest.mark.parametrize("value", [0, -0.1, 0.100001, 1])
def test_position_cap_rejected(value):
    with pytest.raises(ExecutionConfigError):
        ExecutionConfig(max_position_pct=value).validate()


@pytest.mark.parametrize("value", [0, -1, 300.1])
def test_approval_age_rejected(value):
    with pytest.raises(ExecutionConfigError):
        ExecutionConfig(max_approval_age_seconds=value).validate()


def test_market_mode_is_explicit():
    assert ExecutionConfig(market_mode="futures").validate().market_mode == "futures"
    with pytest.raises(ExecutionConfigError):
        ExecutionConfig(market_mode="margin").validate()
