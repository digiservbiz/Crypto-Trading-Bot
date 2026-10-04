"""Configuration guards for controlled testnet/paper validation."""

from __future__ import annotations


class ValidationConfigError(ValueError):
    pass


def validate_controlled_validation_config(config: dict) -> None:
    exchange = config.get("exchange", {})
    execution = config.get("execution", {})
    if not bool(exchange.get("sandbox", False)):
        raise ValidationConfigError("controlled validation requires exchange.sandbox=true")
    if execution.get("market_mode", "spot") not in {"spot", "futures"}:
        raise ValidationConfigError("execution.market_mode must be spot or futures")
    symbols = config.get("data", {}).get("symbols", [])
    if not isinstance(symbols, list) or not symbols:
        raise ValidationConfigError("controlled validation requires configured symbols")
