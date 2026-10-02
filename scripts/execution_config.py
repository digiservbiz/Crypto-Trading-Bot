"""Validation for trading execution configuration.

This module keeps deployment invariants independent from strategy code.
"""

from __future__ import annotations

from dataclasses import dataclass


class ExecutionConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ExecutionConfig:
    max_position_pct: float = 0.10
    max_approval_age_seconds: float = 300.0
    sandbox: bool = True
    market_mode: str = "spot"

    def validate(self) -> "ExecutionConfig":
        if not 0 < self.max_position_pct <= 0.10:
            raise ExecutionConfigError("max_position_pct must be > 0 and <= 0.10")
        if not 0 < self.max_approval_age_seconds <= 300:
            raise ExecutionConfigError(
                "max_approval_age_seconds must be > 0 and <= 300"
            )
        if self.market_mode not in {"spot", "futures"}:
            raise ExecutionConfigError("market_mode must be spot or futures")
        return self
