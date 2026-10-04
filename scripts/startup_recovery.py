"""Conservative startup reconciliation policy.

A restarted process must not invent local positions. If authoritative exchange
recovery fails, trading remains blocked until an operator explicitly resolves
the state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .position_recovery import RecoveredPosition, recover_positions


@dataclass(frozen=True)
class StartupState:
    positions: tuple[RecoveredPosition, ...]
    trading_blocked: bool
    reason: str


def recover_startup_state(exchange: Any, symbols: list[str]) -> StartupState:
    positions = tuple(recover_positions(exchange, symbols))
    fetch_positions = getattr(exchange, "fetch_positions", None)
    if not callable(fetch_positions):
        return StartupState(positions, True, "exchange position recovery unsupported")
    try:
        raw = fetch_positions(symbols)
    except Exception:
        return StartupState(positions, True, "exchange position recovery failed")
    if not isinstance(raw, list):
        return StartupState(positions, True, "exchange returned invalid position data")
    return StartupState(positions, False, "exchange position state recovered")


def require_startup_recovery(state: StartupState) -> None:
    if state.trading_blocked:
        raise RuntimeError(f"startup recovery required: {state.reason}")
