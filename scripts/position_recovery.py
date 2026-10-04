"""Restart recovery helpers for exchange-backed position state.

Local in-memory position state is not authoritative after a process restart.
These helpers normalize exchange position/balance snapshots so the bot can
rebuild conservative state before resuming trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RecoveredPosition:
    symbol: str
    side: str
    amount: float
    entry_price: float | None = None


def normalize_position(raw: dict[str, Any]) -> RecoveredPosition | None:
    """Normalize a CCXT-style position record.

    Unknown/zero-sized positions are ignored. Side is restricted to long/short
    semantics so malformed exchange data cannot become an executable position.
    """
    symbol = str(raw.get("symbol") or "")
    side = str(raw.get("side") or "").lower()
    amount_raw = raw.get("contracts", raw.get("amount", 0))
    try:
        amount = abs(float(amount_raw or 0))
    except (TypeError, ValueError):
        return None

    if not symbol or amount <= 0 or side not in {"long", "short"}:
        return None

    entry_raw = raw.get("entryPrice", raw.get("entry_price"))
    try:
        entry_price = float(entry_raw) if entry_raw is not None else None
    except (TypeError, ValueError):
        entry_price = None

    if entry_price is not None and entry_price <= 0:
        entry_price = None

    return RecoveredPosition(
        symbol=symbol,
        side=side,
        amount=amount,
        entry_price=entry_price,
    )


def recover_positions(exchange: Any, symbols: list[str]) -> list[RecoveredPosition]:
    """Fetch authoritative positions when supported by the exchange.

    Failure is fail-closed: no locally invented positions are returned.
    """
    fetch_positions = getattr(exchange, "fetch_positions", None)
    if not callable(fetch_positions):
        return []

    try:
        raw_positions = fetch_positions(symbols)
    except Exception:
        return []

    if not isinstance(raw_positions, list):
        return []

    recovered: list[RecoveredPosition] = []
    for raw in raw_positions:
        if isinstance(raw, dict):
            position = normalize_position(raw)
            if position is not None:
                recovered.append(position)
    return recovered
