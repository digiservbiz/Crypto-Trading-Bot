"""Dashboard helpers for a multi-market trading cockpit."""

from __future__ import annotations

from typing import Any


def market_snapshot(state: dict[str, Any], symbol: str) -> dict[str, Any]:
    """Build a defensive per-market dashboard snapshot."""
    regimes = state.get("regimes", {}) or {}
    positions = state.get("positions", {}) or {}
    prices = state.get("current_prices", {}) or {}
    regime = regimes.get(symbol, {}) if isinstance(regimes.get(symbol, {}), dict) else {}
    position = positions.get(symbol, {}) if isinstance(positions.get(symbol, {}), dict) else {}

    return {
        "symbol": symbol,
        "price": float(prices.get(symbol, 0.0) or 0.0),
        "regime": regime.get("regime", "unknown"),
        "confidence": float(regime.get("confidence", 0.0) or 0.0),
        "side": position.get("side"),
        "entry_price": float(position.get("entry_price", 0.0) or 0.0),
        "pnl_pct": float(position.get("pnl_pct", 0.0) or 0.0),
        "size_pct": float(position.get("size_pct", 0.0) or 0.0),
    }
