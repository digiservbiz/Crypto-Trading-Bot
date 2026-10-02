"""Market selection helpers shared by the trading bot UI and configuration."""

from __future__ import annotations

import re
from typing import Iterable


_SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,15}/[A-Z0-9]{2,15}$")


def normalize_symbols(symbols: Iterable[str]) -> list[str]:
    """Return a validated, de-duplicated ordered market list."""
    result: list[str] = []
    seen: set[str] = set()
    for raw in symbols:
        symbol = str(raw or "").strip().upper()
        if not _SYMBOL_RE.fullmatch(symbol):
            raise ValueError(f"invalid trading symbol: {raw}")
        if symbol not in seen:
            result.append(symbol)
            seen.add(symbol)
    if not result:
        raise ValueError("at least one trading symbol is required")
    return result


def configured_symbols(config: dict) -> list[str]:
    """Read and validate the configured multi-market universe."""
    symbols = config.get("data", {}).get("symbols", ["BTC/USDT", "ETH/USDT"])
    return normalize_symbols(symbols)
