"""Exchange market eligibility checks for configured trading pairs.

This module performs read-only market discovery. It never submits orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class MarketEligibility:
    symbol: str
    eligible: bool
    reason: str


def _market_for(markets: dict[str, Any], symbol: str) -> dict[str, Any] | None:
    raw = markets.get(symbol)
    return raw if isinstance(raw, dict) else None


def check_market_eligibility(
    markets: dict[str, Any],
    symbols: Iterable[str],
    *,
    market_mode: str = "spot",
) -> list[MarketEligibility]:
    """Validate configured symbols against an authoritative load_markets snapshot."""
    if market_mode not in {"spot", "futures"}:
        raise ValueError("market_mode must be spot or futures")

    results: list[MarketEligibility] = []
    for symbol in symbols:
        market = _market_for(markets, symbol)
        if market is None:
            results.append(MarketEligibility(symbol, False, "market not found"))
            continue
        if market.get("active") is False:
            results.append(MarketEligibility(symbol, False, "market inactive"))
            continue

        if market_mode == "spot" and market.get("spot") is False:
            results.append(MarketEligibility(symbol, False, "not a spot market"))
            continue
        if market_mode == "futures" and market.get("future") is not True and market.get("swap") is not True:
            results.append(MarketEligibility(symbol, False, "not a futures/swap market"))
            continue

        results.append(MarketEligibility(symbol, True, "eligible"))
    return results


def load_market_eligibility(exchange: Any, symbols: Iterable[str], *, market_mode: str = "spot") -> list[MarketEligibility]:
    """Load authoritative markets from a CCXT-style exchange without trading."""
    load_markets = getattr(exchange, "load_markets", None)
    if not callable(load_markets):
        raise RuntimeError("exchange market discovery is unsupported")
    markets = load_markets()
    if not isinstance(markets, dict):
        raise RuntimeError("exchange returned invalid market metadata")
    return check_market_eligibility(markets, symbols, market_mode=market_mode)
