"""Read-only exchange preflight for controlled validation.

The preflight checks connectivity, market metadata, and configured symbols.
It never creates or cancels orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .market_eligibility import MarketEligibility, load_market_eligibility


@dataclass(frozen=True)
class ExchangePreflight:
    exchange_name: str
    sandbox: bool
    markets_loaded: bool
    results: tuple[MarketEligibility, ...]


def run_exchange_preflight(
    exchange: Any,
    symbols: Iterable[str],
    *,
    exchange_name: str,
    sandbox: bool,
    market_mode: str = "spot",
) -> ExchangePreflight:
    results = tuple(
        load_market_eligibility(exchange, symbols, market_mode=market_mode)
    )
    return ExchangePreflight(
        exchange_name=exchange_name,
        sandbox=bool(sandbox),
        markets_loaded=True,
        results=results,
    )
