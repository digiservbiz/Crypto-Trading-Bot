"""Exchange interface wrapping ccxt.

API keys are resolved in this priority order:
1. Environment variables: EXCHANGE_API_KEY, EXCHANGE_SECRET_KEY, EXCHANGE_NAME
2. Passed config dict (config['exchange'])

This ensures secrets are never hard-coded in config files for live deployments.
"""

import os
import ccxt
from .logger import get_logger

logger = get_logger(__name__)


class Exchange:
    """ccxt exchange wrapper with environment-variable key injection."""

    def __init__(self, config: dict) -> None:
        """Initialize the exchange connection.

        API key resolution order:
        1. EXCHANGE_API_KEY / EXCHANGE_SECRET_KEY env vars
        2. config['exchange']['api_key'] / 'secret_key'

        Args:
            config: Full bot configuration dictionary.
        """
        exchange_config = config.get("exchange", {})

        # Prefer env vars for live deployments (never commit secrets to config)
        api_key = (
            os.environ.get("EXCHANGE_API_KEY")
            or exchange_config.get("api_key", "")
        )
        secret_key = (
            os.environ.get("EXCHANGE_SECRET_KEY")
            or exchange_config.get("secret_key", "")
        )
        exchange_name = (
            os.environ.get("EXCHANGE_NAME")
            or exchange_config.get("name", "binance")
        )

        if not api_key or api_key == "YOUR_API_KEY":
            logger.warning(
                "Exchange API key not set. Set EXCHANGE_API_KEY env var or "
                "config['exchange']['api_key'] for live trading. "
                "Dry-run and market data will still work on public endpoints."
            )

        try:
            exchange_cls = getattr(ccxt, exchange_name, None)
            if exchange_cls is None:
                raise ValueError(f"Unknown exchange: {exchange_name}")
            self.exchange = exchange_cls({
                "apiKey": api_key,
                "secret": secret_key,
            })
            logger.info("Exchange initialized: %s", exchange_name)
        except Exception as exc:
            logger.error("Error connecting to exchange %s: %s", exchange_name, exc)
            raise

    def get_balance(self, currency: str) -> dict:
        """Fetch current balance for a currency.

        Returns:
            Dict with 'free', 'used', 'total' keys, or empty dict on error.
        """
        try:
            return self.exchange.fetch_balance().get(currency, {"free": 0.0, "used": 0.0, "total": 0.0})
        except Exception as exc:
            logger.error("Error fetching balance for %s: %s", currency, exc)
            return {"free": 0.0, "used": 0.0, "total": 0.0}

    def get_price(self, symbol: str) -> float:
        """Fetch last traded price for a symbol.

        Returns:
            Last price as float, or 0.0 on error.
        """
        try:
            return float(self.exchange.fetch_ticker(symbol)["last"])
        except Exception as exc:
            logger.error("Error fetching price for %s: %s", symbol, exc)
            return 0.0

    def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: float = None,
    ) -> dict:
        """Place an order on the exchange.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT").
            order_type: "market" or "limit".
            side: "buy" or "sell".
            amount: Order size in base currency.
            price: Limit price (required for limit orders, ignored for market).

        Returns:
            Order dict from ccxt, or None on error.
        """
        try:
            order = self.exchange.create_order(symbol, order_type, side, amount, price)
            logger.info(
                "Order created: %s %s %s %.6f @ %s | id=%s status=%s",
                symbol, side, order_type, amount,
                price or "market", order.get("id"), order.get("status")
            )
            return order
        except Exception as exc:
            logger.error(
                "Error creating %s %s order for %s (amount=%.6f): %s",
                order_type, side, symbol, amount, exc
            )
            return None


if __name__ == "__main__":
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    exchange = Exchange(config)
    print(exchange.get_balance("USDT"))
    print(exchange.get_price("BTC/USDT"))
