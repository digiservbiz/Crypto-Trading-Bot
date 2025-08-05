import ccxt
import yaml
from .logger import get_logger

logger = get_logger(__name__)

class Exchange:
    def __init__(self, config):
        with open('config.yaml', 'r') as f:
            keys = yaml.safe_load(f)['exchange']

        try:
            self.exchange = getattr(ccxt, keys['name'])({
                'apiKey': keys['api_key'],
                'secret': keys['secret_key'],
            })
        except Exception as e:
            logger.error(f"Error connecting to exchange: {e}")
            raise e

    def get_balance(self, currency):
        try:
            return self.exchange.fetch_balance()[currency]
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return None

    def get_price(self, symbol):
        try:
            return self.exchange.fetch_ticker(symbol)['last']
        except Exception as e:
            logger.error(f"Error fetching price: {e}")
            return None

    def create_order(self, symbol, type, side, amount, price=None):
        try:
            return self.exchange.create_order(symbol, type, side, amount, price)
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            return None

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    exchange = Exchange(config)
    print(exchange.get_balance('USDT'))
    print(exchange.get_price('BTC/USDT'))
