import ccxt
import yaml

class Exchange:
    def __init__(self, config):
        with open('config.yaml', 'r') as f:
            keys = yaml.safe_load(f)['exchange']

        self.exchange = getattr(ccxt, keys['name'])({
            'apiKey': keys['api_key'],
            'secret': keys['secret_key'],
        })

    def get_balance(self, currency):
        return self.exchange.fetch_balance()[currency]

    def get_price(self, symbol):
        return self.exchange.fetch_ticker(symbol)['last']

    def create_order(self, symbol, type, side, amount, price=None):
        return self.exchange.create_order(symbol, type, side, amount, price)

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    exchange = Exchange(config)
    print(exchange.get_balance('USDT'))
    print(exchange.get_price('BTC/USDT'))
