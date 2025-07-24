from arch import arch_model
import pandas as pd
import pickle

def train_garch(symbol='BTC/USDT'):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, '1d', limit=1000)
    returns = pd.Series([x[4] for x in ohlcv]).pct_change().dropna()
    
    model = arch_model(returns, vol='GARCH', p=1, q=1)
    results = model.fit(update_freq=5)
    
    with open('/app/models/garch/model.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    train_garch()
