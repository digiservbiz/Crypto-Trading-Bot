from arch import arch_model
import pandas as pd
import joblib
import ccxt

def train_garch(data_path='data/sample_ohlcv.csv', models_dir='models'):
    df = pd.read_csv(data_path)
    returns = df['close'].pct_change().dropna()
    
    model = arch_model(returns, vol='GARCH', p=1, q=1)
    results = model.fit(update_freq=5)
    
    joblib.dump(results, f'{models_dir}/garch/model.joblib')

if __name__ == "__main__":
    train_garch()
