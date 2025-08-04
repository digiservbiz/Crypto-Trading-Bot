from arch import arch_model
import pandas as pd
import joblib
import yaml

def train_garch(config):
    df = pd.read_csv(config['data']['sample_data_path'])
    returns = df['close'].pct_change().dropna()
    
    model = arch_model(returns, vol='GARCH', p=config['models']['garch']['p'], q=config['models']['garch']['q'])
    model_fitted = model.fit(update_freq=5)
    
    joblib.dump(model_fitted, f"{config['inference']['models_dir']}/garch/model.joblib")

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_garch(config)
