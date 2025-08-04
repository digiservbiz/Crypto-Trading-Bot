from sklearn.ensemble import IsolationForest
import pandas as pd
import joblib
import numpy as np
import yaml

def train_anomaly(config):
    # Mock data - replace with real pump/dump cases
    normal = pd.DataFrame({'volume': np.random.normal(1000, 200, 1000)})
    anomalies = pd.DataFrame({'volume': np.random.normal(5000, 1000, 50)})
    X = pd.concat([normal, anomalies])
    
    model = IsolationForest(contamination=config['models']['anomaly']['contamination'])
    model.fit(X)
    
    joblib.dump(model, f"{config['inference']['models_dir']}/anomaly/model.joblib")

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_anomaly(config)
