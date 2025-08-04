from sklearn.ensemble import IsolationForest
import pandas as pd
import joblib
import numpy as np

def train_anomaly(models_dir='models'):
    # Mock data - replace with real pump/dump cases
    normal = pd.DataFrame({'volume': np.random.normal(1000, 200, 1000)})
    anomalies = pd.DataFrame({'volume': np.random.normal(5000, 1000, 50)})
    X = pd.concat([normal, anomalies])
    
    model = IsolationForest(contamination=0.05)
    model.fit(X)
    
    joblib.dump(model, f'{models_dir}/anomaly/model.joblib')

if __name__ == "__main__":
    train_anomaly()
