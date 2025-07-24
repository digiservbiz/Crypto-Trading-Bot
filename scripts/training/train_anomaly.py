from sklearn.ensemble import IsolationForest
import pandas as pd
import joblib

def train_anomaly():
    # Mock data - replace with real pump/dump cases
    normal = pd.DataFrame({'volume': np.random.normal(1000, 200, 1000)})
    anomalies = pd.DataFrame({'volume': np.random.normal(5000, 1000, 50)})
    X = pd.concat([normal, anomalies])
    
    model = IsolationForest(contamination=0.05)
    model.fit(X)
    
    joblib.dump(model, '/app/models/anomaly/model.joblib')

if __name__ == "__main__":
    train_anomaly()
