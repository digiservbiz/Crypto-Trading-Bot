import torch
import joblib
from models import LSTMModel
from training.train_lstm import LitLSTM

class AIEngine:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.lstm = self._load_lstm()
        self.garch = joblib.load(f'{self.models_dir}/garch/model.joblib')
        self.anomaly = joblib.load(f'{self.models_dir}/anomaly/model.joblib')
    
    def _load_lstm(self):
        model = LitLSTM.load_from_checkpoint(f'{self.models_dir}/lstm/model.ckpt')
        model.eval()
        return model
    
    def predict(self, data_tensor, data_dict):
        lstm_out = self.lstm(data_tensor)  # Expects preprocessed tensor
        vol = self.garch.forecast(horizon=1).variance.iloc[-1]
        is_anomaly = self.anomaly.predict(data_dict['volume'])
        return {
            'direction': lstm_out > 0.5,
            'volatility': vol,
            'is_anomaly': is_anomaly == -1
        }
