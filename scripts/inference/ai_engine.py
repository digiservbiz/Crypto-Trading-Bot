import torch
import joblib
from models import LSTMModel, TransformerModel
from training.train_sequential import LitSequential

class AIEngine:
    def __init__(self, config):
        self.config = config
        self.models_dir = config['inference']['models_dir']
        self.sequential_model = self._load_sequential_model()
        self.garch = joblib.load(f'{self.models_dir}/garch/model.joblib')
        self.anomaly = joblib.load(f'{self.models_dir}/anomaly/model.joblib')
    
    def _load_sequential_model(self):
        model = LitSequential.load_from_checkpoint(f"{self.models_dir}/{self.config['models']['model_type']}/model.ckpt")
        model.eval()
        return model
    
    def predict(self, data_tensor, data_dict):
        sequential_out = self.sequential_model(data_tensor)  # Expects preprocessed tensor
        vol = self.garch.forecast(horizon=1).variance.iloc[-1]
        is_anomaly = self.anomaly.predict(data_dict['volume'])
        return {
            'direction': sequential_out > 0.5,
            'volatility': vol,
            'is_anomaly': is_anomaly == -1
        }
