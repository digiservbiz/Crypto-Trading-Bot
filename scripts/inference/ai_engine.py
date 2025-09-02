import torch
import joblib
from models import LSTMModel, TransformerModel
from training.train_sequential import LitSequential, CryptoDataModule

class AIEngine:
    def __init__(self, config):
        self.config = config
        self.models_dir = config['inference']['models_dir']
        self.sequential_models = {}
        self.garch = joblib.load(f'{self.models_dir}/garch/model.joblib')
        self.anomaly = joblib.load(f'{self.models_dir}/anomaly/model.joblib')
    
    def _load_sequential_model(self, symbol, volatility_type):
        model_path = f"{self.models_dir}/{self.config['models']['model_type']}/{symbol.replace('/', '_')}_{volatility_type}_model.ckpt"
        dm = CryptoDataModule(self.config)
        dm.prepare_data()

        model = LitSequential.load_from_checkpoint(
            model_path,
            config=self.config,
            input_size=len(dm.features)
        )
        model.eval()
        return model
    
    def predict(self, data_tensor, data_dict, symbol):
        volatility = data_dict['volatility'][-1]
        
        if self.config['models']['model_selection']['enabled']:
            volatility_type = 'high_volatility' if volatility > self.config['models']['model_selection']['volatility_threshold'] else 'low_volatility'
        else:
            volatility_type = 'model'

        if symbol not in self.sequential_models or volatility_type not in self.sequential_models[symbol]:
            if symbol not in self.sequential_models:
                self.sequential_models[symbol] = {}
            self.sequential_models[symbol][volatility_type] = self._load_sequential_model(symbol, volatility_type)
        
        sequential_model = self.sequential_models[symbol][volatility_type]
        sequential_out = sequential_model(data_tensor)
        
        is_anomaly = self.anomaly.predict(data_dict['volume'])
        return {
            'direction': sequential_out > 0.5,
            'volatility': volatility,
            'is_anomaly': is_anomaly == -1
        }
