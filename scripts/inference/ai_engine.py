import torch
import joblib
from models import LSTMModel, TransformerModel
from training.train_sequential import LitSequential, CryptoDataModule

class AIEngine:
    def __init__(self, config):
        self.config = config
        self.models_dir = config['inference']['models_dir']
        self.sequential_models = {}
        # GARCH and Anomaly models are assumed to be more general or need a different multi-symbol strategy
        self.garch = joblib.load(f'{self.models_dir}/garch/model.joblib')
        self.anomaly = joblib.load(f'{self.models_dir}/anomaly/model.joblib')
    
    def _load_sequential_model(self, symbol):
        # Construct a path for the symbol-specific model
        model_path = f"{self.models_dir}/{self.config['models']['model_type']}/{symbol.replace('/', '_')}_model.ckpt"
        
        # This part assumes that you have a way to get the features for a symbol,
        # potentially by running a part of the CryptoDataModule logic for that symbol.
        # For now, we'll keep it simple, but this might need adjustment.
        dm = CryptoDataModule(self.config) # This may need to be adapted for multiple symbols
        dm.prepare_data() # This loads data for all symbols, which is inefficient but works for now

        model = LitSequential.load_from_checkpoint(
            model_path,
            config=self.config,
            input_size=len(dm.features)
        )
        model.eval()
        return model
    
    def predict(self, data_tensor, data_dict, symbol):
        # Load model on demand
        if symbol not in self.sequential_models:
            self.sequential_models[symbol] = self._load_sequential_model(symbol)
        
        sequential_model = self.sequential_models[symbol]
        sequential_out = sequential_model(data_tensor)
        
        vol = self.garch.forecast(horizon=1).variance.iloc[-1]
        is_anomaly = self.anomaly.predict(data_dict['volume'])
        return {
            'direction': sequential_out > 0.5,
            'volatility': vol,
            'is_anomaly': is_anomaly == -1
        }
