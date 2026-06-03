"""AI inference engine for the crypto trading bot.

Wraps LSTM/Transformer sequential models, GARCH volatility model, and
anomaly detection model. Handles graceful degradation when trained models
are not available (run training scripts first).
"""

import logging
import os
from typing import Any, Dict, Optional

import torch
import joblib
import numpy as np

logger = logging.getLogger(__name__)


class AIEngine:
    """Inference engine for neural price prediction and anomaly detection.

    Loads three model types:
    - Sequential (LSTM or Transformer): Predicts price direction
    - GARCH: Volatility regime estimation (optional)
    - Anomaly detection (IsolationForest): Flags unusual volume/price patterns

    All models are loaded lazily with graceful degradation — the engine
    works in heuristic fallback mode if trained models are not available.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the AIEngine.

        Args:
            config: Full bot configuration dictionary.
        """
        self.config = config
        self.models_dir = config["inference"]["models_dir"]
        self.sequential_models: Dict[str, Dict[str, Any]] = {}

        # Load auxiliary models with graceful degradation
        self.garch = self._try_load_joblib(
            os.path.join(self.models_dir, "garch", "model.joblib"), "GARCH"
        )
        self.anomaly = self._try_load_joblib(
            os.path.join(self.models_dir, "anomaly", "model.joblib"), "Anomaly"
        )

    def _try_load_joblib(self, path: str, label: str) -> Optional[Any]:
        """Attempt to load a joblib model, returning None on any failure."""
        if not os.path.exists(path):
            logger.warning(
                "%s model not found at %s. Run training scripts first.", label, path
            )
            return None
        try:
            model = joblib.load(path)
            logger.info("%s model loaded from %s", label, path)
            return model
        except Exception as exc:
            logger.warning("Could not load %s model from %s: %s", label, path, exc)
            return None

    def _load_sequential_model(self, symbol: str, volatility_type: str) -> Optional[Any]:
        """Load and return a PyTorch Lightning sequential model checkpoint.

        Returns None if the checkpoint is not found or fails to load.
        """
        # Use absolute imports so this works from any working directory
        from scripts.models import LSTMModel, TransformerModel  # noqa: F401 (needed for registry)
        from scripts.training.train_sequential import LitSequential, CryptoDataModule

        model_type = self.config["models"]["model_type"]
        ckpt_name = f"{symbol.replace('/', '_')}_{volatility_type}_model.ckpt"
        model_path = os.path.join(self.models_dir, model_type, ckpt_name)

        if not os.path.exists(model_path):
            logger.warning("Sequential model checkpoint not found: %s", model_path)
            return None

        try:
            dm = CryptoDataModule(self.config, symbol)
            dm.prepare_data()
            model = LitSequential.load_from_checkpoint(
                model_path,
                config=self.config,
                input_size=len(dm.features),
            )
            model.eval()
            logger.info("Sequential model loaded: %s", model_path)
            return model
        except Exception as exc:
            logger.warning("Failed to load sequential model %s: %s", model_path, exc)
            return None

    def predict(
        self,
        data_tensor: torch.Tensor,
        data_dict: Dict[str, Any],
        symbol: str,
    ) -> Dict[str, Any]:
        """Generate a prediction for the given market data.

        Args:
            data_tensor: FloatTensor of shape (1, seq_len, n_features).
            data_dict: Dict with 'volume' (array) and 'volatility' (array).
            symbol: Trading pair symbol.

        Returns:
            Dict with keys:
                direction (bool): True = up, False = down
                volatility (float): Current volatility
                is_anomaly (array or bool): True if volume anomaly detected
        """
        volatility = float(data_dict["volatility"][-1]) if len(data_dict.get("volatility", [])) > 0 else 0.0

        # Determine volatility type for model selection
        if self.config["models"]["model_selection"]["enabled"]:
            threshold = self.config["models"]["model_selection"]["volatility_threshold"]
            volatility_type = "high_volatility" if volatility > threshold else "low_volatility"
        else:
            volatility_type = "model"

        # Load model lazily if needed
        if symbol not in self.sequential_models:
            self.sequential_models[symbol] = {}
        if volatility_type not in self.sequential_models[symbol]:
            self.sequential_models[symbol][volatility_type] = self._load_sequential_model(
                symbol, volatility_type
            )

        sequential_model = self.sequential_models[symbol].get(volatility_type)

        # Direction prediction
        if sequential_model is not None:
            try:
                with torch.no_grad():
                    output = sequential_model(data_tensor)
                direction = bool(output.squeeze() > 0.5)
            except Exception as exc:
                logger.debug("Sequential model inference failed: %s — using fallback", exc)
                direction = self._heuristic_direction(data_tensor)
        else:
            direction = self._heuristic_direction(data_tensor)

        # Anomaly detection
        if self.anomaly is not None:
            try:
                volume_data = np.array(data_dict["volume"]).reshape(-1, 1)
                is_anomaly = self.anomaly.predict(volume_data)
            except Exception as exc:
                logger.debug("Anomaly model inference failed: %s", exc)
                is_anomaly = np.array([1])  # Conservative: assume normal
        else:
            is_anomaly = np.array([1])  # 1 = normal (IsolationForest convention)

        return {
            "direction": direction,
            "volatility": volatility,
            "is_anomaly": is_anomaly == -1,  # -1 = anomaly in IsolationForest
        }

    def _heuristic_direction(self, data_tensor: torch.Tensor) -> bool:
        """Fallback direction: last bar close > previous bar close."""
        try:
            # data_tensor shape: (1, seq_len, n_features) — feature 0 is close
            if data_tensor.shape[1] >= 2:
                last_close = float(data_tensor[0, -1, 0])
                prev_close = float(data_tensor[0, -2, 0])
                return last_close > prev_close
        except Exception:
            pass
        return False
