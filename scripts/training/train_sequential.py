import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import numpy as np
import yaml
import os


def _add_indicators_sequential(df: pd.DataFrame) -> None:
    """Add technical indicators using pure pandas/numpy — no pandas-ta required.

    Appends columns in-place:
        MACD_12_26_9, RSI_14, BBL_20_2.0, BBM_20_2.0, BBU_20_2.0
    """
    close = df["close"]

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    df["MACD_12_26_9"] = macd_line

    # RSI (14)
    delta = close.diff()
    avg_gain = delta.clip(lower=0).rolling(14).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # Bollinger Bands (20, 2)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std(ddof=1)
    df["BBL_20_2.0"] = bb_mid - 2 * bb_std
    df["BBM_20_2.0"] = bb_mid
    df["BBU_20_2.0"] = bb_mid + 2 * bb_std


# 1. Lightning Data Module
class CryptoDataModule(pl.LightningDataModule):
    def __init__(self, config, symbol):
        super().__init__()
        self.config = config
        self.symbol = symbol

    def prepare_data(self):
        # In a real-world scenario, you would have separate data files for each symbol.
        # For example: f"data/{self.symbol.replace('/','_')}.csv"
        # For now, we use the single sample data file for demonstration.
        data_path = self.config['data']['sample_data_path']
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}. Please ensure you have data for training.")

        df = pd.read_csv(data_path)

        # Feature engineering — pure pandas/numpy, no pandas-ta
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        _add_indicators_sequential(df)
        df.dropna(inplace=True)

        # Create sequences
        X, y = [], []
        self.features = ['close', 'volume', 'volatility', 'MACD_12_26_9', 'RSI_14', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0']
        for i in range(self.config['data']['lookback'], len(df) - 1):
            X.append(df.iloc[i-self.config['data']['lookback']:i][self.features].values)
            y.append(df['returns'].iloc[i+1] > 0)  # Binary classification

        self.X = torch.FloatTensor(np.array(X))
        self.y = torch.FloatTensor(np.array(y))

    def train_dataloader(self):
        dataset = TensorDataset(self.X[:4000], self.y[:4000])
        return DataLoader(dataset, batch_size=self.config['training']['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        dataset = TensorDataset(self.X[4000:], self.y[4000:])
        return DataLoader(dataset, batch_size=self.config['training']['batch_size'], num_workers=4, pin_memory=True)

from scripts.models import LSTMModel, TransformerModel

# 2. Lightning Model
class LitSequential(pl.LightningModule):
    def __init__(self, config, input_size):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        model_type = config['models']['model_type']
        if model_type == 'lstm':
            self.model = LSTMModel(input_size=input_size, **config['models']['lstm'])
        elif model_type == 'transformer':
            self.model = TransformerModel(input_size=input_size, **config['models']['transformer'])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss"}}

# 3. Training Function for Multi-Pair Trading
def train(config):
    symbols = config['data']['symbols']
    model_type = config['models']['model_type']
    
    for symbol in symbols:
        print(f"--- Starting training for {symbol} ---")
        
        # Setup
        dm = CryptoDataModule(config, symbol)
        dm.prepare_data()
        model = LitSequential(config, input_size=len(dm.features))

        # Callbacks
        checkpoint_cb = ModelCheckpoint(
            dirpath=f"models/{model_type}",
            filename=f"{symbol.replace('/', '_')}_model",
            save_top_k=1,
            monitor='train_loss'
        )
        early_stop_cb = EarlyStopping(monitor='train_loss', patience=10)

        # Trainer
        trainer = pl.Trainer(
            max_epochs=config['training']['max_epochs'],
            callbacks=[checkpoint_cb, early_stop_cb],
            log_every_n_steps=5,
            enable_progress_bar=True,
            precision=32
        )

        # Train
        trainer.fit(model, dm)
        print(f"--- Finished training for {symbol} ---")

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train(config)
