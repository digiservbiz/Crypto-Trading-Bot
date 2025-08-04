import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import numpy as np

# 1. Lightning Data Module
class CryptoDataModule(pl.LightningDataModule):
    def __init__(self, data_path='data/sample_ohlcv.csv', lookback=60, batch_size=1024):
        super().__init__()
        self.data_path = data_path
        self.lookback = lookback
        self.batch_size = batch_size

    def prepare_data(self):
        # Load data from csv
        df = pd.read_csv(self.data_path)
        
        # Feature engineering
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df.dropna(inplace=True)
        
        # Create sequences
        X, y = [], []
        for i in range(self.lookback, len(df)):
            X.append(df.iloc[i-self.lookback:i][['close', 'volume', 'volatility']].values)
            y.append(df['returns'].iloc[i] > 0)  # Binary classification
            
        self.X = torch.FloatTensor(np.array(X))
        self.y = torch.FloatTensor(np.array(y))
        
    def train_dataloader(self):
        dataset = TensorDataset(self.X[:4000], self.y[:4000])  # First 4000 for training
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    def val_dataloader(self):
        dataset = TensorDataset(self.X[4000:], self.y[4000:])  # Last 1000 for validation
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True)

from ..models import LSTMModel

# 2. Lightning Model
class LitLSTM(pl.LightningModule):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        super().__init__()
        self.save_hyperparameters()
        self.model = LSTMModel(input_size, hidden_size, num_layers)
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
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "epoch"
            }
        }

# 3. Training Function with Multi-GPU Support
def train():
    # Config
    config = {
        "data_path": "data/sample_ohlcv.csv",
        "lookback": 60,
        "batch_size": 2048,  # Larger batches for multi-GPU
        "max_epochs": 100,
        "precision": "16-mixed"  # Mixed precision training
    }
    
    # Setup
    dm = CryptoDataModule(
        data_path=config['data_path'],
        lookback=config['lookback'],
        batch_size=config['batch_size']
    )
    model = LitLSTM()
    
    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath='models/',
        filename='lstm-{epoch}-{train_loss:.2f}',
        save_top_k=3,
        monitor='train_loss'
    )
    early_stop_cb = EarlyStopping(monitor='train_loss', patience=10)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        callbacks=[checkpoint_cb, early_stop_cb],
        log_every_n_steps=5,
        enable_progress_bar=True,
        precision=32
    )
    
    # Train
    trainer.fit(model, dm)

if __name__ == "__main__":
    train()