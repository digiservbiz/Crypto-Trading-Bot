import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import ccxt
import pandas as pd
import numpy as np

# 1. Lightning Data Module
class CryptoDataModule(pl.LightningDataModule):
    def __init__(self, symbol='BTC/USDT', timeframe='5m', lookback=60, batch_size=1024):
        super().__init__()
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback = lookback
        self.batch_size = batch_size

    def prepare_data(self):
        # Fetch data from exchange
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=5000)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
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

# 2. Lightning Model
class LitLSTM(pl.LightningModule):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1])
    
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
                "monitor": "val_loss",
                "interval": "epoch"
            }
        }

# 3. Training Function with Multi-GPU Support
def train():
    # Config
    config = {
        "symbol": "BTC/USDT",
        "timeframe": "5m",
        "lookback": 60,
        "batch_size": 2048,  # Larger batches for multi-GPU
        "max_epochs": 100,
        "precision": "16-mixed"  # Mixed precision training
    }
    
    # Setup
    dm = CryptoDataModule(
        symbol=config['symbol'],
        timeframe=config['timeframe'],
        lookback=config['lookback'],
        batch_size=config['batch_size']
    )
    model = LitLSTM()
    
    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath='models/',
        filename='lstm-{epoch}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss'
    )
    early_stop_cb = EarlyStopping(monitor='val_loss', patience=10)
    
    # Trainer
    trainer = pl.Trainer(
        accelerator="auto",  # Automatically selects GPU/CPU
        devices="auto",     # Uses all available GPUs
        strategy="ddp_find_unused_parameters_true",  # For multi-GPU
        max_epochs=config['max_epochs'],
        precision=config['precision'],
        callbacks=[checkpoint_cb, early_stop_cb],
        log_every_n_steps=5,
        enable_progress_bar=True
    )
    
    # Train
    trainer.fit(model, dm)

if __name__ == "__main__":
    train()