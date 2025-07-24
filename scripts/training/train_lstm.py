import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

class LSTMModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

def prepare_data(symbol='BTC/USDT', lookback=60):
    # Fetch data from Binance
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, '5m', limit=5000)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Feature engineering
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df.dropna(inplace=True)
    
    # Normalize
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['close', 'volume', 'volatility']])
    
    # Create sequences
    X, y = [], []
    for i in range(lookback, len(df)):
        X.append(scaled[i-lookback:i])
        y.append(df['returns'].iloc[i] > 0)  # Binary classification
    return torch.FloatTensor(X), torch.FloatTensor(y)

def train():
    X, y = prepare_data()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = LSTMModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(100):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
    torch.save(model.state_dict(), '/app/models/lstm/model.pt')

if __name__ == "__main__":
    train()
