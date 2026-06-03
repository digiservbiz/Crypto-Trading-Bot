# Skill: trader-train

## Overview
LSTM and Transformer neural model training using PyTorch Lightning. Trains direction prediction models for each symbol and volatility regime.

## Trigger
```
trader train lstm|transformer [options]
```

## Description
The `trader-train` skill runs the PyTorch Lightning training pipeline to produce neural models for price direction prediction. Models are saved to the `models/` directory and loaded by the `AIEngine` during inference.

## Supported Model Types

### LSTM (Long Short-Term Memory)
- **Architecture**: Stacked LSTM layers + linear output head
- **Input**: Sequence of OHLCV + indicator features (60 timesteps)
- **Output**: Probability of upward price movement
- **Config**: `hidden_size: 64`, `num_layers: 2`
- **Best for**: Trending regimes with persistent patterns

### Transformer
- **Architecture**: Multi-head self-attention encoder + linear output
- **Input**: Same OHLCV + indicator features
- **Output**: Probability of upward price movement
- **Config**: `d_model: 64`, `nhead: 4`, `num_encoder_layers: 2`
- **Best for**: Volatile regimes with complex non-linear patterns

### N-BEATS (planned)
- Neural basis expansion analysis for interpretable time series
- Future implementation

## Feature Set

Default features used for training:

```python
features = [
    'close', 'volume', 'volatility',
    'MACD_12_26_9', 'RSI_14',
    'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0',
    'ATRr_14', 'OBV'
]
```

## Volatility-Based Model Selection

Two model variants are trained per symbol:
- `{symbol}_high_volatility_model.ckpt` — for ATR ratio > threshold
- `{symbol}_low_volatility_model.ckpt` — for ATR ratio ≤ threshold

The `AIEngine` automatically selects the appropriate model at inference time.

## Training Configuration

```yaml
training:
  batch_size: 2048
  max_epochs: 100

models:
  lstm:
    hidden_size: 64
    num_layers: 2
  transformer:
    d_model: 64
    nhead: 4
    num_encoder_layers: 2
    dim_feedforward: 256
    dropout: 0.1
```

## Output

Model checkpoint: `models/{model_type}/{symbol}_{volatility_type}_model.ckpt`

Training metrics logged to `lightning_logs/`:
- `train_loss`, `val_loss` per epoch
- Final validation accuracy

## Usage Examples

```bash
# Train LSTM for BTC
trader train lstm --symbol BTC/USDT

# Train Transformer for ETH
trader train transformer --symbol ETH/USDT

# High volatility variant only
trader train lstm --symbol BTC/USDT --volatility-type high_volatility

# Custom epochs
trader train transformer --symbol BTC/USDT --epochs 200

# Via Python directly
python scripts/training/train_sequential.py --config config.yaml --symbol BTC/USDT
```

## Tier Classification
All training operations are **Tier 3** (heavy, async). Training is NEVER run in the live trading loop. Schedule training as an offline job.

## Memory Namespace
None — model checkpoints are stored directly in `models/`.
