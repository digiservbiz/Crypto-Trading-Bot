# Skill: market-ingest

## Overview
REST and WebSocket OHLCV ingestion from ccxt-compatible exchanges. Provides normalized, analysis-ready market data to the trading pipeline.

## Trigger
```
market ingest [options]
```

## Description
The `market-ingest` skill handles all market data acquisition and preprocessing. It supports both historical (REST) and real-time (WebSocket) data modes, with automatic normalization, gap detection, and feature engineering.

## Ingestion Modes

### REST (Historical)
- Fetch batches of OHLCV data from exchange REST API
- Handles pagination for large date ranges
- Rate limit compliance (configurable sleep between requests)
- Supports all ccxt-compatible exchanges

```python
# Example: fetch 1000 candles of BTC/USDT 5m
exchange = ccxt.binance()
ohlcv = exchange.fetch_ohlcv('BTC/USDT', '5m', limit=1000)
```

### WebSocket (Real-time)
- Subscribe to live OHLCV stream via ccxt async
- Writes completed candles on close event
- Reconnection with exponential backoff (1s, 2s, 4s, 8s, 16s, then fail)
- Gap detection: flag if gap > 2× expected interval

## Normalization Pipeline

Applied to all ingested data:

1. **Gap filling**: forward-fill up to 4-hour gaps
2. **Outlier removal**: remove prices > 5× rolling std
3. **Feature engineering**: compute all indicators
4. **Scaling**: min-max normalize feature columns (optional)

### Computed Features

| Feature | Source | Formula |
|---|---|---|
| `returns` | close | `pct_change()` |
| `log_returns` | close | `log(close / close.shift(1))` |
| `volatility` | returns | `rolling(20).std()` |
| `MACD_12_26_9` | close | pandas_ta |
| `RSI_14` | close | pandas_ta |
| `BBL/BBM/BBU_5_2.0` | close | pandas_ta |
| `ATRr_14` | H, L, C | pandas_ta |
| `OBV` | close, volume | pandas_ta |
| `volume_ma_20` | volume | `rolling(20).mean()` |
| `volume_ratio` | volume | `volume / volume_ma_20` |

## Output

Normalized OHLCV DataFrame saved to:
- `data/{symbol}_{timeframe}.parquet` (primary)
- `data/{symbol}_{timeframe}.csv` (if `--format csv`)

Data quality report:

```json
{
  "symbol": "BTC/USDT",
  "timeframe": "5m",
  "rows_fetched": 8640,
  "rows_after_normalization": 8617,
  "gaps_detected": 2,
  "gap_timestamps": [1714176000, 1714435200],
  "date_range": "2024-04-21/2024-05-21",
  "normalization_applied": true
}
```

## Usage Examples

```bash
# Ingest last 30 days
market ingest --symbol BTC/USDT --timeframe 5m --period last-30d

# Specific date range
market ingest --symbol ETH/USDT --from 2023-01-01 --to 2024-01-01

# All configured symbols
market ingest --all-symbols --timeframe 1h

# Real-time WebSocket mode
market ingest --symbol BTC/USDT --mode websocket

# CSV output
market ingest --symbol BTC/USDT --format csv
```

## Exchange Configuration

```yaml
exchange:
  name: 'binance'
  api_key: 'YOUR_API_KEY'
  secret_key: 'YOUR_SECRET_KEY'
  rate_limit_sleep: 0.1   # seconds between requests
  failover_exchange: 'kraken'
```

## Error Codes

| Code | Description |
|---|---|
| `EXCHANGE_TIMEOUT` | Request timed out — retry initiated |
| `RATE_LIMIT` | Hit exchange rate limit — backing off |
| `GAP_DETECTED` | Missing candles in data |
| `SYMBOL_NOT_FOUND` | Exchange does not support symbol |
| `AUTH_ERROR` | Invalid API keys |
