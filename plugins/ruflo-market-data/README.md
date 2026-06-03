# ruflo-market-data Plugin

A crypto market data pipeline plugin for the ruflo framework. Provides REST/WebSocket feed ingestion, OHLCV normalization, candlestick pattern vectorization, and HNSW-indexed pattern similarity search.

## Features

- **Multi-exchange ingestion**: Binance, Coinbase, Kraken, OKX, Bybit via ccxt
- **REST & WebSocket**: Historical backfill + real-time live feed
- **OHLCV normalization**: Gap filling, outlier removal, feature engineering
- **Pattern recognition**: 12+ candlestick pattern types
- **HNSW indexing**: Sub-10ms approximate nearest-neighbor pattern search
- **Pattern matching**: Find historical analogs with forward return statistics

---

## Architecture

```
Exchange REST/WebSocket
         │
         ▼
  market-ingest (data-engineer agent)
         │
    ┌────┴────┐
    │  OHLCV  │
    │normali- │
    │zation   │
    └────┬────┘
         │
    ┌────┴──────────────────┐
    │                       │
    ▼                       ▼
 Parquet files        HNSW Pattern Index
 data/*.parquet       data/patterns/*.bin
    │                       │
    ▼                       ▼
 market-analyst        market-pattern
 (regime detection)    (similarity search)
```

---

## Agents

### data-engineer
The market data pipeline specialist.

- **Responsibilities**: OHLCV ingestion, normalization, vectorization, pattern indexing
- **Inputs**: Exchange data via REST/WebSocket
- **Outputs**: OHLCVSnapshot, PatternMatch, parquet files, HNSW indexes
- **Communication**: Sends OHLCVSnapshot to market-analyst, PatternMatch to trading-strategist

---

## Skills

| Skill | Command | Description |
|---|---|---|
| `market-ingest` | `market ingest` | OHLCV ingestion with normalization |
| `market-pattern` | `market patterns\|search` | Pattern vectorization and search |

---

## Quick Start

```bash
# Ingest BTC data
market ingest --symbol BTC/USDT --timeframe 5m --period last-30d

# Build pattern index
market patterns --symbol BTC/USDT

# Search for similar patterns
market search --symbol BTC/USDT --top-k 5 --show-outcomes

# Compare correlation between symbols
market compare --symbols BTC/USDT,ETH/USDT

# View data inventory
market history
```

---

## Data Storage

```
data/
├── BTC_USDT_5m.parquet           # Normalized OHLCV (columnar)
├── ETH_USDT_5m.parquet
├── patterns/
│   ├── BTC_USDT_patterns.bin     # HNSW index
│   ├── BTC_USDT_patterns_meta.json
│   └── ETH_USDT_patterns.bin
└── memory/
    └── trading-analysis.json
```

---

## Configuration

```yaml
data:
  symbols: ['BTC/USDT', 'ETH/USDT']
  timeframe: '5m'
  lookback: 60

exchange:
  name: 'binance'
  api_key: 'YOUR_API_KEY'
  secret_key: 'YOUR_SECRET_KEY'
  rate_limit_sleep: 0.1
```

---

## Dependencies

- `ccxt` — Exchange connectivity
- `pandas` — Data manipulation
- `pandas_ta` — Technical indicators
- `hnswlib` — HNSW nearest neighbor index
- `pyarrow` — Parquet file support
- `numpy` — Numerical operations

---

## License

MIT — see repository for full license text.
