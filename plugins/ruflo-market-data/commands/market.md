# market — CLI Command Reference

The `market` command provides tools for data ingestion, pattern analysis, similarity search, and market data management.

---

## Commands

### `market ingest`

Ingest OHLCV data from an exchange into local storage.

```bash
market ingest --symbol BTC/USDT --timeframe 5m --period last-30d
market ingest --symbol ETH/USDT --timeframe 1h --from 2023-01-01 --to 2024-01-01
market ingest --all-symbols --timeframe 5m
```

**Options:**
| Flag | Description | Default |
|---|---|---|
| `--symbol` | Trading pair | From config |
| `--symbols` | Comma-separated list | From config |
| `--all-symbols` | Ingest all configured symbols | false |
| `--timeframe` | OHLCV timeframe | `5m` |
| `--period` | Shorthand period (`last-7d`, `last-30d`, `last-90d`) | `last-30d` |
| `--from` | Start date (YYYY-MM-DD) | None |
| `--to` | End date (YYYY-MM-DD) | Now |
| `--exchange` | Exchange override | From config |
| `--format` | Storage format: `csv\|parquet` | `parquet` |
| `--normalize` | Apply normalization pipeline | `true` |
| `--verbose` | Show progress | `false` |

**Output:**
```
Ingesting BTC/USDT 5m from 2024-04-21 to 2024-05-21...
  Fetched: 8640 candles
  Gaps detected: 0
  Saved to: data/BTC_USDT_5m.parquet
  Normalization: DONE
```

---

### `market patterns`

Extract and index candlestick patterns from stored OHLCV data.

```bash
market patterns --symbol BTC/USDT
market patterns --symbol ETH/USDT --rebuild-index
market patterns --all-symbols --min-significance 0.7
```

**Options:**
| Flag | Description | Default |
|---|---|---|
| `--symbol` | Symbol to analyze | From config |
| `--all-symbols` | Analyze all symbols | false |
| `--rebuild-index` | Force HNSW index rebuild | false |
| `--min-significance` | Minimum pattern significance | `0.5` |
| `--window-size` | Candlestick window (candles) | `5` |
| `--output` | Save patterns to JSON | None |

**Detected pattern types:**
- `doji`, `hammer`, `hanging-man`, `engulfing-bull`, `engulfing-bear`
- `morning-star`, `evening-star`, `three-white-soldiers`, `three-black-crows`
- `spinning-top`, `marubozu-bull`, `marubozu-bear`

**Output:**
```
Pattern Analysis — BTC/USDT
  Total candles analyzed:  8,640
  Patterns detected:       1,247
  Index size:              1,247 vectors (15-dim)
  Top patterns:
    engulfing-bull:    342 (27.4%)
    doji:              289 (23.2%)
    hammer:            198 (15.9%)
  Saved to: data/patterns/BTC_USDT_patterns.bin
```

---

### `market search`

Search for historical patterns similar to the current market state.

```bash
market search --symbol BTC/USDT
market search --symbol BTC/USDT --top-k 10 --min-similarity 0.85
market search --symbol ETH/USDT --show-outcomes
```

**Options:**
| Flag | Description | Default |
|---|---|---|
| `--symbol` | Symbol to analyze | `BTC/USDT` |
| `--top-k` | Number of similar patterns to return | `5` |
| `--min-similarity` | Minimum cosine similarity | `0.80` |
| `--show-outcomes` | Show forward returns for matched patterns | `true` |
| `--window` | Candle window for current state | `5` |
| `--output-json` | Save matches to JSON | None |

**Output:**
```
Pattern Search — BTC/USDT (current state)
Current pattern: engulfing-bull (confidence: 0.84)

Similar historical patterns:
  1. [2024-02-14 09:00] similarity=0.94 → forward_return=+3.2% (5 candles)
  2. [2023-11-02 14:00] similarity=0.91 → forward_return=+1.8% (5 candles)
  3. [2023-08-17 06:00] similarity=0.88 → forward_return=-0.4% (5 candles)
  4. [2024-01-09 21:00] similarity=0.85 → forward_return=+2.1% (5 candles)
  5. [2023-05-29 03:00] similarity=0.83 → forward_return=+5.7% (5 candles)

Aggregate: +2.48% mean forward return (4/5 positive)
```

---

### `market history`

Display data availability and storage statistics.

```bash
market history
market history --symbol BTC/USDT
market history --verbose
```

**Options:**
| Flag | Description | Default |
|---|---|---|
| `--symbol` | Show history for specific symbol | All |
| `--verbose` | Show detailed stats | false |
| `--format` | Output format: `table\|json` | `table` |

**Output:**
```
Market Data Inventory:
  BTC/USDT  5m  rows=8,640   from=2024-04-21  to=2024-05-21  gaps=0
  ETH/USDT  5m  rows=8,640   from=2024-04-21  to=2024-05-21  gaps=2
  BTC/USDT  1h  rows=720     from=2024-04-21  to=2024-05-21  gaps=0
```

---

### `market compare`

Compare pattern similarity and return statistics across two symbols or time periods.

```bash
market compare --symbols BTC/USDT,ETH/USDT
market compare --symbol BTC/USDT --periods 2023/2024
market compare --symbols BTC/USDT,ETH/USDT --metric correlation
```

**Options:**
| Flag | Description | Default |
|---|---|---|
| `--symbols` | Two symbols to compare | Required |
| `--symbol` | Single symbol for period comparison | None |
| `--periods` | Two periods to compare (year/year) | None |
| `--metric` | `correlation\|pattern-overlap\|return-dist` | `correlation` |
| `--lookback-days` | Rolling correlation window | `30` |
| `--output` | Save to JSON | None |

**Output:**
```
Cross-Symbol Comparison: BTC/USDT vs ETH/USDT
  Pearson correlation (30d):    0.847  [WARNING: approaching 0.85 limit]
  Return distribution overlap:  73.2%
  Shared pattern types:         8/12
  Divergence events (last 30d): 4
```

---

## Global Options

| Flag | Description |
|---|---|
| `--config <path>` | Config file (default: `config.yaml`) |
| `--data-dir <path>` | Data directory (default: `data/`) |
| `--verbose`, `-v` | Verbose output |
| `--json` | JSON output format |

---

## Storage Formats

Data is stored in `data/` directory:

```
data/
├── {symbol}_{timeframe}.csv        # CSV format (legacy)
├── {symbol}_{timeframe}.parquet    # Parquet format (recommended)
├── patterns/
│   └── {symbol}_patterns.bin       # HNSW index binary
└── memory/
    └── trading-analysis.json       # Agent memory namespace
```
