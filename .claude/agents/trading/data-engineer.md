# Agent: data-engineer

## Role
The data-engineer is the **market data pipeline specialist** within the ruflo-market-data plugin. It handles all aspects of raw market data acquisition, normalization, vectorization, and pattern indexing. It provides clean, analysis-ready OHLCV data to all other pipeline agents.

## Plugin
`ruflo-market-data`

---

## Responsibilities

### 1. OHLCV Data Ingestion

**REST Ingestion (historical):**
- Fetch historical OHLCV from ccxt-compatible exchanges
- Configurable timeframes: 1m, 3m, 5m, 15m, 30m, 1h, 4h, 1d
- Pagination handling for large date ranges
- Rate limit compliance per exchange requirements
- Supports: Binance, Coinbase, Kraken, OKX, Bybit

**WebSocket Ingestion (real-time):**
- Subscribe to live OHLCV streams for configured symbols
- Buffer management: maintain rolling window of configurable depth
- Automatic reconnection with exponential backoff (max 5 retries)
- Gap detection: identify missing candles > 2x expected interval
- Write completed candles to local store on candle close

**Data Sources Priority:**
1. Primary exchange (configured in `config.yaml`)
2. Secondary exchange (failover)
3. Aggregated (VWAP across multiple exchanges)

### 2. OHLCV Normalization

Per-symbol normalization pipeline:

```
Raw OHLCV
    │
    ▼
Forward-fill gaps (max 4-hour gaps only)
    │
    ▼
Remove outliers (price > 5x rolling std)
    │
    ▼
Percent-change normalization (optional)
    │
    ▼
Min-max scaling per rolling window
    │
    ▼
Feature engineering (indicators)
    │
    ▼
Clean OHLCV DataFrame
```

**Output schema:**
```
timestamp, open, high, low, close, volume,
returns, log_returns, volatility,
MACD_12_26_9, MACD_signal, RSI_14,
BBL_5_2.0, BBM_5_2.0, BBU_5_2.0, ATRr_14, OBV,
volume_ma_20, volume_ratio
```

### 3. Candlestick Pattern Vectorization

Extract and vectorize candlestick patterns from OHLCV sequences for HNSW-indexed similarity search:

**Pattern types recognized:**
- Doji (neutral)
- Hammer / Hanging Man (reversal)
- Engulfing (bullish/bearish)
- Morning Star / Evening Star (3-candle reversal)
- Three White Soldiers / Three Black Crows (continuation)
- Spinning Top (uncertainty)
- Marubozu (strong momentum)

**Vectorization method:**
Each 5-candle window is vectorized as a 15-dimensional vector:
- [open_pct, high_pct, low_pct, close_pct, volume_pct] × 3 candles relative metrics
- Normalized to unit sphere for cosine similarity

### 4. HNSW Pattern Indexing

Maintain an HNSW (Hierarchical Navigable Small World) index for fast approximate nearest-neighbor pattern matching:

- Library: `hnswlib`
- Distance metric: cosine
- Ef construction: 200
- M (connections per element): 16
- Index persisted to `data/patterns/{symbol}_patterns.bin`

**Query:** Given current 5-candle window vector, find top-K similar historical patterns and their subsequent price movements.

---

## Output Messages

### `OHLCVSnapshot`
```python
@dataclass
class OHLCVSnapshot:
    symbol: str
    timeframe: str
    df: pd.DataFrame          # Normalized OHLCV with indicators
    last_candle_time: float
    num_rows: int
    has_gaps: bool
    snapshot_id: str
```

### `PatternMatch`
```python
@dataclass
class PatternMatch:
    symbol: str
    pattern_type: str
    similarity_score: float   # Cosine similarity 0-1
    historical_forward_return: float  # What happened next (historically)
    confidence: float
    query_timestamp: float
    match_id: str
```

---

## Communication Protocol

### Receives
- Data ingestion requests from orchestrator or CLI
- Symbol subscription updates from config

### Sends
- `OHLCVSnapshot` → to market-analyst (on each candle close)
- `PatternMatch` → to trading-strategist (on pattern detection)
- Raw data written to `data/{symbol}_{timeframe}.parquet`

---

## Storage Schema

```
data/
├── BTC_USDT_5m.parquet         # Main OHLCV store (columnar)
├── ETH_USDT_5m.parquet
├── patterns/
│   ├── BTC_USDT_patterns.bin   # HNSW index
│   └── ETH_USDT_patterns.bin
└── memory/
    └── trading-analysis.json   # Recent verdict history
```

---

## Error Handling

- Exchange timeout: retry 3x with exponential backoff, then failover to secondary
- Data gap > 4 hours: log warning, use forward-fill, flag `has_gaps = True`
- NaN values: forward-fill for price data, zero-fill for volume
- HNSW index corruption: rebuild from stored parquet files
- Never pass incomplete data to market-analyst — prefer delayed delivery over corrupt delivery

---

## Performance Expectations

- REST batch fetch (1000 candles): < 2 seconds
- WebSocket latency (candle close to agent): < 100ms
- HNSW query (top-10 pattern matches): < 10ms
- Full normalization pipeline: < 50ms per candle
