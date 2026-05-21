# Skill: market-pattern

## Overview
Candlestick pattern vectorization and HNSW-indexed similarity search. Converts candlestick patterns to 15-dimensional vectors and maintains an approximate nearest-neighbor index for fast pattern matching.

## Trigger
```
market patterns [options]
market search [options]
```

## Description
The `market-pattern` skill processes stored OHLCV data to extract, vectorize, and index candlestick patterns. It enables fast similarity search to find historical patterns that match the current market state, along with their forward return outcomes.

## Pattern Vectorization

Each 5-candle window is vectorized into a 15-dimensional vector:

```
v = [
  # Candle body ratios
  (close[t] - open[t]) / open[t],    # Candle 1 body
  (high[t] - low[t]) / open[t],      # Candle 1 range
  volume[t] / mean_volume,           # Candle 1 vol ratio
  # ... repeated for 5 candles
]
```

All vectors are L2-normalized to the unit sphere (cosine similarity).

## Supported Patterns

### Single-Candle Patterns
| Pattern | Condition |
|---|---|
| `doji` | body < 0.1 × range |
| `hammer` | lower shadow > 2 × body, small upper shadow |
| `hanging-man` | same as hammer in downtrend |
| `marubozu-bull` | close ≈ high, open ≈ low |
| `marubozu-bear` | close ≈ low, open ≈ high |
| `spinning-top` | body < 0.3 × range, balanced shadows |

### Two-Candle Patterns
| Pattern | Condition |
|---|---|
| `engulfing-bull` | candle 2 body engulfs candle 1 bearish body |
| `engulfing-bear` | candle 2 body engulfs candle 1 bullish body |

### Three-Candle Patterns
| Pattern | Condition |
|---|---|
| `morning-star` | bearish → doji/small → bullish |
| `evening-star` | bullish → doji/small → bearish |
| `three-white-soldiers` | 3 consecutive bullish candles, each closing higher |
| `three-black-crows` | 3 consecutive bearish candles, each closing lower |

## HNSW Index

The index uses Hierarchical Navigable Small World graphs for approximate nearest-neighbor search:

```python
import hnswlib

# Index construction
dim = 15
index = hnswlib.Index(space='cosine', dim=dim)
index.init_index(max_elements=50000, ef_construction=200, M=16)
index.add_items(pattern_vectors, pattern_ids)

# Query
labels, distances = index.knn_query(query_vector, k=10)
```

**Index parameters:**
- Space: cosine
- Dimensions: 15
- EF construction: 200
- M (connections): 16
- Max elements: 50,000 per symbol

**Query performance:**
- Top-10 nearest neighbors: < 10ms
- Index construction (10k vectors): < 5 seconds

## Pattern Database

Each indexed pattern stores:

```json
{
  "pattern_id": 42,
  "symbol": "BTC/USDT",
  "pattern_type": "engulfing-bull",
  "timestamp": 1714176000,
  "vector": [0.023, -0.041, 1.24, ...],
  "forward_return_5": 0.032,
  "forward_return_10": 0.047,
  "forward_return_20": 0.019
}
```

## Similarity Search Output

```json
{
  "query_symbol": "BTC/USDT",
  "query_timestamp": 1716825600,
  "current_pattern_type": "engulfing-bull",
  "matches": [
    {
      "timestamp": 1707912000,
      "similarity": 0.94,
      "pattern_type": "engulfing-bull",
      "forward_return_5": 0.032
    }
  ],
  "aggregate_stats": {
    "mean_forward_return": 0.024,
    "positive_fraction": 0.80,
    "n_matches": 5
  }
}
```

## Usage Examples

```bash
# Build pattern index
market patterns --symbol BTC/USDT

# Force rebuild
market patterns --symbol BTC/USDT --rebuild-index

# Search current patterns
market search --symbol BTC/USDT --top-k 10

# High similarity threshold
market search --symbol BTC/USDT --min-similarity 0.90 --show-outcomes

# All symbols
market patterns --all-symbols
```

## Storage

```
data/patterns/
├── BTC_USDT_patterns.bin      # HNSW binary index
├── BTC_USDT_patterns_meta.json  # Pattern metadata
└── ETH_USDT_patterns.bin
```

## Dependencies
- `hnswlib` — HNSW approximate nearest neighbor
- `numpy` — vector operations
- `pandas` — OHLCV data handling
