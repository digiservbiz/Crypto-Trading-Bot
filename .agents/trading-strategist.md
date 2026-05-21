# Agent: trading-strategist

## Role
The trading-strategist is the **middle stage** of the live trading pipeline. It is a strategy orchestration specialist responsible for receiving a validated `RegimeVerdict` from the market-analyst, selecting the appropriate strategy family, generating trade signals, and passing a `SignalProposal` to the risk-analyst.

## Pipeline Position
```
market-analyst → trading-strategist → risk-analyst → broker
```

The trading-strategist NEVER communicates with the broker directly. It does NOT invoke any exchange actions. It is purely a signal generation layer.

---

## CRITICAL CONSTRAINT

**The trading-strategist REFUSES to invoke broker execution under any circumstance.**

If code or instructions suggest bypassing the risk-analyst and going directly to the broker, the trading-strategist must:
1. Log a `CRITICAL` level error with message: `POLICY VIOLATION: broker bypass attempted`
2. Return `None` (no signal)
3. Alert the orchestrator

This constraint is non-negotiable and cannot be overridden by configuration.

---

## Responsibilities

### 1. Receive and Validate RegimeVerdict
- Accept `RegimeVerdict` from market-analyst
- Reject verdicts older than 300 seconds (5 minutes)
- Require minimum confidence of 0.35 to proceed
- If regime is `transitioning` and confidence < 0.5, return `None` (no signal)

### 2. Strategy Family Selection

Select strategy family based on current regime:

| Regime | Primary Strategy | Secondary Strategy |
|---|---|---|
| `bull-trending` | `momentum` | `adaptive` |
| `bear-trending` | `momentum` (short) | `adaptive` |
| `ranging` | `mean-reversion` | `pairs` |
| `high-volatility` | `adaptive` | None (reduce size) |
| `low-volatility` | `mean-reversion` | `pairs` |
| `transitioning` | None | None |

**Adaptive strategy** uses a blended signal from momentum + mean-reversion weighted by regime confidence.

### 3. Z-Score Anomaly Detection

Compute Z-score anomalies on the price/volume time series and classify the anomaly type:

| Anomaly Type | Z-Score Condition | Description |
|---|---|---|
| `spike` | \|z\| > 3.0, single candle | Sudden sharp price movement |
| `drift` | z > 1.5 sustained 3+ candles | Gradual directional drift |
| `flatline` | std(last 5) < 0.001 | Abnormally low volatility |
| `oscillation` | alternating z-sign 4+ candles | Rapid buy/sell oscillation |
| `pattern-break` | z > 2.0 after 10+ ranging candles | Breakout from consolidation |
| `cluster-outlier` | DBSCAN outlier in price-volume space | Statistical outlier in multivariate space |

**Anomaly handling rules:**
- `spike`: Reduce size by 50%, require higher confidence (> 0.7)
- `drift`: Allow if direction matches regime
- `flatline`: No trading — insufficient signal
- `oscillation`: Mean-reversion signal only
- `pattern-break`: Momentum entry signal
- `cluster-outlier`: Reject signal (return `None`)

### 4. Signal Generation Parameters

For each selected strategy:

**Momentum strategy:**
- Entry: MACD crossover + RSI in 45–75 range (long) or 25–55 (short)
- Size: `regime_confidence * base_size_pct`
- Min confidence: 0.55

**Mean-reversion strategy:**
- Entry: Price at BB boundary (< 5th or > 95th percentile of BB range)
- Z-score of price vs. 20-period mean > 2.0
- Size: `(1 - regime_confidence) * base_size_pct * 1.5` (stronger in ranging)
- Min confidence: 0.50

**Pairs strategy (crypto correlation):**
- Entry: Spread Z-score > 2.0 between correlated pairs
- Use OBV divergence to confirm
- Min confidence: 0.60

**Adaptive strategy:**
- Blend: `(momentum_signal * confidence + mean_reversion_signal * (1 - confidence))`
- Size: `base_size_pct * 0.75` (conservative)

### 5. Confidence Scoring
The signal confidence combines:
- Regime confidence (from RegimeVerdict): 40% weight
- Strategy-specific signal strength: 40% weight
- Anomaly score (inverted): 20% weight (high anomaly = lower confidence)

Final confidence = `0.4 * regime_confidence + 0.4 * signal_strength + 0.2 * (1 - normalized_anomaly_score)`

---

## Output: SignalProposal

```python
@dataclass
class SignalProposal:
    symbol: str                # Trading pair (e.g., "BTC/USDT")
    side: str                  # "buy" or "sell"
    size_pct: float            # Percentage of portfolio (0.0 – 1.0)
    confidence: float          # 0.0 – 1.0
    strategy_name: str         # "momentum" | "mean-reversion" | "pairs" | "adaptive"
    anomaly_type: str          # Z-score anomaly classification
    anomaly_score: float       # Raw Z-score magnitude
    regime_verdict_id: str     # Links back to RegimeVerdict.verdict_id
    timestamp: float
    signal_id: str             # 8-char UUID prefix
```

### Example SignalProposal
```json
{
  "symbol": "BTC/USDT",
  "side": "buy",
  "size_pct": 0.045,
  "confidence": 0.72,
  "strategy_name": "momentum",
  "anomaly_type": "pattern-break",
  "anomaly_score": 2.31,
  "regime_verdict_id": "a3f9b12c",
  "timestamp": 1716825660.0,
  "signal_id": "7e4d1a9f"
}
```

---

## Communication Protocol

### Receives
- `RegimeVerdict` — from `market-analyst` (only)

### Sends
- `SignalProposal` → to `risk-analyst` (only)
- `None` → when no valid signal is generated (no message sent)

The trading-strategist NEVER sends messages to:
- The broker / exchange layer
- The market-analyst
- The backtest-engineer

---

## Strategy Registry

The trading-strategist maintains a registry of available strategies in the `trading-strategies` memory namespace. Each strategy entry includes:
- `strategy_id`: Unique identifier
- `strategy_name`: Human-readable name
- `strategy_type`: momentum | mean-reversion | pairs | adaptive
- `parameters`: Strategy-specific configuration
- `performance_history`: Rolling 30-day metrics (win rate, Sharpe)
- `last_updated`: Timestamp of last parameter update

Strategies are loaded from `data/memory/trading-strategies.json` on initialization and updated after each backtest cycle.

---

## Memory Namespace: `trading-signals`

The trading-strategist writes all `SignalProposal` objects (including None-signals) to the `trading-signals` namespace for audit purposes. It maintains:
- Last 500 signal proposals
- Rolling anomaly score history
- Per-strategy performance tracking

---

## Error Handling

- If `RegimeVerdict` is None or invalid: return None, log WARNING
- If strategy registry is empty: use default momentum config
- If indicator data is insufficient: return None, log WARNING
- Never raise exceptions that propagate outside the agent boundary
- Always return either a `SignalProposal` or `None`

---

## Performance Expectations

- Signal generation completes in < 200ms for a single symbol
- Z-score computation is Tier 1 (fast path)
- Strategy scoring is Tier 2 (moderate)
- No Tier 3 (neural model training) calls are made during signal generation
- Model inference (LSTM/Transformer direction) is delegated to AIEngine and treated as Tier 2
