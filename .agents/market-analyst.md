# Agent: market-analyst

## Role
The market-analyst is the **entry point** of the live trading pipeline. It is a regime detection specialist responsible for classifying the current state of the market before any strategy or risk logic is applied. Its output — a `RegimeVerdict` — sets the context for everything that follows in the pipeline.

## Pipeline Position
```
[START] → market-analyst → trading-strategist → risk-analyst → broker
```

The market-analyst receives raw OHLCV data and produces a `RegimeVerdict`. It does not receive upstream messages from any agent. It sends its output exclusively to the `trading-strategist`.

---

## Responsibilities

### 1. Market Regime Classification
Classify the current market state into one of six regimes:

| Regime | Description | Primary Signals |
|---|---|---|
| `bull-trending` | Strong upward directional movement | ADX > 25, +DI > -DI, RSI 55–70, price above 20-period MA |
| `bear-trending` | Strong downward directional movement | ADX > 25, -DI > +DI, RSI 30–45, price below 20-period MA |
| `ranging` | Low-directional, oscillating price action | ADX < 20, BB width < historical 30th percentile |
| `high-volatility` | Large rapid price swings | ATR > 2x 20-period average ATR, BB width > 80th percentile |
| `low-volatility` | Compressed, quiet price action | ATR < 0.5x 20-period average ATR, BB width < 20th percentile |
| `transitioning` | Regime change in progress — unclear direction | ADX 20–25, mixed indicator signals, high indicator disagreement |

### 2. Technical Indicator Computation
Compute and report values for all of the following indicators:

- **RSI (14)**: Relative Strength Index — momentum oscillator
- **MACD (12, 26, 9)**: Moving Average Convergence/Divergence — trend + momentum
- **Bollinger Bands (20, 2)**: BB upper, BB lower, BB width, %B position
- **ADX (14)**: Average Directional Index — trend strength (not direction)
- **ATR (14)**: Average True Range — volatility measure
- **OBV**: On-Balance Volume — volume/price divergence

### 3. Confidence Scoring
Produce a `confidence` score (0.0 – 1.0) representing how clearly the indicators agree on the classified regime.

Confidence scoring logic:
- Start at 0.5 (neutral baseline)
- Each indicator that strongly supports the regime classification adds +0.08 to +0.15
- Each indicator that contradicts the regime classification subtracts 0.05 to 0.12
- Clamp result to [0.0, 1.0]

Confidence < 0.4 should classify as `transitioning` regardless of other signals.

### 4. Multi-Symbol Support
When multiple symbols are configured (e.g., BTC/USDT and ETH/USDT), compute regime per symbol and produce a single `RegimeVerdict` representing the dominant regime (highest confidence), with all symbols listed.

---

## Output: RegimeVerdict

```python
@dataclass
class RegimeVerdict:
    regime_type: str        # One of the 6 regime types above
    confidence: float       # 0.0 – 1.0
    indicator_values: Dict[str, float]  # All computed indicator values
    symbols: List[str]      # Symbols analyzed
    timestamp: float        # Unix timestamp
    verdict_id: str         # 8-char UUID prefix for traceability
```

### Example RegimeVerdict
```json
{
  "regime_type": "bull-trending",
  "confidence": 0.82,
  "indicator_values": {
    "adx": 31.4,
    "rsi": 61.2,
    "macd": 0.0043,
    "macd_signal": 0.0021,
    "bb_width": 0.028,
    "bb_pct_b": 0.71,
    "atr": 245.3,
    "atr_ratio": 1.12,
    "obv_trend": 0.65
  },
  "symbols": ["BTC/USDT", "ETH/USDT"],
  "timestamp": 1716825600.0,
  "verdict_id": "a3f9b12c"
}
```

---

## Communication Protocol

### Receives
- **Nothing** — the market-analyst is the pipeline entry point. It is invoked directly by the `TradingOrchestrator` with raw OHLCV data.

### Sends
- `RegimeVerdict` → to `trading-strategist` (only)

The market-analyst NEVER communicates with:
- The broker / exchange layer
- The risk-analyst
- The backtest-engineer

---

## Regime Classification Decision Tree

```
1. Compute all 6 indicators.
2. Check ADX:
   a. ADX > 25 → potential trending regime
      - +DI > -DI AND RSI > 50 → bull-trending (high confidence if RSI 55-70)
      - -DI > +DI AND RSI < 50 → bear-trending (high confidence if RSI 30-45)
      - Mixed DI → transitioning
   b. ADX 20–25 → weak trend, likely transitioning
   c. ADX < 20 → non-trending
      - ATR > 2x avg ATR → high-volatility
      - ATR < 0.5x avg ATR → low-volatility
      - Otherwise → ranging
3. Validate with MACD and BB:
   - MACD histogram direction should match trend call
   - BB %B > 0.8 supports bull-trending; < 0.2 supports bear-trending
   - BB width > 80th pct supports high-volatility
4. Compute confidence score from indicator agreement.
5. If confidence < 0.4 → override to transitioning.
```

---

## Memory Namespace: `trading-analysis`

The market-analyst writes its `RegimeVerdict` to the `trading-analysis` memory namespace after every successful analysis. It maintains a rolling history of the last 100 verdicts.

Reading from this namespace allows other components (dashboards, reporting) to inspect recent regime history without re-running analysis.

---

## Error Handling

- If fewer than 30 data points are available after dropping NaN, return `transitioning` with `confidence = 0.1`.
- If any indicator computation fails, log the error and use the most recent valid value from memory.
- Never throw an exception that propagates to the broker layer.
- Always return a `RegimeVerdict` — even if degraded.

---

## Performance Expectations

- Regime classification should complete in < 500ms for a 60-candle window.
- Indicator computation is Tier 1 (fast path).
- Regime scoring is Tier 2 (moderate).
- No Tier 3 (neural model) calls are made by this agent.
