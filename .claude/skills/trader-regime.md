# Skill: trader-regime

## Overview
Market regime classification using multi-indicator fusion. Identifies the current market state to drive strategy selection in the trading pipeline.

## Trigger
```
trader regime [options]
```

## Description
The `trader-regime` skill runs the `MarketAnalystAgent` to classify current market conditions. It computes 6 technical indicators and fuses their signals into one of 6 regime classifications.

## Regime Types

| Regime | Primary Signals | Typical Strategy |
|---|---|---|
| `bull-trending` | ADX > 25, +DI > -DI, RSI 55–70 | momentum |
| `bear-trending` | ADX > 25, -DI > +DI, RSI 30–45 | momentum (short) |
| `ranging` | ADX < 20, BB width in 30th pct | mean-reversion |
| `high-volatility` | ATR > 2x 20-period avg | adaptive (reduced size) |
| `low-volatility` | ATR < 0.5x 20-period avg | mean-reversion |
| `transitioning` | ADX 20–25, mixed signals | none (wait) |

## Indicator Computation

| Indicator | Period | Purpose |
|---|---|---|
| RSI | 14 | Momentum oscillator |
| MACD | 12,26,9 | Trend + momentum |
| Bollinger Bands | 20,2 | Volatility range |
| ADX + DI | 14 | Trend strength/direction |
| ATR | 14 | Absolute volatility |
| OBV | - | Volume/price divergence |

## Confidence Scoring

```
baseline = 0.50
+ each indicator supporting regime: +0.08 to +0.15
- each indicator contradicting: -0.05 to -0.12
clamp to [0.0, 1.0]

if confidence < 0.4: override to "transitioning"
```

## Output

`RegimeVerdict`:

```json
{
  "regime_type": "bull-trending",
  "confidence": 0.82,
  "indicator_values": {
    "adx": 31.4,
    "plus_di": 28.7,
    "minus_di": 14.2,
    "rsi": 61.2,
    "macd": 0.0043,
    "macd_signal": 0.0021,
    "macd_hist": 0.0022,
    "bb_width": 0.028,
    "bb_pct_b": 0.71,
    "atr": 245.3,
    "atr_ratio": 1.12,
    "obv_trend": 0.65
  },
  "symbols": ["BTC/USDT"],
  "verdict_id": "a3f9b12c"
}
```

## Usage Examples

```bash
# Analyze current BTC regime
trader regime --symbol BTC/USDT

# Verbose with all indicator values
trader regime --symbol BTC/USDT --verbose

# Multiple symbols
trader regime --symbols BTC/USDT,ETH/USDT

# Show recent history
trader regime --symbol BTC/USDT --history 10
```

## Performance Expectations
- Regime classification: < 500ms for 60-candle window
- Indicator computation: Tier 1 (fast path)
- No neural model calls

## Memory Namespace
`trading-analysis` — stores rolling RegimeVerdict history (last 100).
