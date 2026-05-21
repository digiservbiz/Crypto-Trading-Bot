# Skill: trader-signal

## Overview
Z-score anomaly detection signal generation. Runs the market-analyst → trading-strategist pipeline stages to produce trade signals without executing any broker actions.

## Trigger
```
trader signal scan [options]
```

## Description
The `trader-signal` skill coordinates two pipeline agents to generate trade signals:

1. **market-analyst** — Computes regime verdict from OHLCV data
2. **trading-strategist** — Generates signal proposal from regime

It does NOT invoke the risk-analyst or broker. Signal output is for informational scanning only, unless integrated into the full pipeline via the orchestrator.

## Anomaly Detection

Z-score anomaly types detected:

| Type | Condition | Tradeable |
|---|---|---|
| `spike` | \|z\| > 3.0, single candle | Yes (reduced size, higher confidence req.) |
| `drift` | z > 1.5 sustained 3+ candles | Yes (direction match required) |
| `flatline` | std(last 5) < 0.001 | No |
| `oscillation` | Alternating z-sign 4+ candles | Yes (mean-reversion only) |
| `pattern-break` | z > 2.0 after 10+ ranging candles | Yes (momentum entry) |
| `cluster-outlier` | DBSCAN multivariate outlier | No |
| `none` | Normal market activity | Yes |

## Signal Confidence Formula

```
confidence = 0.40 × regime_confidence
           + 0.40 × strategy_signal_strength
           + 0.20 × (1 - normalized_anomaly_score)
```

## Strategy Selection by Regime

| Regime | Strategy | Condition |
|---|---|---|
| bull-trending | momentum | MACD > 0 + RSI 45–75 |
| bear-trending | momentum (short) | MACD < 0 + RSI 25–55 |
| ranging | mean-reversion | BB boundary + z > 2.0 |
| high-volatility | adaptive | Blended, reduced size |
| low-volatility | mean-reversion | BB boundary |
| transitioning | none | No signal |

## Output

`SignalProposal` per symbol:

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
  "signal_id": "7e4d1a9f"
}
```

## Usage Examples

```bash
# Scan all configured symbols
trader signal scan

# Filter by strategy and min confidence
trader signal scan --strategy momentum --min-confidence 0.65

# Specific symbols
trader signal scan --symbols BTC/USDT,ETH/USDT

# Regime filter
trader signal scan --regime bull-trending

# Save to JSON
trader signal scan --output-json signals.json
```

## Memory Namespace
`trading-signals` — signal history and anomaly score log.
