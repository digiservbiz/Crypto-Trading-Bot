# Skill: trader-risk

## Overview
VaR/CVaR/Kelly criterion risk assessment with circuit breaker enforcement. The blocking gate that every trade signal must pass before broker execution.

## Trigger
```
trader risk assess [options]
```

## Description
The `trader-risk` skill runs the `RiskAnalystAgent` to evaluate a proposed trade against all configured risk thresholds and circuit breakers. It is the ONLY component authorized to approve broker execution.

**INVARIANT: No trade may execute without an approved RiskDecision from this skill.**

## Risk Metrics Computed

### Value at Risk (VaR)
- Method: Historical simulation using last 252 periods
- Confidence: 95% (5th percentile of return distribution)
- Threshold: VaR ≤ 2% of portfolio value
- Formula: `VaR = -percentile(returns, 5) × position_value`

### Conditional VaR (CVaR / Expected Shortfall)
- Method: Mean of returns below VaR threshold
- Confidence: 95%
- Threshold: CVaR ≤ 3% of portfolio value
- Formula: `CVaR = -mean(returns[returns < VaR_threshold]) × position_value`

### Kelly Criterion Position Sizing
- Formula: `f* = (p × b - q) / b`
  - p = win probability (from signal confidence)
  - b = win/loss ratio (estimated from Sharpe)
- Apply: 25% of full Kelly (quarter-Kelly for safety)
- Cap: Never exceed `config.trading.max_position_size`

### Sharpe Ratio (Rolling 30-day)
- Full size if Sharpe > 1.5
- 50% size if Sharpe 1.0–1.5
- Reject if Sharpe < 1.0

## Circuit Breakers

| Breaker | Trigger | Action |
|---|---|---|
| Daily Loss Halt | Loss ≥ 3% of start-of-day balance | Reject all trades (resets UTC midnight) |
| Weekly Reduction | Loss ≥ 5% of start-of-week balance | 50% size reduction for 7 days |
| Correlation Spike | Pairwise correlation > 0.85 | Reject trades increasing correlation |
| Max Drawdown Halt | Portfolio drawdown ≥ 15% | Full halt (MANUAL RESET required) |
| VIX Spike | ATR ratio > 2x 90-day average | 75% size reduction |
| Concentration Limit | Single asset would exceed 10% | Reject further increases |

## Output

`RiskDecision`:

```json
{
  "signal_id": "7e4d1a9f",
  "decision": "approve",
  "adjusted_size_pct": 0.031,
  "var_95": 0.0148,
  "cvar_95": 0.0221,
  "sharpe_ratio": 1.87,
  "circuit_breaker_triggered": false,
  "circuit_breaker_reason": null
}
```

## Usage Examples

```bash
# Assess risk for current portfolio
trader risk assess

# Specific trade assessment
trader risk assess --symbol BTC/USDT --side buy --size 0.05

# View circuit breaker state
trader risk assess --portfolio

# Check risk state
cat data/memory/trading-risk.json
```

## Persistent State

Circuit breaker state survives restarts via `data/memory/trading-risk.json`:

```json
{
  "daily_loss_pct": -0.012,
  "weekly_loss_pct": -0.031,
  "current_drawdown_pct": -0.047,
  "max_pairwise_correlation": 0.71,
  "circuit_breaker_log": [...]
}
```

## Memory Namespace
`trading-risk` — exclusively owned by risk-analyst. No other agent may write to this namespace.
