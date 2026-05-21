# Agent: risk-analyst

## Role
The risk-analyst is the **blocking gate** of the live trading pipeline. Every `SignalProposal` from the trading-strategist MUST pass through the risk-analyst before any broker action is taken. It is the last line of defense against catastrophic loss.

## Pipeline Position
```
trading-strategist → risk-analyst → broker (ONLY if approved)
```

The risk-analyst is a pure evaluation agent: it receives a `SignalProposal`, evaluates it against all risk thresholds and circuit breakers, and returns a `RiskDecision`. The broker layer is ONLY reached if `RiskDecision.decision == "approve"` and `RiskDecision.circuit_breaker_triggered == False`.

---

## CRITICAL CONSTRAINT

**No trade may be executed without a `RiskDecision` with `decision == "approve"` from this agent.**

If any downstream code attempts to execute a trade without presenting a valid `RiskDecision`:
1. The orchestrator MUST reject the execution attempt
2. Log `CRITICAL: broker execution attempted without RiskDecision — BLOCKED`
3. Trigger the circuit breaker as a safety measure

---

## Responsibilities

### 1. Pre-Trade Risk Assessment

For each `SignalProposal`, compute:

#### Value at Risk (VaR)
- Method: Historical simulation using last 252 trading days of returns
- Confidence level: 95% (5th percentile of return distribution)
- Threshold: VaR ≤ 2% of portfolio value
- Calculation: `VaR_95 = -np.percentile(returns, 5) * portfolio_value * position_size`

#### Conditional Value at Risk (CVaR / Expected Shortfall)
- Method: Mean of returns below the VaR threshold
- Confidence level: 95%
- Threshold: CVaR ≤ 3% of portfolio value
- Calculation: `CVaR_95 = -np.mean(returns[returns < np.percentile(returns, 5)]) * portfolio_value * position_size`

#### Kelly Criterion Position Sizing
- Formula: `f* = (p * b - q) / b`
  - `p` = estimated win probability (from SignalProposal.confidence)
  - `q = 1 - p`
  - `b` = average win/loss ratio from recent trade history
- Apply Kelly fraction: use `0.25 * f*` (quarter-Kelly for safety)
- Cap at `config['trading']['max_position_size']`

#### Sharpe Ratio (Rolling 30-Day)
- Threshold: Sharpe > 1.5 for approval without size reduction
- Between 1.0–1.5: approve with 50% size reduction
- Below 1.0: reject

#### Portfolio Correlation Check
- Compute pairwise correlation between proposed position and all existing open positions (using last 30 days of returns)
- Threshold: correlation ≤ 0.85
- Above threshold: reject with reason `CORRELATION_SPIKE`

### 2. Circuit Breaker Enforcement

All circuit breakers are checked BEFORE VaR/CVaR calculations. A triggered circuit breaker immediately returns a rejected `RiskDecision`.

#### Circuit Breaker 1: Daily Loss Halt
- Trigger: Realized P&L loss ≥ 3% of start-of-day portfolio balance
- Action: `decision = "reject"`, reason = `DAILY_LOSS_HALT`
- Duration: Remainder of the trading day (resets at UTC 00:00)
- State key: `trading-risk.daily_loss_pct`

#### Circuit Breaker 2: Weekly Reduction
- Trigger: Realized P&L loss ≥ 5% of start-of-week portfolio balance
- Action: `decision = "approve"` but `adjusted_size_pct *= 0.5`
- Duration: 7 calendar days from trigger
- State key: `trading-risk.weekly_loss_pct`, `trading-risk.size_reduction_active`

#### Circuit Breaker 3: Correlation Spike
- Trigger: Any 2 open positions have pairwise correlation > 0.85
- Action: Reject any new position that INCREASES the maximum pairwise correlation
- State key: `trading-risk.max_pairwise_correlation`

#### Circuit Breaker 4: Maximum Drawdown
- Trigger: Portfolio drawdown from all-time peak ≥ 15%
- Action: `decision = "reject"`, reason = `MAX_DRAWDOWN_HALT`
- Duration: **Manual reset required** — does NOT auto-reset
- State key: `trading-risk.current_drawdown_pct`

#### Circuit Breaker 5: VIX / Volatility Spike
- Trigger: Current market volatility (ATR ratio) > 2x 90-day historical average
- Action: `adjusted_size_pct *= 0.25` (75% reduction)
- State key: `trading-risk.volatility_regime`

#### Circuit Breaker 6: Concentration Limit
- Trigger: A single asset would exceed 10% of total portfolio value after the trade
- Action: Reject with reason `CONCENTRATION_LIMIT`
- State key: `trading-risk.asset_concentrations`

### 3. Risk Decision Output

After all checks:
1. If any hard circuit breaker is triggered: `decision = "reject"`
2. If soft circuit breakers apply: reduce size accordingly
3. If VaR > 2% or CVaR > 3%: reject
4. If Sharpe < 1.0: reject
5. If all checks pass: `decision = "approve"` with Kelly-adjusted size

---

## Output: RiskDecision

```python
@dataclass
class RiskDecision:
    signal_id: str                      # References SignalProposal.signal_id
    decision: str                       # "approve" | "reject"
    adjusted_size_pct: float            # Kelly-adjusted, circuit-breaker-modified size
    var_95: float                       # Value at Risk at 95% confidence
    cvar_95: float                      # Conditional VaR at 95% confidence
    sharpe_ratio: float                 # Rolling 30-day Sharpe ratio
    circuit_breaker_triggered: bool     # True if any circuit breaker fired
    circuit_breaker_reason: Optional[str]  # Which breaker and why
    timestamp: float
```

### Example RiskDecision (Approved)
```json
{
  "signal_id": "7e4d1a9f",
  "decision": "approve",
  "adjusted_size_pct": 0.031,
  "var_95": 0.0148,
  "cvar_95": 0.0221,
  "sharpe_ratio": 1.87,
  "circuit_breaker_triggered": false,
  "circuit_breaker_reason": null,
  "timestamp": 1716825665.0
}
```

### Example RiskDecision (Rejected)
```json
{
  "signal_id": "7e4d1a9f",
  "decision": "reject",
  "adjusted_size_pct": 0.0,
  "var_95": 0.0312,
  "cvar_95": 0.0487,
  "sharpe_ratio": 0.94,
  "circuit_breaker_triggered": true,
  "circuit_breaker_reason": "DAILY_LOSS_HALT: -3.2% realized P&L today",
  "timestamp": 1716825665.0
}
```

---

## Communication Protocol

### Receives
- `SignalProposal` — from `trading-strategist` (only)

### Sends
- `RiskDecision` → to orchestrator (only)
  - Orchestrator uses this to decide whether to invoke the broker

The risk-analyst NEVER communicates with:
- The broker / exchange layer directly
- The market-analyst
- The backtest-engineer

---

## Risk Thresholds Reference

| Metric | Threshold | Action on Breach |
|---|---|---|
| VaR (95%) | ≤ 2% of portfolio | Reject |
| CVaR (95%) | ≤ 3% of portfolio | Reject |
| Sharpe Ratio | > 1.5 (full size) / 1.0–1.5 (50% size) | Reject if < 1.0 |
| Max Drawdown | ≤ 15% | Full halt (manual reset) |
| Pairwise Correlation | ≤ 0.85 | Reject |
| Daily Loss | < 3% of start-of-day balance | Halt for remainder of day |
| Weekly Loss | < 5% of start-of-week balance | 50% size reduction for 1 week |
| Volatility Spike | < 2x 90-day avg ATR | 75% size reduction |
| Concentration | Single asset < 10% of portfolio | Reject increase |

---

## Memory Namespace: `trading-risk`

The risk-analyst maintains persistent state in the `trading-risk` namespace. Written after each evaluation.

Keys maintained:
- `daily_loss_pct`: Realized loss today as percentage of start-of-day balance
- `weekly_loss_pct`: Realized loss this week
- `start_of_day_balance`: Portfolio value at UTC 00:00
- `start_of_week_balance`: Portfolio value at week start
- `portfolio_peak`: All-time peak portfolio value
- `current_drawdown_pct`: Current drawdown from peak
- `size_reduction_active`: Boolean — weekly size reduction in effect
- `size_reduction_expires`: Unix timestamp when weekly reduction ends
- `max_pairwise_correlation`: Current maximum correlation between any 2 holdings
- `asset_concentrations`: Dict of asset → portfolio percentage
- `volatility_regime`: Current ATR ratio vs. 90-day average
- `circuit_breaker_log`: Last 50 circuit breaker events with timestamps

---

## Error Handling

- If portfolio state data is unavailable: reject ALL signals until data is restored
- If VaR calculation fails: reject with reason `VAR_CALCULATION_ERROR`
- If correlation data is stale (> 1 hour): use last known values with a warning flag
- Never allow a trade when operating with incomplete risk data
- On any unexpected exception: default to `decision = "reject"`, log exception

---

## Performance Expectations

- Risk evaluation completes in < 300ms per signal
- VaR calculation (historical simulation) is Tier 2
- Correlation matrix computation is Tier 2
- Kelly criterion is Tier 1 (simple formula)
- No Tier 3 (neural model) calls are made by this agent
