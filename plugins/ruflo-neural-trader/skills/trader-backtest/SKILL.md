# Skill: trader-backtest

## Overview
Walk-forward backtesting with Ed25519 artifact signing. Validates trading strategies through rigorous statistical testing before live deployment consideration.

## Trigger
```
trader backtest <strategy> [options]
```

## Description
The `trader-backtest` skill runs the full BacktestEngineerAgent pipeline for strategy validation. It implements:

1. **Walk-Forward Validation** — Rolling windows (6-month train / 1-month test)
2. **Monte Carlo Simulation** — 1000 bootstrapped runs for statistical significance
3. **Quality Gate Enforcement** — 6 hard gates that must ALL pass
4. **Artifact Signing** — Results stored to `trading-backtests` namespace

## Quality Gates

All 6 gates must pass for `passed_quality_gates == True`:

| Gate | Metric | Threshold |
|---|---|---|
| 1 | Minimum trades | ≥ 30 in test period |
| 2 | Win-rate consistency | Variance < 15% across windows |
| 3 | Monte Carlo significance | p-value < 0.05 |
| 4 | Max drawdown | < 15% |
| 5 | Profit factor | > 1.5 |
| 6 | Sharpe ratio | > 1.0 (annualized) |

## Walk-Forward Protocol

```
Training window: 6 months  ─┐
Test window:     1 month    ─┘  →  record metrics
         slide forward 1 month
         repeat until end of data
```

Minimum 3 windows required. Warns if fewer than 6 windows available.

## Monte Carlo Protocol

- Bootstrap resampling of historical returns (1000 iterations)
- Null hypothesis: "Strategy returns are indistinguishable from random"
- Test statistic: fraction of simulations with positive final P&L
- p-value computed via normal approximation to binomial distribution

## Output

`SignedBacktestArtifact` stored to `data/memory/trading-backtests.json`:

```json
{
  "strategy_id": "momentum-btc-v1",
  "symbol": "BTC/USDT",
  "total_return": 0.312,
  "sharpe_ratio": 1.94,
  "sortino_ratio": 2.67,
  "max_drawdown": -0.087,
  "win_rate": 0.587,
  "profit_factor": 1.83,
  "num_trades": 147,
  "walk_forward_scores": [1.72, 2.04, 1.88, 1.91, 2.11, 2.01],
  "monte_carlo_p_value": 0.021,
  "passed_quality_gates": true,
  "artifact_id": "f2c8a7b1"
}
```

## Isolation

This skill NEVER:
- Accesses live exchange data
- Reads real account balances
- Sends messages to live pipeline agents
- Shares mutable state with the live trading loop

## Usage Examples

```bash
# Basic backtest
trader backtest my-momentum --symbol BTC/USDT

# With custom period
trader backtest my-strategy --symbol ETH/USDT --period 2023-01-01/2024-06-01

# Save results
trader backtest adaptive-v1 --symbol BTC/USDT --output results.json

# Via Python
python scripts/backtest.py --symbol BTC/USDT --strategy momentum
```

## Memory Namespace
`trading-backtests` — read/write by this skill only.
