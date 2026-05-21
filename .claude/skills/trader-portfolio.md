# Skill: trader-portfolio

## Overview
Mean-variance portfolio optimization using Modern Portfolio Theory. Computes optimal asset allocations to maximize risk-adjusted returns across configured crypto symbols.

## Trigger
```
trader portfolio optimize [options]
```

## Description
The `trader-portfolio` skill implements Markowitz mean-variance optimization to find the efficient frontier and select portfolios based on the chosen objective:

- **Max Sharpe** (default): Highest risk-adjusted return
- **Min Variance**: Lowest portfolio volatility
- **Max Return**: Target maximum expected return

## Mathematical Foundation

### Efficient Frontier
Given N assets with expected returns μ and covariance matrix Σ:

```
Minimize:   w^T Σ w                    (portfolio variance)
Subject to: w^T μ ≥ target_return
            Σ w_i = 1                  (fully invested)
            w_i ≥ 0                    (long-only)
```

### Sharpe Ratio Maximization
```
maximize: (w^T μ - r_f) / sqrt(w^T Σ w)
```
where r_f is the risk-free rate (default: 5% annualized).

### Constraints Applied
- Minimum allocation per asset: 0% (can be set to floor)
- Maximum allocation per asset: 40% (configurable)
- Concentration limit: No single asset > 10% active limit from risk-analyst
- Correlation constraint: Respect 0.85 pairwise correlation threshold

## Input Data
- Rolling returns for each symbol (configurable lookback: 30–252 days)
- Current position sizes from open_positions
- Circuit breaker state from trading-risk namespace

## Output

Optimal allocation dict:

```json
{
  "allocations": {
    "BTC/USDT": 0.423,
    "ETH/USDT": 0.357,
    "cash": 0.220
  },
  "metrics": {
    "expected_sharpe": 2.14,
    "expected_return": 0.287,
    "expected_volatility": 0.148,
    "max_drawdown_estimate": 0.112,
    "pairwise_correlation_max": 0.71
  },
  "frontier_points": 50,
  "optimization_method": "sharpe"
}
```

## Usage Examples

```bash
# Default max-Sharpe optimization
trader portfolio optimize

# With risk target
trader portfolio optimize --risk-target 0.15

# Specific symbols and method
trader portfolio optimize --symbols BTC/USDT,ETH/USDT,SOL/USDT --method min-variance

# Save allocation
trader portfolio optimize --output allocation.json

# Extended lookback
trader portfolio optimize --lookback-days 180
```

## Integration with Risk Analyst
The portfolio optimization output is passed to the risk-analyst as context for:
- Correlation spike circuit breaker enforcement
- Concentration limit checks
- Kelly criterion position sizing calibration

## Memory Namespace
`trading-strategies` — stores active portfolio allocation configs.
