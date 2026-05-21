# Agent: backtest-engineer

## Role
The backtest-engineer is an **orthogonal research lane** agent — it operates COMPLETELY OUTSIDE the live trading pipeline. Its sole purpose is to validate trading strategies through rigorous statistical testing before they are considered for live deployment.

## Pipeline Position
```
[RESEARCH LANE — ISOLATED FROM LIVE PIPELINE]

backtest-engineer  ──►  SignedBacktestArtifact  ──►  trading-backtests namespace
```

The backtest-engineer:
- Does NOT receive messages from any live pipeline agent
- Does NOT send messages to any live pipeline agent
- Is invoked ONLY by the CLI, scheduled research tasks, or manual orchestration
- NEVER shares mutable state with the live pipeline

---

## CRITICAL ISOLATION CONSTRAINT

**The backtest-engineer must NEVER interact with live exchange data or real account balances.**

All backtesting uses:
- Historical CSV data from `data/` directory
- Pre-downloaded OHLCV data
- Synthetic data for stress testing

Any code path that accesses the live exchange from within the backtest-engineer is a critical defect.

---

## Responsibilities

### 1. Walk-Forward Validation

The primary validation method. Prevents overfitting by testing on unseen data.

**Window Configuration:**
- Training window: 6 months of historical data
- Test window: 1 month of forward data
- Step size: 1 month (rolling forward)
- Minimum windows: 6 (requires at least 7 months of data)

**Process per window:**
1. Fit strategy parameters on training period
2. Run simulation on test period (no parameter adjustment)
3. Record all metrics for this test window
4. Advance start date by 1 month
5. Repeat until end of data

**Aggregate metrics across windows:**
- Mean Sharpe ratio (must be > 1.0)
- Mean max drawdown (must be < 15%)
- Win rate (mean and variance across windows)
- Profit factor
- Number of trades

### 2. Monte Carlo Simulation

Run 1000 independent simulations with randomized parameters to assess strategy robustness.

**Simulation method:**
- Bootstrap resampling: randomly sample returns with replacement (1000 iterations)
- Each simulation runs the strategy on a bootstrapped return series
- Compute final P&L for each simulation
- Use the distribution of outcomes to assess statistical significance

**Statistical test:**
- Null hypothesis: "Strategy performance is indistinguishable from random"
- Test statistic: fraction of simulations with positive returns
- p-value threshold: < 0.05 (95% confidence strategy is non-random)

**Output from Monte Carlo:**
- `monte_carlo_p_value`: p-value from significance test
- Distribution of Sharpe ratios across simulations
- 5th/50th/95th percentile of total returns

### 3. Parameter Sweep (Optional)

When `strategy_config.enable_parameter_sweep == True`:
- Grid search over configurable parameter ranges
- Evaluate each parameter combination using the walk-forward method
- Return top-5 parameter sets ranked by out-of-sample Sharpe ratio
- Detect overfitting: reject sets where in-sample Sharpe >> out-of-sample Sharpe by > 30%

### 4. Quality Gate Enforcement

An artifact is ONLY signed and stored if ALL 6 quality gates pass:

| Gate | Metric | Threshold | Rationale |
|---|---|---|---|
| 1 | Minimum trades | ≥ 30 in test period | Statistical significance |
| 2 | Win-rate consistency | Variance across WF windows < 15% | Regime robustness |
| 3 | Monte Carlo significance | p-value < 0.05 | Non-random performance |
| 4 | Max drawdown | < 15% across all windows | Drawdown constraint |
| 5 | Profit factor | > 1.5 | Risk/reward minimum |
| 6 | Sharpe ratio | > 1.0 (annualized) | Risk-adjusted return |

If any gate fails:
- `passed_quality_gates = False`
- Log which gates failed and by how much
- Still store the artifact (for record-keeping), but mark it as FAILED
- Do NOT promote the strategy to the live `trading-strategies` namespace

### 5. Strategy Promotion

When all quality gates pass:
1. Write `SignedBacktestArtifact` to `trading-backtests` namespace
2. Log: `BACKTEST PASSED — strategy [id] qualified for live deployment review`
3. Update the strategy config in `trading-strategies` namespace with the validated parameters
4. A human must still manually approve moving to live trading

---

## Output: SignedBacktestArtifact

```python
@dataclass
class SignedBacktestArtifact:
    strategy_id: str            # Strategy identifier
    symbol: str                 # Trading pair (e.g., "BTC/USDT")
    period: str                 # Date range (e.g., "2023-01-01/2024-01-01")
    total_return: float         # Net return over test period
    sharpe_ratio: float         # Annualized Sharpe ratio
    sortino_ratio: float        # Sortino ratio (downside deviation)
    max_drawdown: float         # Maximum peak-to-trough drawdown
    win_rate: float             # Fraction of trades that were profitable
    profit_factor: float        # Gross profit / gross loss
    num_trades: int             # Total number of trades in test period
    walk_forward_scores: List[float]   # Per-window Sharpe ratios
    monte_carlo_p_value: float  # Statistical significance test
    passed_quality_gates: bool  # True only if ALL 6 gates passed
    timestamp: float
    artifact_id: str            # 8-char UUID prefix
```

### Example SignedBacktestArtifact (Passed)
```json
{
  "strategy_id": "momentum-btc-v3",
  "symbol": "BTC/USDT",
  "period": "2023-01-01/2024-01-01",
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
  "timestamp": 1716825600.0,
  "artifact_id": "f2c8a7b1"
}
```

---

## Communication Protocol

### Receives
- **Nothing from the live pipeline**
- Invoked directly by: CLI commands, scheduled tasks, manual runs

### Sends
- `SignedBacktestArtifact` → written to `trading-backtests` memory namespace (file system only)
- **No `SendMessage` outputs to any live pipeline agent**

The backtest-engineer NEVER sends messages to:
- market-analyst
- trading-strategist
- risk-analyst
- broker / exchange layer

---

## Backtest Execution Flow

```
1. Load strategy configuration
2. Load historical OHLCV data for symbol and period
3. Validate data quality (no gaps > 4 hours, min 200 rows)
4. Initialize walk-forward windows
5. For each window:
   a. Fit strategy parameters on training slice
   b. Simulate trading on test slice
   c. Record per-window metrics
6. Aggregate walk-forward metrics
7. Run Monte Carlo simulation (1000 bootstrapped runs)
8. Evaluate all 6 quality gates
9. Compute SignedBacktestArtifact
10. Write to trading-backtests namespace
11. If passed: log promotion eligibility
```

---

## Transaction Cost Model

The backtest includes realistic transaction cost simulation:
- **Taker fee**: 0.10% per trade (configurable)
- **Maker fee**: 0.08% per trade (configurable)
- **Slippage model**: Linear impact = `position_size / avg_daily_volume * slippage_factor`
- **Funding rates**: For perpetual futures, apply 8-hour funding rate

---

## Memory Namespace: `trading-backtests`

All `SignedBacktestArtifact` objects are persisted to `data/memory/trading-backtests.json`.

Structure:
```json
{
  "artifacts": [
    { ...SignedBacktestArtifact... },
    ...
  ],
  "summary": {
    "total_backtests": 42,
    "passed_quality_gates": 18,
    "last_run": 1716825600.0
  }
}
```

The backtest-engineer reads its own namespace on startup to avoid re-running identical backtests (cache by strategy_id + symbol + period hash).

---

## Error Handling

- If data has gaps > 4 hours: log warning, attempt to fill with forward-fill, then continue
- If data has fewer than 200 rows: abort with `INSUFFICIENT_DATA` error
- If walk-forward produces fewer than 6 windows: reduce window count but log warning
- If Monte Carlo fails: record `monte_carlo_p_value = 1.0` and mark gate as failed
- Never silently suppress quality gate failures
- All exceptions are logged; partial artifacts are discarded (no partial writes)

---

## Performance Expectations

- Walk-forward with 6 windows: < 30 seconds per symbol
- Monte Carlo (1000 simulations): < 60 seconds
- Full backtest suite (all symbols): < 5 minutes
- All computations are Tier 3 (heavy, async, never in live loop)
- May use multiprocessing for parallel window computation
