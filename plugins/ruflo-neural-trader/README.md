# ruflo-neural-trader Plugin

A comprehensive neural trading plugin for the ruflo multi-agent framework. Provides self-learning LSTM/Transformer/N-BEATS strategies, rigorous walk-forward backtesting, VaR/CVaR risk management, and multi-agent swarm coordination for crypto trading.

## Features

- **4 Specialized Agents**: market-analyst, trading-strategist, risk-analyst, backtest-engineer
- **7 Skills**: backtest, signal, portfolio, regime, train, risk, cloud-backtest
- **Neural Models**: LSTM, Transformer (N-BEATS planned)
- **Risk Management**: VaR/CVaR, Kelly criterion, 6 circuit breakers
- **Backtesting**: Walk-forward validation + Monte Carlo (1000 simulations)
- **Multi-Symbol**: Concurrent analysis of BTC/USDT, ETH/USDT, and more

---

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │        ruflo-neural-trader           │
                    └─────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    LIVE TRADING PIPELINE                        │
│                                                                 │
│  market-analyst  ──RegimeVerdict──►  trading-strategist         │
│        │                                    │                   │
│   [RSI, MACD,                         [Z-score,                 │
│  ADX, ATR, BB,                      momentum,                   │
│    OBV]                             mean-rev]                   │
│                                             │                   │
│                              SignalProposal │                   │
│                                             ▼                   │
│                                      risk-analyst               │
│                                             │                   │
│                                    [VaR, CVaR,                  │
│                                  Kelly, Sharpe,                  │
│                                  circuit-breakers]              │
│                                             │                   │
│                              RiskDecision   │                   │
│                                             ▼                   │
│                               broker (ONLY if approved)         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   RESEARCH LANE (Isolated)                      │
│                                                                 │
│  backtest-engineer  ──►  SignedBacktestArtifact                 │
│        │                         │                              │
│  [Walk-forward,             trading-backtests                   │
│   Monte Carlo,                 namespace                        │
│   Quality gates]                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Agents

### market-analyst
Entry point of the live pipeline. Classifies market regime from OHLCV data.

- **Input**: Raw OHLCV data
- **Output**: `RegimeVerdict` (regime_type, confidence, indicator_values)
- **Regimes**: bull-trending, bear-trending, ranging, high-volatility, low-volatility, transitioning
- **Indicators**: RSI(14), MACD(12,26,9), BB(20,2), ADX(14), ATR(14), OBV

### trading-strategist
Strategy orchestration and signal generation.

- **Input**: `RegimeVerdict`
- **Output**: `SignalProposal` or None
- **Strategies**: momentum, mean-reversion, pairs, adaptive
- **Anomalies**: spike, drift, flatline, oscillation, pattern-break, cluster-outlier
- **CRITICAL**: Never invokes broker — always routes through risk-analyst

### risk-analyst
Blocking gate before any broker execution.

- **Input**: `SignalProposal`
- **Output**: `RiskDecision` (approve/reject + adjusted size)
- **Thresholds**: VaR ≤ 2%, CVaR ≤ 3%, Sharpe > 1.5, drawdown < 15%, correlation < 0.85
- **Circuit breakers**: 3% daily loss, 5% weekly reduction, correlation spike, 15% drawdown, VIX 2x, 10% concentration

### backtest-engineer
Orthogonal research lane — NEVER in live pipeline.

- **Input**: Historical data + strategy config
- **Output**: `SignedBacktestArtifact` → trading-backtests namespace
- **Methods**: Walk-forward (6M train / 1M test), Monte Carlo (1000 runs)
- **Quality gates**: 6 gates, all must pass

---

## Skills

| Skill | Command | Description |
|---|---|---|
| `trader-backtest` | `trader backtest` | Walk-forward backtesting with quality gates |
| `trader-signal` | `trader signal scan` | Z-score signal scanning |
| `trader-portfolio` | `trader portfolio optimize` | Mean-variance optimization |
| `trader-regime` | `trader regime` | Market regime classification |
| `trader-train` | `trader train lstm\|transformer` | Neural model training |
| `trader-risk` | `trader risk assess` | VaR/CVaR risk assessment |
| `trader-cloud-backtest` | `trader backtest --cloud` | Cloud-dispatched backtests |

---

## Quick Start

### Python (direct)

```bash
# Install dependencies
pip install -r requirements.txt

# Run signal scan
python -c "
from scripts.agents.orchestrator import TradingOrchestrator
import yaml, pandas as pd
config = yaml.safe_load(open('config.yaml'))
orch = TradingOrchestrator(config)
# Load your OHLCV data...
df = pd.read_csv('data/sample_ohlcv.csv')
result = orch.run_pipeline(df, ['BTC/USDT'])
print(result)
"

# Run backtest
python scripts/backtest.py --symbol BTC/USDT --strategy momentum

# Run live bot (dry run)
streamlit run scripts/app.py -- --dry-run
```

### ruflo plugin

```bash
# Initialize ruflo
npm run agent:init

# Start plugin
npm run plugin:neural-trader

# Scan signals
npm run trader:signal

# Run backtest
npm run trader:backtest

# Live trading (dry run)
npm run trader:live
```

---

## Configuration

### config.yaml (key sections)

```yaml
data:
  symbols: ['BTC/USDT', 'ETH/USDT']
  timeframe: '5m'
  lookback: 60

trading:
  risk_percentage: 0.01          # Max 1% per trade
  stop_loss_percentage: 0.02
  take_profit_percentage: 0.04
  trailing_stop:
    enabled: true
    percentage: 0.01

models:
  model_type: 'lstm'
  model_selection:
    enabled: true
    volatility_threshold: 0.02
```

### Circuit Breaker Thresholds

All thresholds enforced by the risk-analyst. Hardcoded — not configurable in `config.yaml` to prevent accidental loosening.

| Threshold | Value |
|---|---|
| VaR (95%) | ≤ 2% of portfolio |
| CVaR (95%) | ≤ 3% of portfolio |
| Daily loss halt | ≥ 3% loss |
| Weekly reduction | ≥ 5% loss → 50% size |
| Max drawdown | ≥ 15% → manual reset |
| Correlation | > 0.85 → reject |
| Concentration | > 10% per asset |

---

## Memory Namespaces

| Namespace | Owner | Path |
|---|---|---|
| `trading-analysis` | market-analyst | `data/memory/trading-analysis.json` |
| `trading-signals` | trading-strategist | `data/memory/trading-signals.json` |
| `trading-risk` | risk-analyst | `data/memory/trading-risk.json` |
| `trading-strategies` | trading-strategist | `data/memory/trading-strategies.json` |
| `trading-backtests` | backtest-engineer | `data/memory/trading-backtests.json` |

---

## Safety Philosophy

This plugin implements defense-in-depth for financial safety:

1. **Pipeline invariant**: No broker action without approved `RiskDecision`
2. **Agent isolation**: Each agent has one responsibility, no cross-contamination
3. **Circuit breakers**: Automatic loss protection at multiple levels
4. **Quality gates**: Backtested strategies only — 6 statistical tests required
5. **Dry-run mode**: Full functionality without real money
6. **Audit trail**: Every agent decision logged with timestamps and trace IDs

---

## License

MIT — see repository for full license text.
