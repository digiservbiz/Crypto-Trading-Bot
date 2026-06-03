# Crypto Trading Bot — ruflo Multi-Agent Configuration

This project is a production-grade AI-powered crypto trading bot built on the **ruflo** multi-agent orchestration framework. It implements a strict pipeline of specialized agents that collaborate via typed message passing, with hard-coded circuit breakers to prevent catastrophic loss.

---

## Behavioral Rules

### Non-Negotiable Pipeline Constraint
**The broker may NEVER be invoked without a prior `RiskDecision` with `decision == "approve"`.**

Any code path that reaches exchange order execution MUST demonstrate a `RiskDecision` object with:
- `decision == "approve"`
- `circuit_breaker_triggered == False`
- A valid `signal_id` matching the active `SignalProposal`

Violating this rule is a critical defect. If you are uncertain whether a `RiskDecision` exists, do NOT execute the trade.

### Concurrent Operations
- Market data ingestion for multiple symbols runs concurrently (asyncio / ThreadPoolExecutor).
- Agent pipeline stages are sequential per symbol: analyst → strategist → risk → broker.
- Backtest runs are always isolated from the live pipeline and never share mutable state.
- Memory namespaces are per-agent and must not be cross-written.

### Agent Communication Protocol
Agents communicate exclusively via typed dataclass messages. No agent inspects another agent's internal state. All messages include a timestamp and a short UUID for traceability.

---

## Agent Pipeline

### Live Trading Pipeline (Sequential)

```
market-analyst
     │
     │  RegimeVerdict
     ▼
trading-strategist
     │
     │  SignalProposal
     ▼
risk-analyst
     │
     │  RiskDecision (approve/reject)
     ▼
broker (ONLY if approved)
```

### Research Lane (Orthogonal — Never in Live Pipeline)

```
backtest-engineer  ──►  SignedBacktestArtifact  ──►  trading-backtests namespace
```

The backtest-engineer **does not** send messages to any live-pipeline agent and **does not** receive messages from them. It is invoked directly by the CLI or a scheduled research task.

---

## Agent Descriptions

### `market-analyst`
- **Entry point** of the live pipeline.
- Classifies current market regime from OHLCV + indicator data.
- **Regimes**: `bull-trending`, `bear-trending`, `ranging`, `high-volatility`, `low-volatility`, `transitioning`
- **Indicators used**: RSI, MACD, Bollinger Bands, ADX, ATR, OBV
- **Output**: `RegimeVerdict` → sent to `trading-strategist`
- **Never** communicates with broker, risk-analyst, or backtest-engineer directly.

### `trading-strategist`
- **Middle stage** — receives `RegimeVerdict`, generates trade signals.
- Selects strategy family: `momentum`, `mean-reversion`, `pairs`, `adaptive`
- Uses Z-score anomaly detection: `spike`, `drift`, `flatline`, `oscillation`, `pattern-break`, `cluster-outlier`
- **CRITICAL**: REFUSES to invoke broker under any circumstance. Always routes through risk-analyst.
- **Output**: `SignalProposal` → sent to `risk-analyst`
- Returns `None` signal if confidence is below threshold or regime is `transitioning`.

### `risk-analyst`
- **Blocking gate** — every `SignalProposal` must pass through here before execution.
- Computes VaR (95%), CVaR (95%), Kelly-criterion position size, Sharpe ratio.
- Enforces all circuit breakers (see Circuit Breakers section).
- **Output**: `RiskDecision` (approve or reject with reason)
- A rejected `RiskDecision` STOPS the pipeline. No broker action taken.

### `backtest-engineer`
- **Orthogonal research lane** — not part of the live pipeline.
- Runs walk-forward validation (6-month train / 1-month test windows).
- Runs Monte Carlo simulation (1000 simulations).
- Enforces 6 quality gates before signing artifact.
- **Output**: `SignedBacktestArtifact` → stored to `trading-backtests` memory namespace.
- Produces **no** `SendMessage` outputs to other agents.

---

## Memory Namespace Conventions

| Namespace | Owner | Contents |
|---|---|---|
| `trading-strategies` | trading-strategist | Active strategy configs, parameter sets |
| `trading-backtests` | backtest-engineer | `SignedBacktestArtifact` records, walk-forward scores |
| `trading-risk` | risk-analyst | Circuit breaker state, daily/weekly P&L accumulators |
| `trading-analysis` | market-analyst | Recent `RegimeVerdict` history, indicator snapshots |
| `trading-signals` | trading-strategist | `SignalProposal` log, anomaly score history |

Rules:
- Each agent reads/writes ONLY its own namespace.
- No cross-namespace writes.
- Namespaces persist across bot restarts via JSON files in `data/memory/`.

---

## 3-Tier Model Routing

### Tier 1 — Simple Transforms (fastest, cheapest)
- OHLCV normalization and feature engineering
- Moving average calculations
- Z-score computation for anomaly detection
- Raw indicator calculation (RSI, MACD, BB, ATR, ADX, OBV)
- Used by: market-analyst (indicator phase), trading-strategist (signal prep)

### Tier 2 — Regime Classification & Signal Generation (moderate)
- Market regime classification (ADX + RSI + ATR + volatility fusion)
- Signal confidence scoring
- Strategy family selection
- VaR/CVaR estimation via historical simulation
- Sharpe/Sortino calculation
- Used by: market-analyst (regime output), trading-strategist (signal scoring), risk-analyst (assessment)

### Tier 3 — Neural Training & Portfolio Optimization (heavy, async)
- LSTM model training (PyTorch Lightning)
- Transformer model training
- Walk-forward backtesting
- Monte Carlo simulation (1000 runs)
- Mean-variance portfolio optimization
- Used by: backtest-engineer, training scripts

Tier 3 tasks are NEVER run synchronously in the live trading loop.

---

## Circuit Breakers

The risk-analyst enforces all circuit breakers. When triggered, the circuit breaker halts ALL trading for the defined cooling period.

| Breaker | Trigger Condition | Action |
|---|---|---|
| Daily Loss Halt | Realized P&L loss ≥ 3% of start-of-day balance | Halt all trading for remainder of day |
| Weekly Reduction | Realized P&L loss ≥ 5% of start-of-week balance | Reduce all position sizes by 50% for 1 week |
| Correlation Spike | Portfolio correlation > 0.85 between any 2 holdings | Reject new positions that increase correlation |
| Drawdown Limit | Portfolio drawdown from peak ≥ 15% | Full halt, requires manual reset |
| VIX Spike | Implied volatility > 2x 90-day historical average | Reduce position sizes by 75% |
| Concentration Limit | Single asset > 10% of total portfolio value | Reject any further increase in that asset |

Circuit breaker state is stored in the `trading-risk` memory namespace and survives restarts.

### Risk Thresholds (per trade)
- VaR (95% confidence) ≤ 2% of portfolio
- CVaR (95%) ≤ 3% of portfolio
- Portfolio max drawdown ≤ 15%
- Sharpe ratio > 1.5 (rolling 30-day)
- Pairwise correlation ≤ 0.85

---

## Communication Protocol — Message Types

### `RegimeVerdict`
```python
@dataclass
class RegimeVerdict:
    regime_type: str        # bull-trending | bear-trending | ranging | high-volatility | low-volatility | transitioning
    confidence: float       # 0.0 – 1.0
    indicator_values: Dict[str, float]  # adx, rsi, atr, macd, bb_width, obv_trend
    symbols: List[str]
    timestamp: float
    verdict_id: str         # 8-char UUID prefix
```

### `SignalProposal`
```python
@dataclass
class SignalProposal:
    symbol: str
    side: str               # buy | sell
    size_pct: float         # % of portfolio to allocate
    confidence: float       # 0.0 – 1.0
    strategy_name: str      # momentum | mean-reversion | pairs | adaptive
    anomaly_type: str       # spike | drift | flatline | oscillation | pattern-break | cluster-outlier
    anomaly_score: float    # Z-score magnitude
    regime_verdict_id: str  # links back to RegimeVerdict
    timestamp: float
    signal_id: str          # 8-char UUID prefix
```

### `RiskDecision`
```python
@dataclass
class RiskDecision:
    signal_id: str          # references SignalProposal.signal_id
    decision: str           # approve | reject
    adjusted_size_pct: float
    var_95: float
    cvar_95: float
    sharpe_ratio: float
    circuit_breaker_triggered: bool
    circuit_breaker_reason: Optional[str]
    timestamp: float
```

### `SignedBacktestArtifact`
```python
@dataclass
class SignedBacktestArtifact:
    strategy_id: str
    symbol: str
    period: str
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    walk_forward_scores: List[float]
    monte_carlo_p_value: float
    passed_quality_gates: bool
    timestamp: float
    artifact_id: str        # 8-char UUID prefix
```

---

## Quality Gates (BacktestEngineer)

An artifact is only signed and stored if ALL gates pass:

1. **Minimum trades**: ≥ 30 trades in test period
2. **Win-rate consistency**: variance across walk-forward windows < 15%
3. **Monte Carlo significance**: p-value < 0.05 (returns are non-random)
4. **Max drawdown**: < 15% across all windows
5. **Profit factor**: > 1.5
6. **Sharpe ratio**: > 1.0 (annualized)

---

## File Organization

```
/
├── CLAUDE.md                          # This file — ruflo configuration
├── config.yaml                        # Bot configuration
├── config.live.yaml                   # Live trading overrides
├── requirements.txt                   # Python dependencies
├── package.json                       # ruflo/Node integration
│
├── .agents/                           # Agent definition files
│   ├── market-analyst.md
│   ├── trading-strategist.md
│   ├── risk-analyst.md
│   └── backtest-engineer.md
│
├── scripts/
│   ├── bot.py                         # Main trading loop (uses orchestrator)
│   ├── backtest.py                    # Backtesting entry point
│   ├── exchange.py                    # ccxt exchange wrapper
│   ├── inference/
│   │   └── ai_engine.py               # LSTM/Transformer inference
│   ├── training/
│   │   ├── train_sequential.py        # PyTorch Lightning training
│   │   ├── train_garch.py
│   │   └── train_anomaly.py
│   └── agents/                        # Python agent implementations
│       ├── __init__.py
│       ├── base_agent.py              # Shared dataclasses and BaseAgent
│       ├── market_analyst.py          # MarketAnalystAgent
│       ├── trading_strategist.py      # TradingStrategistAgent
│       ├── risk_analyst.py            # RiskAnalystAgent
│       ├── backtest_engineer.py       # BacktestEngineerAgent
│       └── orchestrator.py            # TradingOrchestrator (pipeline coordinator)
│
├── plugins/
│   ├── ruflo-neural-trader/           # Primary trading plugin
│   │   ├── .claude-plugin/plugin.json
│   │   ├── agents/                    # Agent definitions (copies from .agents/)
│   │   ├── commands/trader.md         # CLI command reference
│   │   ├── skills/                    # 7 trading skills
│   │   └── README.md
│   └── ruflo-market-data/             # Market data pipeline plugin
│       ├── .claude-plugin/plugin.json
│       ├── agents/data-engineer.md
│       ├── commands/market.md
│       ├── skills/                    # 2 data skills
│       └── README.md
│
├── data/
│   ├── memory/                        # Agent memory namespaces (JSON)
│   │   ├── trading-analysis.json
│   │   ├── trading-signals.json
│   │   ├── trading-risk.json
│   │   ├── trading-strategies.json
│   │   └── trading-backtests.json
│   └── sample_ohlcv.csv
│
└── models/                            # Saved model checkpoints
```

---

## Development Commands

```bash
# Run the live trading bot (dry-run)
streamlit run scripts/app.py -- --dry-run

# Run a backtest on historical data
python scripts/backtest.py --config config.yaml --symbol BTC/USDT

# Train a new LSTM model
python scripts/training/train_sequential.py --config config.yaml --symbol BTC/USDT

# Run the full agent pipeline test
python -m pytest tests/ -v

# Start with ruflo plugin
npm run plugin:neural-trader
npm run trader:live

# Scan for signals without trading
npm run trader:signal

# Run backtesting via CLI
npm run trader:backtest
```

---

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `EXCHANGE_API_KEY` | Exchange API key | Live trading only |
| `EXCHANGE_SECRET_KEY` | Exchange secret key | Live trading only |
| `TELEGRAM_TOKEN` | Telegram bot token for alerts | Optional |
| `TELEGRAM_CHAT_ID` | Telegram chat ID for alerts | Optional |
| `DRY_RUN` | Set to `true` to disable real orders | Recommended for testing |
| `CIRCUIT_BREAKER_OVERRIDE` | Emergency manual override (admin only) | Never in production |

---

## Safety Philosophy

This bot implements defense-in-depth:
1. **Agent isolation**: each agent has a single responsibility and cannot invoke broker actions.
2. **Typed messages**: all inter-agent communication uses strongly-typed dataclasses.
3. **Circuit breakers**: automatic halts prevent compounding losses.
4. **Quality gates**: backtests must pass statistical significance tests before any strategy goes live.
5. **Dry-run mode**: all features work without real money, always test there first.
6. **Logging**: every agent decision is logged with timestamps and IDs for audit trails.
