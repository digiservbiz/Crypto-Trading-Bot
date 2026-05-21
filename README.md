# Crypto Trading Bot — ruflo Multi-Agent Framework

An AI-powered cryptocurrency trading bot built on the **ruflo** multi-agent orchestration framework. It implements a strict pipeline of specialized agents that collaborate via typed message passing, with hard-coded circuit breakers to prevent catastrophic loss.

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │        LIVE TRADING PIPELINE            │
                    └─────────────────────────────────────────┘

         OHLCV Data
              │
              ▼
    ┌─────────────────┐
    │  market-analyst │  Regime detection (ADX, RSI, MACD, BB, ATR, OBV)
    └────────┬────────┘
             │
             │  RegimeVerdict
             │  {regime_type, confidence, indicator_values}
             ▼
    ┌──────────────────────┐
    │  trading-strategist  │  Strategy selection + Z-score signals
    └──────────┬───────────┘
               │
               │  SignalProposal
               │  {symbol, side, size_pct, confidence, strategy_name}
               ▼
    ┌──────────────────┐
    │   risk-analyst   │  VaR/CVaR/Kelly/circuit-breakers (BLOCKING GATE)
    └────────┬─────────┘
             │
             │  RiskDecision {decision: approve | reject}
             ▼
    ┌─────────────────────────────┐
    │  broker (ONLY if approved)  │  ccxt exchange execution
    └─────────────────────────────┘

                    ┌─────────────────────────────────────────┐
                    │        RESEARCH LANE (Isolated)         │
                    └─────────────────────────────────────────┘

    ┌────────────────────┐       ┌───────────────────────────┐
    │  backtest-engineer │ ───►  │  trading-backtests         │
    │  (walk-forward +   │       │  namespace                 │
    │   Monte Carlo)     │       │  data/memory/              │
    └────────────────────┘       └───────────────────────────┘
```

---

## ruflo Integration

This bot is built on the [ruflo](https://github.com/ruvnet/ruflo) multi-agent orchestration framework. It provides:

- **Agent definitions** in `.agents/` — detailed persona files for each agent
- **Plugin system** via `plugins/ruflo-neural-trader/` and `plugins/ruflo-market-data/`
- **CLI commands** via `trader` and `market` command groups
- **Memory namespaces** for persistent inter-agent state
- **Typed message protocol** with `RegimeVerdict`, `SignalProposal`, `RiskDecision`, `SignedBacktestArtifact`

---

## Quick Start

### Python Path

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure your exchange (edit config.yaml)
cp config.yaml config.local.yaml
# Edit config.local.yaml with your API keys

# 3. Run signal scan (no trading)
python -m scripts.agents.orchestrator

# 4. Run backtest
python scripts/backtest.py --symbol BTC/USDT --strategy momentum

# 5. Run bot (dry run — recommended first)
streamlit run scripts/app.py

# 6. Train models
python scripts/training/train_sequential.py
```

### ruflo Plugin Path

```bash
# 1. Initialize ruflo
npm run agent:init

# 2. Start neural-trader plugin
npm run plugin:neural-trader

# 3. Scan for signals
npm run trader:signal

# 4. Run backtest
npm run trader:backtest

# 5. Assess risk
npm run trader:risk

# 6. Start live trading (dry run)
npm run trader:live
```

### Docker

```bash
# Build and run
docker build -t crypto-trading-bot .
docker run -p 8501:8501 -p 8000:8000 \
  -e EXCHANGE_API_KEY=your_key \
  -e EXCHANGE_SECRET_KEY=your_secret \
  crypto-trading-bot
```

---

## Agents

### market-analyst
- **Role**: Entry point of live pipeline — regime detection
- **Input**: Raw OHLCV data
- **Output**: `RegimeVerdict` with 6 possible regimes
- **Regimes**: `bull-trending`, `bear-trending`, `ranging`, `high-volatility`, `low-volatility`, `transitioning`
- **Indicators**: RSI(14), MACD(12,26,9), Bollinger Bands(20,2), ADX(14), ATR(14), OBV

### trading-strategist
- **Role**: Strategy orchestration — signal generation
- **Input**: `RegimeVerdict`
- **Output**: `SignalProposal` or None
- **Strategies**: `momentum`, `mean-reversion`, `pairs`, `adaptive`
- **Anomaly detection**: Z-score (`spike`, `drift`, `flatline`, `oscillation`, `pattern-break`, `cluster-outlier`)
- **CRITICAL**: Never invokes broker — always routes through risk-analyst

### risk-analyst
- **Role**: Blocking gate — no execution without approval
- **Input**: `SignalProposal`
- **Output**: `RiskDecision` (approve/reject + adjusted size)
- **Thresholds**: VaR ≤ 2%, CVaR ≤ 3%, Sharpe > 1.5, drawdown < 15%, correlation < 0.85

### backtest-engineer
- **Role**: Research lane — strategy validation (isolated from live pipeline)
- **Input**: Historical data + strategy config
- **Output**: `SignedBacktestArtifact` → `trading-backtests` namespace
- **Methods**: Walk-forward (6M train / 1M test), Monte Carlo (1000 runs), 6 quality gates

---

## Circuit Breakers

Enforced by the risk-analyst. Cannot be disabled without code changes.

| Breaker | Trigger | Action |
|---|---|---|
| Daily Loss Halt | Loss ≥ 3% of start-of-day balance | Halt all trading until midnight UTC |
| Weekly Reduction | Loss ≥ 5% of start-of-week balance | 50% size reduction for 7 days |
| Correlation Spike | Pairwise correlation > 0.85 | Reject trades increasing correlation |
| Max Drawdown Halt | Portfolio drawdown ≥ 15% | Full halt — **MANUAL RESET REQUIRED** |
| VIX Spike | ATR ratio > 2× 90-day average | 75% size reduction |
| Concentration Limit | Single asset > 10% of portfolio | Reject further increases |

---

## Configuration

Edit `config.yaml`:

```yaml
data:
  symbols: ['BTC/USDT', 'ETH/USDT']
  timeframe: '5m'
  lookback: 60

exchange:
  name: 'binance'
  api_key: 'YOUR_API_KEY'
  secret_key: 'YOUR_SECRET_KEY'

trading:
  risk_percentage: 0.01           # Max 1% per trade
  stop_loss_percentage: 0.02
  take_profit_percentage: 0.04
  trailing_stop:
    enabled: true
    percentage: 0.01

models:
  model_type: 'lstm'              # lstm or transformer
  model_selection:
    enabled: true
    volatility_threshold: 0.02
```

---

## File Organization

```
/
├── CLAUDE.md                      # ruflo agent configuration
├── config.yaml                    # Bot configuration
├── requirements.txt               # Python dependencies
├── package.json                   # ruflo/Node integration
│
├── .agents/                       # Agent definition files
│   ├── market-analyst.md
│   ├── trading-strategist.md
│   ├── risk-analyst.md
│   └── backtest-engineer.md
│
├── scripts/
│   ├── bot.py                     # Main trading loop (uses orchestrator)
│   ├── backtest.py                # Backtesting CLI entry point
│   ├── exchange.py                # ccxt exchange wrapper
│   ├── app.py                     # Streamlit UI
│   ├── inference/
│   │   └── ai_engine.py           # LSTM/Transformer inference
│   ├── training/
│   │   ├── train_sequential.py    # PyTorch Lightning training
│   │   ├── train_garch.py
│   │   └── train_anomaly.py
│   └── agents/                    # Python agent implementations
│       ├── __init__.py
│       ├── base_agent.py          # Shared dataclasses and BaseAgent
│       ├── market_analyst.py      # MarketAnalystAgent
│       ├── trading_strategist.py  # TradingStrategistAgent
│       ├── risk_analyst.py        # RiskAnalystAgent
│       ├── backtest_engineer.py   # BacktestEngineerAgent
│       └── orchestrator.py        # TradingOrchestrator
│
├── plugins/
│   ├── ruflo-neural-trader/       # Primary trading plugin
│   │   ├── .claude-plugin/plugin.json
│   │   ├── agents/                # Agent definitions
│   │   ├── commands/trader.md     # CLI command reference
│   │   ├── skills/                # 7 trading skills
│   │   └── README.md
│   └── ruflo-market-data/         # Market data plugin
│       ├── .claude-plugin/plugin.json
│       ├── agents/data-engineer.md
│       ├── commands/market.md
│       ├── skills/
│       └── README.md
│
├── data/
│   └── memory/                    # Agent memory namespaces (JSON)
│
└── models/                        # Saved model checkpoints
```

---

## Development Commands

```bash
# Run the live trading bot (dry-run)
streamlit run scripts/app.py -- --dry-run

# Run a backtest
python scripts/backtest.py --config config.yaml --symbol BTC/USDT

# Run backtest with specific strategy
python scripts/backtest.py --symbol BTC/USDT --strategy mean-reversion

# List previous backtest results
python scripts/backtest.py --list

# Train a new LSTM model
python scripts/training/train_sequential.py --config config.yaml

# Run tests
python -m pytest tests/ -v

# Start with ruflo plugin
npm run plugin:neural-trader
npm run trader:live

# Scan for signals (no trades)
npm run trader:signal

# Run backtesting
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

---

## Monitoring

- **Streamlit UI**: `http://localhost:8501`
- **Prometheus metrics**: `http://localhost:8000`
- **Grafana dashboards**: `http://localhost:3000` (via docker-compose)

Key metrics exported:
- `bot_balance` — Current USDT balance
- `bot_total_profit_loss` — Total P&L since start
- `bot_open_positions` — Number of open positions
- `bot_market_regime` — Current regime code per symbol
- `bot_pipeline_cycles_total` — Total pipeline runs
- `bot_pipeline_approvals_total` — Approved signals
- `bot_pipeline_rejections_total` — Rejected signals
- `bot_circuit_breaker_total` — Circuit breaker events by type

---

## Safety

This bot implements defense-in-depth:

1. **Pipeline invariant**: Broker only invoked with approved `RiskDecision`
2. **Agent isolation**: Each agent has one responsibility, typed messages only
3. **Circuit breakers**: Automatic halts at 6 trigger conditions
4. **Quality gates**: 6 statistical tests required before strategy goes live
5. **Dry-run mode**: All features work without real money
6. **Audit trail**: Every agent decision logged with timestamp and trace ID
7. **Memory persistence**: Circuit breaker state survives restarts

**Always use `--dry-run` until you have verified the bot works as expected.**
