# trader — CLI Command Reference

The `trader` command is the primary interface for the ruflo-neural-trader plugin. It provides access to all trading operations: strategy management, signal scanning, backtesting, model training, risk assessment, portfolio optimization, and live trading.

---

## Synopsis

```
trader <command> [subcommand] [options]
```

---

## Commands

### `trader strategy create <name>`

Create a new named trading strategy configuration.

```bash
trader strategy create my-btc-momentum --type momentum
trader strategy create eth-mean-rev --type mean-reversion
trader strategy create btc-eth-pairs --type pairs
trader strategy create adaptive-v1 --type adaptive
```

**Options:**
| Flag | Description | Default |
|---|---|---|
| `--type` | Strategy family | Required: `momentum\|mean-reversion\|pairs\|adaptive` |
| `--symbol` | Target trading pair | `BTC/USDT` |
| `--lookback` | Rolling window period | `20` |
| `--z-threshold` | Z-score entry threshold | `1.5` |
| `--max-size` | Maximum position size (fraction) | `0.05` |
| `--save` | Save to trading-strategies namespace | `true` |

**Output:** Creates a strategy config entry in `data/memory/trading-strategies.json`

---

### `trader backtest <strategy>`

Run a full backtest on a named strategy using the BacktestEngineerAgent.

Includes walk-forward validation (6-month train / 1-month test) and Monte Carlo simulation (1000 runs).

```bash
trader backtest my-btc-momentum --symbol BTC/USDT --period 2023-01-01/2024-01-01
trader backtest eth-mean-rev --symbol ETH/USDT --period last-12m
trader backtest adaptive-v1 --symbol BTC/USDT --output results.json
```

**Options:**
| Flag | Description | Default |
|---|---|---|
| `--symbol` | Trading pair | From strategy config |
| `--period` | Date range or shorthand (`last-6m`, `last-12m`, `last-24m`) | `last-12m` |
| `--train-months` | Walk-forward training window | `6` |
| `--test-months` | Walk-forward test window | `1` |
| `--mc-runs` | Monte Carlo simulation count | `1000` |
| `--output` | Save artifact to JSON file | None |
| `--verbose` | Show per-window metrics | `false` |

**Quality Gates (all must pass):**
1. Minimum 30 trades in test period
2. Win-rate variance < 15% across windows
3. Monte Carlo p-value < 0.05
4. Max drawdown < 15%
5. Profit factor > 1.5
6. Sharpe ratio > 1.0

**Example output:**
```
=== BACKTEST RESULTS ===
Strategy: my-btc-momentum
Symbol:   BTC/USDT
Period:   2023-01-01/2024-01-01

Total Return:  +31.2%
Sharpe Ratio:   1.94
Sortino Ratio:  2.67
Max Drawdown:  -8.7%
Win Rate:      58.7%
Profit Factor:  1.83
Num Trades:    147

Quality Gates: ALL PASSED ✓
```

---

### `trader train <model-type>`

Train a new LSTM or Transformer neural model for price direction prediction.

```bash
trader train lstm --symbol BTC/USDT
trader train transformer --symbol ETH/USDT --epochs 150
trader train lstm --symbol BTC/USDT --volatility-type high_volatility
```

**Options:**
| Flag | Description | Default |
|---|---|---|
| `--symbol` | Training symbol | `BTC/USDT` |
| `--epochs` | Maximum training epochs | `100` |
| `--batch-size` | Training batch size | `2048` |
| `--volatility-type` | `high_volatility\|low_volatility\|model` | `model` |
| `--lookback` | Sequence length for model input | `60` |
| `--hidden-size` | LSTM hidden dimensions | `64` |
| `--save-dir` | Model checkpoint directory | `models/` |

**Runs:** `scripts/training/train_sequential.py` with PyTorch Lightning

---

### `trader signal scan`

Scan configured symbols for trade signals without executing trades.
Runs the market-analyst → trading-strategist pipeline.

```bash
trader signal scan
trader signal scan --strategy momentum --symbols BTC/USDT,ETH/USDT
trader signal scan --min-confidence 0.65 --regime bull-trending
```

**Options:**
| Flag | Description | Default |
|---|---|---|
| `--strategy` | Filter by strategy name | All |
| `--symbols` | Comma-separated symbol list | From config |
| `--min-confidence` | Minimum signal confidence to display | `0.50` |
| `--regime` | Filter by regime type | All |
| `--output-json` | Save signals to JSON | None |

**Output format:**
```json
{
  "symbol": "BTC/USDT",
  "side": "buy",
  "confidence": 0.72,
  "strategy": "momentum",
  "regime": "bull-trending",
  "anomaly_type": "pattern-break",
  "anomaly_score": 2.31,
  "size_pct": 0.045
}
```

---

### `trader regime`

Analyze and display the current market regime for a symbol.

```bash
trader regime --symbol BTC/USDT
trader regime --symbol ETH/USDT --verbose
trader regime --symbols BTC/USDT,ETH/USDT
```

**Options:**
| Flag | Description | Default |
|---|---|---|
| `--symbol` | Target symbol | `BTC/USDT` |
| `--symbols` | Multiple symbols | From config |
| `--verbose` | Show all indicator values | `false` |
| `--history` | Show last N regime verdicts | `0` |

**Output:**
```
Symbol: BTC/USDT
Regime: bull-trending (confidence: 0.82)
Indicators:
  ADX: 31.4  RSI: 61.2  MACD hist: +0.0043
  BB %B: 0.71  ATR ratio: 1.12  OBV trend: +0.65
```

---

### `trader risk assess`

Run a risk assessment for a potential trade or current portfolio.

```bash
trader risk assess
trader risk assess --symbol BTC/USDT --side buy --size 0.05
trader risk assess --portfolio
```

**Options:**
| Flag | Description | Default |
|---|---|---|
| `--symbol` | Symbol to assess | `BTC/USDT` |
| `--side` | Trade direction | `buy` |
| `--size` | Proposed position size | `0.05` |
| `--portfolio` | Show full portfolio risk metrics | `false` |

**Output:**
```
Risk Assessment — BTC/USDT BUY
  VaR (95%):     1.48%  [PASS — limit: 2.0%]
  CVaR (95%):    2.21%  [PASS — limit: 3.0%]
  Sharpe:        1.87   [PASS — min: 1.5]
  Kelly size:    3.1%   [adjusted from proposed 5.0%]
  Circuit Breakers: NONE TRIGGERED
  Decision: APPROVE
```

---

### `trader portfolio optimize`

Run mean-variance portfolio optimization across configured symbols.

```bash
trader portfolio optimize
trader portfolio optimize --risk-target 0.15
trader portfolio optimize --symbols BTC/USDT,ETH/USDT,SOL/USDT --method sharpe
```

**Options:**
| Flag | Description | Default |
|---|---|---|
| `--symbols` | Symbols to optimize | From config |
| `--risk-target` | Target portfolio volatility | `0.15` |
| `--method` | Optimization objective: `sharpe\|min-variance\|max-return` | `sharpe` |
| `--lookback-days` | Historical window for correlation | `90` |
| `--output` | Save allocation to JSON | None |

**Output:**
```
Optimal Portfolio Allocation:
  BTC/USDT:  42.3%
  ETH/USDT:  35.7%
  Uncorrelated cash: 22.0%
  
Expected Sharpe: 2.14
Expected volatility: 14.8%
Max drawdown estimate: 11.2%
```

---

### `trader live`

Start the live trading bot with the full multi-agent pipeline.

```bash
trader live --symbols BTC/USDT,ETH/USDT
trader live --symbols BTC/USDT --dry-run
trader live --config config.live.yaml --dry-run
```

**Options:**
| Flag | Description | Default |
|---|---|---|
| `--symbols` | Trading symbols | From config |
| `--dry-run` | Simulate trades, no real orders | **RECOMMENDED** |
| `--config` | Config file path | `config.yaml` |
| `--no-ui` | Run without Streamlit UI | `false` |
| `--log-level` | Logging verbosity | `INFO` |

**Safety note:** Always test with `--dry-run` before live trading.

Pipeline order (enforced, non-negotiable):
```
market-analyst → trading-strategist → risk-analyst → broker
```
No trade executes without an approved RiskDecision.

---

### `trader history`

Display recent trading history and pipeline statistics.

```bash
trader history
trader history --limit 20
trader history --symbol BTC/USDT --type signals
```

**Options:**
| Flag | Description | Default |
|---|---|---|
| `--limit` | Number of records to show | `10` |
| `--symbol` | Filter by symbol | All |
| `--type` | Filter type: `signals\|decisions\|trades\|backtests` | All |
| `--format` | Output format: `table\|json\|csv` | `table` |

---

## Global Options

| Flag | Description |
|---|---|
| `--config <path>` | Config file path (default: `config.yaml`) |
| `--verbose`, `-v` | Enable verbose logging |
| `--json` | Output in JSON format |
| `--help`, `-h` | Show help for command |

---

## Environment Variables

| Variable | Description |
|---|---|
| `EXCHANGE_API_KEY` | Exchange API key for live trading |
| `EXCHANGE_SECRET_KEY` | Exchange secret key |
| `TELEGRAM_TOKEN` | Telegram bot token for alerts |
| `TELEGRAM_CHAT_ID` | Telegram chat ID |
| `DRY_RUN` | Set `true` to disable real orders |
| `LOG_LEVEL` | Logging level (`DEBUG`, `INFO`, `WARNING`) |

---

## Exit Codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Backtest quality gates failed |
| `2` | Configuration error |
| `3` | Data loading error |
| `4` | Exchange connection error |
| `5` | Circuit breaker triggered |
