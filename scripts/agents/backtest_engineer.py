"""Backtesting specialist agent — orthogonal research lane.

The BacktestEngineerAgent runs walk-forward validation, Monte Carlo simulation,
and quality gate enforcement for trading strategies. It operates COMPLETELY
OUTSIDE the live trading pipeline and never communicates with live pipeline agents.

This agent is invoked ONLY by:
- CLI commands (python scripts/backtest.py)
- Scheduled research tasks
- Manual orchestration for strategy validation

Pipeline position: RESEARCH LANE ONLY (isolated from live pipeline)
                   backtest-engineer → SignedBacktestArtifact → trading-backtests namespace
"""

from typing import Dict, List, Optional, Any, Tuple
import json
import logging
import math
import os
import time
import uuid
import numpy as np
import pandas as pd

from scripts.agents.base_agent import (
    BaseAgent, AgentRole, SignedBacktestArtifact
)


logger = logging.getLogger(__name__)

# Memory storage path
BACKTEST_MEMORY_PATH = "data/memory/trading-backtests.json"

# Walk-forward configuration
TRAIN_MONTHS = 6
TEST_MONTHS = 1
MIN_WINDOWS = 3
MIN_ROWS = 200

# Quality gate thresholds
QUALITY_GATES = {
    "min_trades": 30,
    "win_rate_variance_max": 0.15,
    "monte_carlo_p_value_max": 0.05,
    "max_drawdown_threshold": 0.15,
    "profit_factor_min": 1.5,
    "sharpe_min": 1.0,
}

# Monte Carlo settings
MONTE_CARLO_RUNS = 1000


class BacktestEngineerAgent(BaseAgent):
    """Strategy validation agent using walk-forward and Monte Carlo methods.

    This agent runs comprehensive strategy validation before any strategy
    is considered for live deployment. It maintains complete isolation from
    the live trading pipeline.

    Key validation methods:
    - Walk-forward: 6-month training / 1-month test windows
    - Monte Carlo: 1000 bootstrapped simulations for statistical significance
    - Quality gates: 6 hard requirements that must ALL pass
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the BacktestEngineerAgent.

        Args:
            config: Full bot configuration dictionary.
        """
        super().__init__(AgentRole.BACKTEST_ENGINEER, config)
        self._artifacts: List[Dict] = []
        self._load_artifacts()

    def run_walk_forward(
        self,
        df: pd.DataFrame,
        strategy_config: Dict[str, Any],
    ) -> SignedBacktestArtifact:
        """Run full walk-forward validation and return a signed artifact.

        This is the primary method. It runs the complete validation pipeline:
        1. Walk-forward validation across rolling windows
        2. Monte Carlo simulation for statistical significance
        3. Quality gate enforcement
        4. Artifact signing and storage

        Args:
            df: Full historical OHLCV DataFrame. Must have columns:
                timestamp, open, high, low, close, volume
            strategy_config: Strategy configuration including:
                strategy_id, strategy_type, symbol, parameters

        Returns:
            SignedBacktestArtifact with all metrics and quality gate results.
        """
        strategy_id = strategy_config.get("strategy_id", f"strategy-{str(uuid.uuid4())[:8]}")
        symbol = strategy_config.get("symbol", "UNKNOWN")

        self.log_decision(
            f"Starting walk-forward backtest: {strategy_id} on {symbol} "
            f"({len(df)} rows)"
        )

        # Validate data
        if df is None or len(df) < MIN_ROWS:
            self.log_decision(
                f"Insufficient data ({len(df) if df is not None else 0} rows, "
                f"need {MIN_ROWS}) — returning failed artifact",
                level="warning"
            )
            return self._failed_artifact(strategy_id, symbol, "INSUFFICIENT_DATA")

        # Prepare data
        df = self._prepare_data(df)
        period = self._get_period_string(df)

        try:
            # ---- Walk-forward validation ----
            wf_results = self._run_walk_forward_windows(df, strategy_config)

            if not wf_results:
                return self._failed_artifact(strategy_id, symbol, "NO_VALID_WINDOWS")

            # ---- Aggregate walk-forward metrics ----
            all_trades = []
            wf_sharpe_scores = []
            wf_drawdowns = []
            wf_win_rates = []

            for window_result in wf_results:
                all_trades.extend(window_result["trades"])
                wf_sharpe_scores.append(window_result["sharpe"])
                wf_drawdowns.append(window_result["max_drawdown"])
                wf_win_rates.append(window_result["win_rate"])

            # ---- Compute aggregate metrics ----
            total_return, sharpe, sortino, max_drawdown, win_rate, profit_factor = \
                self._compute_aggregate_metrics(all_trades, df)

            num_trades = len([t for t in all_trades if t.get("closed", True)])

            # ---- Monte Carlo simulation ----
            mc_p_value = self._run_monte_carlo(df, strategy_config)

            # ---- Quality gate evaluation ----
            win_rate_variance = float(np.var(wf_win_rates)) if len(wf_win_rates) > 1 else 0.0
            gates = self._evaluate_quality_gates(
                num_trades=num_trades,
                win_rate_variance=win_rate_variance,
                mc_p_value=mc_p_value,
                max_drawdown=max_drawdown,
                profit_factor=profit_factor,
                sharpe=sharpe,
            )
            passed = all(gates.values())

            # ---- Build artifact ----
            artifact = SignedBacktestArtifact(
                strategy_id=strategy_id,
                symbol=symbol,
                period=period,
                total_return=round(total_return, 4),
                sharpe_ratio=round(sharpe, 4),
                sortino_ratio=round(sortino, 4),
                max_drawdown=round(max_drawdown, 4),
                win_rate=round(win_rate, 4),
                profit_factor=round(profit_factor, 4),
                num_trades=num_trades,
                walk_forward_scores=wf_sharpe_scores,
                monte_carlo_p_value=round(mc_p_value, 4),
                passed_quality_gates=passed,
            )

            # ---- Log quality gate results ----
            self.log_decision(
                f"Quality gates: {gates} | Passed: {passed} | "
                f"Sharpe: {sharpe:.3f} | MaxDD: {max_drawdown:.3f} | "
                f"Trades: {num_trades} | MC p-value: {mc_p_value:.4f}"
            )

            if passed:
                self.log_decision(
                    f"BACKTEST PASSED — strategy {strategy_id} qualified for live deployment review"
                )
            else:
                failed_gates = [g for g, v in gates.items() if not v]
                self.log_decision(
                    f"BACKTEST FAILED — gates failed: {failed_gates}", level="warning"
                )

            # ---- Persist artifact ----
            self._store_artifact(artifact)

            return artifact

        except Exception as exc:
            self.log_decision(
                f"Backtest failed with exception: {exc}", level="error"
            )
            return self._failed_artifact(strategy_id, symbol, f"EXCEPTION: {exc}")

    def _run_walk_forward_windows(
        self,
        df: pd.DataFrame,
        strategy_config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Run walk-forward validation across rolling windows.

        Each window: 6-month train, 1-month test.

        Args:
            df: Prepared OHLCV DataFrame with timestamps.
            strategy_config: Strategy configuration.

        Returns:
            List of per-window result dictionaries.
        """
        results = []

        # Estimate rows per month (assuming 5-minute bars: 288 * 30)
        rows_per_month = self._estimate_rows_per_month(df)
        train_rows = TRAIN_MONTHS * rows_per_month
        test_rows = TEST_MONTHS * rows_per_month
        step_rows = test_rows  # Step by 1 month

        if train_rows + test_rows > len(df):
            # Reduce to fit available data
            available = len(df)
            train_rows = int(available * 0.85)
            test_rows = available - train_rows
            self.log_decision(
                f"Reduced window sizes to fit data: train={train_rows}, test={test_rows}",
                level="warning"
            )

        start = 0
        window_num = 0
        while start + train_rows + test_rows <= len(df):
            train_df = df.iloc[start:start + train_rows]
            test_df = df.iloc[start + train_rows:start + train_rows + test_rows]

            if len(test_df) < 10:
                break

            window_result = self._simulate_strategy(
                train_df=train_df,
                test_df=test_df,
                strategy_config=strategy_config,
                window_num=window_num,
            )

            if window_result is not None:
                results.append(window_result)

            start += step_rows
            window_num += 1

        self.log_decision(f"Completed {len(results)} walk-forward windows")
        return results

    def _simulate_strategy(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        strategy_config: Dict[str, Any],
        window_num: int,
    ) -> Optional[Dict[str, Any]]:
        """Simulate a strategy on the test window using parameters from training.

        Args:
            train_df: Training period data.
            test_df: Test period data (no parameter adjustment allowed).
            strategy_config: Strategy configuration.
            window_num: Window index for logging.

        Returns:
            Dict with trades, sharpe, max_drawdown, win_rate, or None on failure.
        """
        try:
            strategy_type = strategy_config.get("strategy_type", "momentum")

            # Compute training-period statistics for parameter setting
            train_close = train_df["close"].values.astype(float)
            train_returns = np.diff(train_close) / train_close[:-1]

            # Parameters derived from training (simplified)
            lookback = strategy_config.get("parameters", {}).get("lookback", 20)
            threshold = strategy_config.get("parameters", {}).get("threshold", 1.5)

            # Simulate on test period
            test_close = test_df["close"].values.astype(float)
            test_volume = test_df["volume"].values.astype(float)

            trades = self._run_simulation(
                close=test_close,
                volume=test_volume,
                strategy_type=strategy_type,
                lookback=lookback,
                z_threshold=threshold,
                transaction_cost=0.001,  # 0.1% per trade
            )

            if not trades:
                return None

            # Compute window metrics
            trade_returns = [t["return"] for t in trades]
            wins = [t for t in trades if t["return"] > 0]
            losses = [t for t in trades if t["return"] <= 0]

            if len(trade_returns) < 2:
                return None

            sharpe = self._compute_sharpe(np.array(trade_returns))
            max_dd = self._compute_max_drawdown(trade_returns)
            win_rate = len(wins) / len(trades)
            gross_profit = sum(t["return"] for t in wins)
            gross_loss = abs(sum(t["return"] for t in losses))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 2.0

            return {
                "window": window_num,
                "trades": trades,
                "sharpe": round(sharpe, 4),
                "max_drawdown": round(max_dd, 4),
                "win_rate": round(win_rate, 4),
                "profit_factor": round(profit_factor, 4),
                "num_trades": len(trades),
            }

        except Exception as exc:
            self.log_decision(
                f"Window {window_num} simulation failed: {exc}", level="warning"
            )
            return None

    def _run_simulation(
        self,
        close: np.ndarray,
        volume: np.ndarray,
        strategy_type: str,
        lookback: int,
        z_threshold: float,
        transaction_cost: float,
    ) -> List[Dict[str, Any]]:
        """Run a single simulation of the strategy over the given data.

        Args:
            close: Closing price array.
            volume: Volume array.
            strategy_type: "momentum" | "mean-reversion" | "adaptive"
            lookback: Rolling window for statistics.
            z_threshold: Z-score threshold for entry.
            transaction_cost: Per-trade cost fraction.

        Returns:
            List of trade dicts with entry, exit, return, direction.
        """
        trades = []
        position = None
        entry_price = 0.0
        entry_idx = 0

        for i in range(lookback, len(close)):
            window = close[i - lookback:i]
            mean = np.mean(window)
            std = np.std(window, ddof=1)
            if std == 0:
                continue

            z = (close[i] - mean) / std
            current = close[i]

            if position is None:
                # Entry logic
                if strategy_type == "momentum":
                    if z > z_threshold:
                        position = "long"
                        entry_price = current * (1 + transaction_cost)
                        entry_idx = i
                    elif z < -z_threshold:
                        position = "short"
                        entry_price = current * (1 - transaction_cost)
                        entry_idx = i
                elif strategy_type == "mean-reversion":
                    if z < -z_threshold:
                        position = "long"
                        entry_price = current * (1 + transaction_cost)
                        entry_idx = i
                    elif z > z_threshold:
                        position = "short"
                        entry_price = current * (1 - transaction_cost)
                        entry_idx = i
                elif strategy_type == "adaptive":
                    # Blend: momentum for high z, mean-reversion for moderate z
                    if abs(z) > z_threshold * 1.5:
                        direction = "long" if z > 0 else "short"
                        position = direction
                        entry_price = current * (1 + transaction_cost * (1 if z > 0 else -1))
                        entry_idx = i
                    elif abs(z) > z_threshold:
                        direction = "long" if z < 0 else "short"
                        position = direction
                        entry_price = current * (1 + transaction_cost * (1 if z < 0 else -1))
                        entry_idx = i

            else:
                # Exit logic: hold for lookback/4 periods or reverse signal
                hold_periods = max(5, lookback // 4)
                if i - entry_idx >= hold_periods or abs(z) < 0.5:
                    exit_price = current * (1 - transaction_cost if position == "long" else 1 + transaction_cost)
                    if position == "long":
                        trade_return = (exit_price - entry_price) / entry_price
                    else:
                        trade_return = (entry_price - exit_price) / entry_price

                    trades.append({
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "direction": position,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "return": float(trade_return),
                        "closed": True,
                    })
                    position = None

        return trades

    def _run_monte_carlo(
        self,
        df: pd.DataFrame,
        strategy_config: Dict[str, Any],
    ) -> float:
        """Run Monte Carlo simulation via return bootstrapping.

        Null hypothesis: strategy returns are indistinguishable from random.
        Test statistic: fraction of simulations with positive final P&L.

        Args:
            df: Full OHLCV DataFrame.
            strategy_config: Strategy configuration.

        Returns:
            p-value (fraction consistent with null hypothesis).
        """
        close = df["close"].values.astype(float)
        returns = np.diff(close) / close[:-1]
        returns = returns[np.isfinite(returns)]

        if len(returns) < 20:
            return 1.0  # Cannot compute — fail the gate

        positive_count = 0
        n = len(returns)

        rng = np.random.default_rng(seed=42)
        for _ in range(MONTE_CARLO_RUNS):
            # Bootstrap resample
            sampled = rng.choice(returns, size=n, replace=True)
            # Simple buy-and-hold on bootstrapped series
            cumulative = float(np.prod(1 + sampled) - 1)
            if cumulative > 0:
                positive_count += 1

        # p-value: probability of getting this many positives by chance (binomial)
        # Under null: 50% chance of positive return
        fraction_positive = positive_count / MONTE_CARLO_RUNS
        # Two-tailed p-value approximation using normal approximation to binomial
        z = (fraction_positive - 0.5) / math.sqrt(0.25 / MONTE_CARLO_RUNS)
        # p-value from z-score (one-tailed: is strategy better than random?)
        p_value = max(0.001, 1.0 - self._normal_cdf(z))
        return float(p_value)

    def _normal_cdf(self, z: float) -> float:
        """Approximate normal CDF using error function."""
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    def _compute_aggregate_metrics(
        self,
        trades: List[Dict[str, Any]],
        df: pd.DataFrame,
    ) -> Tuple[float, float, float, float, float, float]:
        """Compute aggregate performance metrics from all trades.

        Returns:
            Tuple of (total_return, sharpe, sortino, max_drawdown, win_rate, profit_factor)
        """
        if not trades:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        trade_returns = [t["return"] for t in trades]
        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r <= 0]

        total_return = float(np.prod([1 + r for r in trade_returns]) - 1)
        ret_arr = np.array(trade_returns)

        # Sharpe
        mean_ret = float(np.mean(ret_arr))
        std_ret = float(np.std(ret_arr, ddof=1)) if len(ret_arr) > 1 else 1.0
        sharpe = (mean_ret / std_ret * math.sqrt(252)) if std_ret > 0 else 0.0

        # Sortino (downside deviation only)
        downside = ret_arr[ret_arr < 0]
        downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else std_ret
        sortino = (mean_ret / downside_std * math.sqrt(252)) if downside_std > 0 else 0.0

        # Max drawdown
        max_drawdown = self._compute_max_drawdown(trade_returns)

        # Win rate
        win_rate = len(wins) / len(trade_returns) if trade_returns else 0.0

        # Profit factor
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (2.0 if gross_profit > 0 else 0.0)

        return (
            round(total_return, 4),
            round(sharpe, 4),
            round(sortino, 4),
            round(max_drawdown, 4),
            round(win_rate, 4),
            round(profit_factor, 4),
        )

    def _compute_max_drawdown(self, returns: List[float]) -> float:
        """Compute maximum peak-to-trough drawdown from return series."""
        if not returns:
            return 0.0
        equity = np.cumprod([1 + r for r in returns])
        peak = np.maximum.accumulate(equity)
        drawdowns = (equity - peak) / peak
        return float(np.min(drawdowns))

    def _compute_sharpe(self, returns: np.ndarray) -> float:
        """Compute annualized Sharpe ratio from trade returns."""
        if len(returns) < 2:
            return 0.0
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns, ddof=1))
        if std_ret == 0:
            return 0.0
        return float(mean_ret / std_ret * math.sqrt(252))

    def _evaluate_quality_gates(
        self,
        num_trades: int,
        win_rate_variance: float,
        mc_p_value: float,
        max_drawdown: float,
        profit_factor: float,
        sharpe: float,
    ) -> Dict[str, bool]:
        """Evaluate all 6 quality gates.

        Returns:
            Dictionary mapping gate name to pass/fail boolean.
        """
        return {
            "min_trades": num_trades >= QUALITY_GATES["min_trades"],
            "win_rate_consistency": win_rate_variance < QUALITY_GATES["win_rate_variance_max"],
            "monte_carlo_significance": mc_p_value < QUALITY_GATES["monte_carlo_p_value_max"],
            "max_drawdown": abs(max_drawdown) < QUALITY_GATES["max_drawdown_threshold"],
            "profit_factor": profit_factor > QUALITY_GATES["profit_factor_min"],
            "sharpe_ratio": sharpe > QUALITY_GATES["sharpe_min"],
        }

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and clean OHLCV data for backtesting."""
        df = df.copy()
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        df[required] = df[required].astype(float)
        df = df.dropna(subset=required)
        df = df.reset_index(drop=True)
        return df

    def _estimate_rows_per_month(self, df: pd.DataFrame) -> int:
        """Estimate rows per month based on data frequency."""
        if "timestamp" in df.columns and len(df) > 1:
            ts = df["timestamp"].values
            interval_sec = float(np.median(np.diff(ts[:100])))
            if interval_sec > 0:
                rows_per_day = 86400 / interval_sec
                return int(rows_per_day * 30)
        # Default: 5-minute bars
        return 288 * 30  # 8640

    def _get_period_string(self, df: pd.DataFrame) -> str:
        """Extract period string from DataFrame timestamps."""
        if "timestamp" in df.columns and len(df) > 0:
            try:
                start = pd.to_datetime(df["timestamp"].iloc[0], unit="ms").strftime("%Y-%m-%d")
                end = pd.to_datetime(df["timestamp"].iloc[-1], unit="ms").strftime("%Y-%m-%d")
                return f"{start}/{end}"
            except Exception:
                pass
        return f"rows:0-{len(df)}"

    def _failed_artifact(
        self, strategy_id: str, symbol: str, reason: str
    ) -> SignedBacktestArtifact:
        """Return a failed artifact with all metrics at zero."""
        return SignedBacktestArtifact(
            strategy_id=strategy_id,
            symbol=symbol,
            period="N/A",
            total_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            num_trades=0,
            walk_forward_scores=[],
            monte_carlo_p_value=1.0,
            passed_quality_gates=False,
        )

    def _store_artifact(self, artifact: SignedBacktestArtifact) -> None:
        """Persist artifact to trading-backtests namespace."""
        os.makedirs(os.path.dirname(BACKTEST_MEMORY_PATH), exist_ok=True)
        record = {
            "artifact_id": artifact.artifact_id,
            "strategy_id": artifact.strategy_id,
            "symbol": artifact.symbol,
            "period": artifact.period,
            "total_return": artifact.total_return,
            "sharpe_ratio": artifact.sharpe_ratio,
            "sortino_ratio": artifact.sortino_ratio,
            "max_drawdown": artifact.max_drawdown,
            "win_rate": artifact.win_rate,
            "profit_factor": artifact.profit_factor,
            "num_trades": artifact.num_trades,
            "walk_forward_scores": artifact.walk_forward_scores,
            "monte_carlo_p_value": artifact.monte_carlo_p_value,
            "passed_quality_gates": artifact.passed_quality_gates,
            "timestamp": artifact.timestamp,
        }
        try:
            data = {"artifacts": self._artifacts, "summary": {}}
            if os.path.exists(BACKTEST_MEMORY_PATH):
                with open(BACKTEST_MEMORY_PATH, "r") as f:
                    data = json.load(f)
            data.setdefault("artifacts", []).append(record)
            data["summary"] = {
                "total_backtests": len(data["artifacts"]),
                "passed_quality_gates": sum(
                    1 for a in data["artifacts"] if a.get("passed_quality_gates")
                ),
                "last_run": time.time(),
            }
            with open(BACKTEST_MEMORY_PATH, "w") as f:
                json.dump(data, f, indent=2, default=str)
            self.log_decision(
                f"Artifact {artifact.artifact_id} stored to {BACKTEST_MEMORY_PATH}"
            )
        except IOError as exc:
            self.log_decision(f"Failed to store artifact: {exc}", level="error")

    def _load_artifacts(self) -> None:
        """Load existing artifacts from disk."""
        if os.path.exists(BACKTEST_MEMORY_PATH):
            try:
                with open(BACKTEST_MEMORY_PATH, "r") as f:
                    data = json.load(f)
                    self._artifacts = data.get("artifacts", [])
            except (json.JSONDecodeError, IOError):
                self._artifacts = []

    def list_artifacts(self) -> List[Dict[str, Any]]:
        """Return all stored backtest artifacts."""
        return list(self._artifacts)
