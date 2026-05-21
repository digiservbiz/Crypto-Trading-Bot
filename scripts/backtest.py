"""Enhanced backtesting entry point using the BacktestEngineerAgent.

This module provides the CLI entry point for running comprehensive strategy
backtests using the ruflo BacktestEngineerAgent. It supports:
- Walk-forward validation (6-month train / 1-month test windows)
- Monte Carlo simulation (1000 bootstrapped runs)
- Quality gate enforcement (6 gates)
- Full metrics: total_return, sharpe, sortino, max_drawdown, win_rate, profit_factor

The backtest-engineer is an ORTHOGONAL research lane — it never interacts
with the live trading pipeline or real exchange accounts.

Usage:
    python scripts/backtest.py --config config.yaml --symbol BTC/USDT
    python scripts/backtest.py --config config.yaml --symbol ETH/USDT --strategy mean-reversion
    python scripts/backtest.py --config config.yaml --list
"""

import argparse
import json
import logging
import sys
from typing import Dict, Any, Optional

import pandas as pd
import yaml

from scripts.agents.backtest_engineer import BacktestEngineerAgent
from scripts.agents.base_agent import SignedBacktestArtifact


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_ohlcv_data(config: Dict[str, Any], symbol: str) -> Optional[pd.DataFrame]:
    """Load OHLCV data for a symbol from the configured data path.

    Attempts to load from:
    1. Symbol-specific file: data/{symbol_safe}.csv
    2. Default sample file: config.data.sample_data_path

    Args:
        config: Bot configuration dict.
        symbol: Trading pair (e.g., "BTC/USDT").

    Returns:
        OHLCV DataFrame or None if data not found.
    """
    symbol_safe = symbol.replace("/", "_")

    # Try symbol-specific file first
    paths_to_try = [
        f"data/{symbol_safe}.csv",
        f"data/{symbol_safe}_ohlcv.csv",
        config.get("data", {}).get("sample_data_path", "data/sample_ohlcv.csv"),
    ]

    for path in paths_to_try:
        try:
            df = pd.read_csv(path)
            logger.info("Loaded %d rows from %s", len(df), path)
            # Ensure required columns
            if "close" not in df.columns:
                logger.warning("Missing 'close' column in %s", path)
                continue
            # Add missing columns with defaults
            if "open" not in df.columns:
                df["open"] = df["close"]
            if "high" not in df.columns:
                df["high"] = df["close"]
            if "low" not in df.columns:
                df["low"] = df["close"]
            if "volume" not in df.columns:
                df["volume"] = 1.0
            if "timestamp" not in df.columns:
                df["timestamp"] = range(len(df))
            return df
        except FileNotFoundError:
            continue
        except Exception as exc:
            logger.error("Error loading %s: %s", path, exc)
            continue

    logger.error("No data found for symbol %s", symbol)
    return None


def run_backtest(
    config: Dict[str, Any],
    symbol: str,
    strategy_type: str = "momentum",
    strategy_id: Optional[str] = None,
    lookback: int = 20,
    z_threshold: float = 1.5,
) -> SignedBacktestArtifact:
    """Run a full backtest using the BacktestEngineerAgent.

    This function coordinates:
    1. Data loading
    2. BacktestEngineerAgent initialization
    3. Walk-forward validation
    4. Quality gate evaluation
    5. Results display

    Args:
        config: Bot configuration dictionary.
        symbol: Trading pair to backtest (e.g., "BTC/USDT").
        strategy_type: Strategy type: momentum | mean-reversion | adaptive
        strategy_id: Optional strategy identifier. Auto-generated if None.
        lookback: Lookback period for strategy parameters.
        z_threshold: Z-score entry threshold for strategy.

    Returns:
        SignedBacktestArtifact with full results.
    """
    logger.info(
        "Starting backtest | symbol=%s | strategy=%s", symbol, strategy_type
    )

    # Load data
    df = load_ohlcv_data(config, symbol)
    if df is None:
        logger.error("Cannot run backtest — no data available for %s", symbol)
        # Return failed artifact
        from scripts.agents.base_agent import SignedBacktestArtifact
        import time, uuid
        return SignedBacktestArtifact(
            strategy_id=strategy_id or "unknown",
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

    # Build strategy config
    if strategy_id is None:
        strategy_id = f"{strategy_type}-{symbol.replace('/', '-').lower()}-v1"

    strategy_config = {
        "strategy_id": strategy_id,
        "strategy_type": strategy_type,
        "symbol": symbol,
        "parameters": {
            "lookback": lookback,
            "threshold": z_threshold,
        },
    }

    # Initialize and run backtest-engineer
    agent = BacktestEngineerAgent(config)
    artifact = agent.run_walk_forward(df, strategy_config)

    return artifact


def print_results(artifact: SignedBacktestArtifact) -> None:
    """Print formatted backtest results to stdout.

    Args:
        artifact: Completed SignedBacktestArtifact.
    """
    separator = "=" * 60
    print(f"\n{separator}")
    print(f"  BACKTEST RESULTS — {artifact.strategy_id}")
    print(f"{separator}")
    print(f"  Symbol        : {artifact.symbol}")
    print(f"  Period        : {artifact.period}")
    print(f"  Artifact ID   : {artifact.artifact_id}")
    print(f"{separator}")
    print(f"  Total Return  : {artifact.total_return * 100:.2f}%")
    print(f"  Sharpe Ratio  : {artifact.sharpe_ratio:.4f}")
    print(f"  Sortino Ratio : {artifact.sortino_ratio:.4f}")
    print(f"  Max Drawdown  : {artifact.max_drawdown * 100:.2f}%")
    print(f"  Win Rate      : {artifact.win_rate * 100:.2f}%")
    print(f"  Profit Factor : {artifact.profit_factor:.4f}")
    print(f"  Num Trades    : {artifact.num_trades}")
    print(f"{separator}")
    print(f"  Walk-Forward Scores:")
    for i, score in enumerate(artifact.walk_forward_scores):
        print(f"    Window {i + 1:02d}   : Sharpe = {score:.4f}")
    print(f"  Monte Carlo p : {artifact.monte_carlo_p_value:.4f}")
    print(f"{separator}")

    # Quality gates
    gates = artifact.quality_gate_summary()
    print("  Quality Gates:")
    for gate, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {gate}")

    print(f"{separator}")
    overall = "PASSED" if artifact.passed_quality_gates else "FAILED"
    print(f"  OVERALL: {overall}")
    print(f"{separator}\n")


def legacy_backtest(config: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy backtest function for backward compatibility.

    Runs a simple backtest and returns metrics in the original format.
    Internally uses the BacktestEngineerAgent for calculation.

    Args:
        config: Bot configuration dict.

    Returns:
        Dict with: total_return, sharpe_ratio, max_drawdown, calmar_ratio,
                   cumulative_returns (as list)
    """
    symbol = config.get("data", {}).get("symbols", ["BTC/USDT"])[0]
    artifact = run_backtest(config, symbol)

    # Build cumulative returns for backward compatibility
    cumulative = [1.0 + artifact.total_return]

    # Estimate Calmar ratio
    calmar = (
        artifact.total_return / abs(artifact.max_drawdown)
        if artifact.max_drawdown < 0 else 0.0
    )

    return {
        "total_return": artifact.total_return,
        "sharpe_ratio": artifact.sharpe_ratio,
        "max_drawdown": artifact.max_drawdown,
        "calmar_ratio": calmar,
        "cumulative_returns": cumulative,
        "artifact": artifact,
    }


def main() -> None:
    """CLI entry point for backtesting."""
    parser = argparse.ArgumentParser(
        description="Crypto Trading Bot — Backtest Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/backtest.py --config config.yaml --symbol BTC/USDT
  python scripts/backtest.py --config config.yaml --symbol ETH/USDT --strategy mean-reversion
  python scripts/backtest.py --config config.yaml --list
        """
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config.yaml"
    )
    parser.add_argument(
        "--symbol", default=None, help="Trading pair (e.g., BTC/USDT)"
    )
    parser.add_argument(
        "--strategy",
        choices=["momentum", "mean-reversion", "adaptive"],
        default="momentum",
        help="Strategy type to backtest",
    )
    parser.add_argument(
        "--strategy-id", default=None, help="Custom strategy identifier"
    )
    parser.add_argument(
        "--lookback", type=int, default=20, help="Lookback period for strategy"
    )
    parser.add_argument(
        "--z-threshold", type=float, default=1.5, help="Z-score entry threshold"
    )
    parser.add_argument(
        "--list", action="store_true", help="List existing backtest artifacts"
    )
    parser.add_argument(
        "--output-json", default=None, help="Save results to JSON file"
    )

    args = parser.parse_args()

    # Load config
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    # List mode
    if args.list:
        agent = BacktestEngineerAgent(config)
        artifacts = agent.list_artifacts()
        if not artifacts:
            print("No backtest artifacts found.")
        else:
            print(f"\nFound {len(artifacts)} backtest artifacts:\n")
            for a in artifacts:
                status = "PASSED" if a.get("passed_quality_gates") else "FAILED"
                print(
                    f"  [{status}] {a.get('strategy_id')} | {a.get('symbol')} | "
                    f"Sharpe={a.get('sharpe_ratio', 0):.3f} | "
                    f"Return={a.get('total_return', 0)*100:.1f}%"
                )
        return

    # Determine symbol
    if args.symbol is None:
        symbols = config.get("data", {}).get("symbols", ["BTC/USDT"])
        symbol = symbols[0] if symbols else "BTC/USDT"
        logger.info("No symbol specified — using first configured: %s", symbol)
    else:
        symbol = args.symbol

    # Run backtest
    artifact = run_backtest(
        config=config,
        symbol=symbol,
        strategy_type=args.strategy,
        strategy_id=args.strategy_id,
        lookback=args.lookback,
        z_threshold=args.z_threshold,
    )

    # Print results
    print_results(artifact)

    # Save to JSON if requested
    if args.output_json:
        result_dict = {
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
            "quality_gates": artifact.quality_gate_summary(),
            "timestamp": artifact.timestamp,
        }
        with open(args.output_json, "w") as f:
            json.dump(result_dict, f, indent=2)
        logger.info("Results saved to %s", args.output_json)

    # Exit with non-zero code if quality gates failed
    if not artifact.passed_quality_gates:
        sys.exit(1)


if __name__ == "__main__":
    main()
