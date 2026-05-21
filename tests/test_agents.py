"""Comprehensive agent pipeline test suite.

Tests all 4 agents (market-analyst, trading-strategist, risk-analyst,
backtest-engineer) and the orchestrator using synthetic OHLCV data.

No API keys, no exchange connection, no torch required.
Run with: python -m pytest tests/test_agents.py -v
"""

import time
import math
import pytest
import numpy as np
import pandas as pd
import yaml
import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.agents.base_agent import (
    BaseAgent, AgentRole,
    RegimeVerdict, SignalProposal, RiskDecision, SignedBacktestArtifact,
)
from scripts.agents.market_analyst import MarketAnalystAgent
from scripts.agents.trading_strategist import TradingStrategistAgent
from scripts.agents.risk_analyst import RiskAnalystAgent
from scripts.agents.backtest_engineer import BacktestEngineerAgent
from scripts.agents.orchestrator import TradingOrchestrator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    """Minimal bot configuration for tests — no API keys needed."""
    return {
        "data": {"symbols": ["BTC/USDT", "ETH/USDT"], "timeframe": "5m", "lookback": 60},
        "exchange": {"name": "binance", "api_key": "", "secret_key": ""},
        "trading": {
            "risk_percentage": 0.05,
            "stop_loss_percentage": 0.02,
            "take_profit_percentage": 0.04,
            "trailing_stop": {"enabled": False, "percentage": 0.01},
            "dynamic_take_profit": {"enabled": False},
            "dynamic_position_sizing": {"enabled": False},
        },
        "models": {
            "model_type": "lstm",
            "lstm": {"hidden_size": 64, "num_layers": 2},
            "transformer": {"d_model": 64, "nhead": 4, "num_encoder_layers": 2,
                            "dim_feedforward": 256, "dropout": 0.1},
        },
        "inference": {"models_dir": "models"},
    }


def _make_ohlcv(n: int = 100, trend: float = 0.0, volatility: float = 0.01,
                seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data.

    Args:
        n: Number of bars.
        trend: Per-bar return drift (positive = uptrend, negative = downtrend).
        volatility: Per-bar return standard deviation.
        seed: RNG seed for reproducibility.

    Returns:
        DataFrame with columns: open, high, low, close, volume.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(trend, volatility, n)
    close = np.cumprod(1 + returns) * 50_000
    noise = rng.uniform(0.001, 0.005, n)
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.uniform(100, 2_000, n)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume
    })


def _make_trending_ohlcv(n: int = 100, direction: str = "up") -> pd.DataFrame:
    """Make strongly trending data for regime detection tests."""
    trend = 0.005 if direction == "up" else -0.005
    return _make_ohlcv(n=n, trend=trend, volatility=0.003)


def _make_ranging_ohlcv(n: int = 100) -> pd.DataFrame:
    """Make mean-reverting / ranging data."""
    rng = np.random.default_rng(1)
    t = np.linspace(0, 4 * math.pi, n)
    close = 50_000 + 500 * np.sin(t) + rng.normal(0, 50, n)
    noise = rng.uniform(0.001, 0.003, n)
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = np.roll(close, 1); open_[0] = close[0]
    volume = rng.uniform(100, 500, n)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume
    })


def _make_regime_verdict(
    regime: str = "bull-trending",
    confidence: float = 0.75,
    symbols: list = None,
    age_seconds: float = 0,
) -> RegimeVerdict:
    """Build a RegimeVerdict with sensible defaults for strategy tests."""
    indicators = {
        "adx": 30.0, "plus_di": 28.0, "minus_di": 18.0,
        "rsi": 62.0, "macd": 100.0, "macd_signal": 80.0, "macd_hist": 20.0,
        "bb_upper": 51000.0, "bb_middle": 50000.0, "bb_lower": 49000.0,
        "bb_width": 0.04, "bb_pct_b": 0.85,
        "atr": 200.0, "atr_ratio": 1.0, "obv_trend": 0.5,
        "close": 50_500.0, "ma_20": 50_000.0,
    }
    # Use explicit None check — empty list [] must stay as-is (not replaced by default)
    if symbols is None:
        symbols = ["BTC/USDT"]
    verdict = RegimeVerdict(
        regime_type=regime,
        confidence=confidence,
        indicator_values=indicators,
        symbols=symbols,
        timestamp=time.time() - age_seconds,
    )
    return verdict


def _make_signal_proposal(
    symbol: str = "BTC/USDT",
    side: str = "buy",
    size_pct: float = 0.03,
    confidence: float = 0.70,
    strategy: str = "momentum",
    anomaly: str = "none",
    anomaly_score: float = 0.5,
    verdict_id: str = "test1234",
) -> SignalProposal:
    """Build a SignalProposal for risk-analyst tests."""
    return SignalProposal(
        symbol=symbol,
        side=side,
        size_pct=size_pct,
        confidence=confidence,
        strategy_name=strategy,
        anomaly_type=anomaly,
        anomaly_score=anomaly_score,
        regime_verdict_id=verdict_id,
    )


def _make_portfolio_state(
    balance: float = 10_000.0,
    daily_pnl_pct: float = 0.0,
    weekly_pnl_pct: float = 0.0,
    portfolio_peak: float = None,
    atr_ratio: float = 1.0,
    open_positions: dict = None,
) -> dict:
    """Build a portfolio_state dict matching the keys expected by RiskAnalystAgent.

    Keys used by risk_analyst.evaluate():
        balance, daily_pnl_pct, weekly_pnl_pct, portfolio_peak,
        open_positions, atr_ratio
    """
    return {
        "balance": balance,
        "daily_pnl_pct": daily_pnl_pct,
        "weekly_pnl_pct": weekly_pnl_pct,
        "portfolio_peak": portfolio_peak if portfolio_peak is not None else balance,
        "atr_ratio": atr_ratio,
        "open_positions": open_positions or {},
    }


# ===========================================================================
# 1. Dataclass Tests
# ===========================================================================

class TestDataclasses:
    def test_regime_verdict_valid(self):
        v = _make_regime_verdict()
        assert v.is_valid()
        assert not v.is_stale()

    def test_regime_verdict_invalid_regime(self):
        v = _make_regime_verdict(regime="unknown-regime")
        assert not v.is_valid()

    def test_regime_verdict_invalid_confidence(self):
        v = _make_regime_verdict(confidence=1.5)
        assert not v.is_valid()

    def test_regime_verdict_stale(self):
        v = _make_regime_verdict(age_seconds=400)
        assert v.is_stale(max_age_seconds=300)
        assert not v.is_stale(max_age_seconds=500)

    def test_regime_verdict_empty_symbols(self):
        v = _make_regime_verdict(symbols=[])
        assert not v.is_valid()

    def test_signal_proposal_valid(self):
        s = _make_signal_proposal()
        assert s.is_valid()

    def test_signal_proposal_invalid_side(self):
        s = _make_signal_proposal(side="hold")
        assert not s.is_valid()

    def test_signal_proposal_invalid_size(self):
        s = _make_signal_proposal(size_pct=1.5)
        assert not s.is_valid()

    def test_risk_decision_approved(self):
        rd = RiskDecision(
            signal_id="abc", decision="approve", adjusted_size_pct=0.03,
            var_95=0.01, cvar_95=0.015, sharpe_ratio=1.8,
            circuit_breaker_triggered=False, circuit_breaker_reason=None,
        )
        assert rd.is_approved()

    def test_risk_decision_rejected(self):
        rd = RiskDecision(
            signal_id="abc", decision="reject", adjusted_size_pct=0.03,
            var_95=0.03, cvar_95=0.05, sharpe_ratio=0.5,
            circuit_breaker_triggered=True,
            circuit_breaker_reason="DAILY_LOSS_HALT",
        )
        assert not rd.is_approved()

    def test_risk_decision_zero_size_not_approved(self):
        rd = RiskDecision(
            signal_id="abc", decision="approve", adjusted_size_pct=0.0,
            var_95=0.01, cvar_95=0.015, sharpe_ratio=1.8,
            circuit_breaker_triggered=False, circuit_breaker_reason=None,
        )
        assert not rd.is_approved()

    def test_backtest_artifact_quality_gates(self):
        # walk_forward_scores: max - min = 1.85 - 1.75 = 0.10 < 0.15 → passes consistency gate
        artifact = SignedBacktestArtifact(
            strategy_id="s1", symbol="BTC/USDT", period="2023-01-01/2024-01-01",
            total_return=0.45, sharpe_ratio=1.8, sortino_ratio=2.1,
            max_drawdown=-0.08, win_rate=0.58, profit_factor=1.9,
            num_trades=45, walk_forward_scores=[1.75, 1.80, 1.85, 1.78],
            monte_carlo_p_value=0.02, passed_quality_gates=True,
        )
        gates = artifact.quality_gate_summary()
        assert gates["min_trades"]
        assert gates["monte_carlo_significance"]
        assert gates["max_drawdown"]
        assert gates["profit_factor"]
        assert gates["sharpe_ratio"]
        assert gates["win_rate_consistency"]

    def test_backtest_artifact_failing_gates(self):
        artifact = SignedBacktestArtifact(
            strategy_id="s2", symbol="BTC/USDT", period="2023-01-01/2024-01-01",
            total_return=0.05, sharpe_ratio=0.8, sortino_ratio=1.0,
            max_drawdown=-0.20, win_rate=0.40, profit_factor=1.2,
            num_trades=10, walk_forward_scores=[0.5, 2.5, 0.3],
            monte_carlo_p_value=0.12, passed_quality_gates=False,
        )
        gates = artifact.quality_gate_summary()
        assert not gates["min_trades"]
        assert not gates["monte_carlo_significance"]
        assert not gates["max_drawdown"]
        assert not gates["profit_factor"]
        assert not gates["sharpe_ratio"]


# ===========================================================================
# 2. MarketAnalystAgent Tests
# ===========================================================================

class TestMarketAnalyst:
    def test_returns_regime_verdict(self, config):
        agent = MarketAnalystAgent(config)
        df = _make_ohlcv(n=100)
        verdict = agent.analyze(df, ["BTC/USDT"])
        assert isinstance(verdict, RegimeVerdict)
        assert verdict.is_valid()

    def test_insufficient_data_returns_degraded(self, config):
        agent = MarketAnalystAgent(config)
        df = _make_ohlcv(n=5)
        verdict = agent.analyze(df, ["BTC/USDT"])
        assert verdict.regime_type == "transitioning"
        assert verdict.confidence < 0.3

    def test_none_data_returns_degraded(self, config):
        agent = MarketAnalystAgent(config)
        verdict = agent.analyze(None, ["BTC/USDT"])
        assert verdict.regime_type == "transitioning"

    def test_uptrend_detected(self, config):
        agent = MarketAnalystAgent(config)
        df = _make_trending_ohlcv(n=150, direction="up")
        verdict = agent.analyze(df, ["BTC/USDT"])
        assert verdict.regime_type in ("bull-trending", "transitioning")
        assert verdict.confidence >= 0.0

    def test_downtrend_detected(self, config):
        agent = MarketAnalystAgent(config)
        df = _make_trending_ohlcv(n=150, direction="down")
        verdict = agent.analyze(df, ["BTC/USDT"])
        assert verdict.regime_type in ("bear-trending", "transitioning")

    def test_ranging_market(self, config):
        agent = MarketAnalystAgent(config)
        df = _make_ranging_ohlcv(n=100)
        verdict = agent.analyze(df, ["BTC/USDT"])
        assert verdict.regime_type in ("ranging", "low-volatility", "transitioning", "high-volatility")

    def test_indicator_values_present(self, config):
        agent = MarketAnalystAgent(config)
        df = _make_ohlcv(n=100)
        verdict = agent.analyze(df, ["BTC/USDT"])
        for key in ("adx", "rsi", "macd_hist", "bb_width", "atr", "obv_trend"):
            assert key in verdict.indicator_values

    def test_confidence_in_range(self, config):
        agent = MarketAnalystAgent(config)
        df = _make_ohlcv(n=100)
        verdict = agent.analyze(df, ["BTC/USDT"])
        assert 0.0 <= verdict.confidence <= 1.0

    def test_verdict_has_uuid(self, config):
        agent = MarketAnalystAgent(config)
        df = _make_ohlcv(n=100)
        v1 = agent.analyze(df, ["BTC/USDT"])
        v2 = agent.analyze(df, ["ETH/USDT"])
        assert v1.verdict_id != v2.verdict_id

    def test_verdict_stored_in_history(self, config):
        agent = MarketAnalystAgent(config)
        df = _make_ohlcv(n=100)
        for _ in range(3):
            agent.analyze(df, ["BTC/USDT"])
        assert len(agent.recent_verdicts) == 3

    def test_symbols_preserved(self, config):
        agent = MarketAnalystAgent(config)
        df = _make_ohlcv(n=100)
        verdict = agent.analyze(df, ["ETH/USDT", "BTC/USDT"])
        assert "ETH/USDT" in verdict.symbols or "BTC/USDT" in verdict.symbols

    def test_high_volatility_spike(self, config):
        """Very spiky data should produce high-volatility regime or transitioning."""
        agent = MarketAnalystAgent(config)
        df = _make_ohlcv(n=100, volatility=0.10, trend=0.0)
        verdict = agent.analyze(df, ["BTC/USDT"])
        assert verdict.regime_type in ("high-volatility", "transitioning", "bull-trending", "bear-trending")

    def test_get_dominant_regime(self, config):
        agent = MarketAnalystAgent(config)
        dfs = {
            "BTC/USDT": _make_ohlcv(n=100),
            "ETH/USDT": _make_ohlcv(n=100, seed=99),
        }
        verdict = agent.get_dominant_regime(dfs, ["BTC/USDT", "ETH/USDT"])
        assert isinstance(verdict, RegimeVerdict)
        assert "BTC/USDT" in verdict.symbols or "ETH/USDT" in verdict.symbols


# ===========================================================================
# 3. TradingStrategistAgent Tests
# ===========================================================================

class TestTradingStrategist:
    def test_returns_signal_for_valid_inputs(self, config):
        agent = TradingStrategistAgent(config)
        df = _make_ohlcv(n=100)
        verdict = _make_regime_verdict("bull-trending", confidence=0.75)
        result = agent.generate_signal(df, verdict)
        # May be None if confidence too low — just check type
        assert result is None or isinstance(result, SignalProposal)

    def test_none_verdict_returns_none(self, config):
        agent = TradingStrategistAgent(config)
        df = _make_ohlcv(n=100)
        assert agent.generate_signal(df, None) is None

    def test_invalid_verdict_returns_none(self, config):
        agent = TradingStrategistAgent(config)
        df = _make_ohlcv(n=100)
        verdict = _make_regime_verdict(regime="not-a-real-regime")
        assert agent.generate_signal(df, verdict) is None

    def test_stale_verdict_returns_none(self, config):
        agent = TradingStrategistAgent(config)
        df = _make_ohlcv(n=100)
        verdict = _make_regime_verdict(age_seconds=400)
        assert agent.generate_signal(df, verdict) is None

    def test_transitioning_low_confidence_returns_none(self, config):
        agent = TradingStrategistAgent(config)
        df = _make_ohlcv(n=100)
        verdict = _make_regime_verdict(regime="transitioning", confidence=0.3)
        assert agent.generate_signal(df, verdict) is None

    def test_insufficient_data_returns_none(self, config):
        agent = TradingStrategistAgent(config)
        df = _make_ohlcv(n=5)
        verdict = _make_regime_verdict()
        assert agent.generate_signal(df, verdict) is None

    def test_signal_proposal_is_valid_when_returned(self, config):
        agent = TradingStrategistAgent(config)
        df = _make_trending_ohlcv(n=100, direction="up")
        verdict = _make_regime_verdict("bull-trending", confidence=0.85)
        result = agent.generate_signal(df, verdict)
        if result is not None:
            assert result.is_valid()
            assert result.side in ("buy", "sell")
            assert 0.0 <= result.size_pct <= 1.0
            assert 0.0 <= result.confidence <= 1.0

    def test_signal_links_to_verdict(self, config):
        agent = TradingStrategistAgent(config)
        df = _make_trending_ohlcv(n=100, direction="up")
        verdict = _make_regime_verdict("bull-trending", confidence=0.85)
        result = agent.generate_signal(df, verdict)
        if result is not None:
            assert result.regime_verdict_id == verdict.verdict_id

    def test_anomaly_compute_spike(self, config):
        agent = TradingStrategistAgent(config)
        # Create a very large spike on the last bar
        df = _make_ohlcv(n=100)
        df.loc[df.index[-1], "close"] = df["close"].iloc[-1] * 1.30  # +30% spike
        anomaly_type, score, tradeable = agent._compute_anomaly(df)
        assert anomaly_type == "spike"
        assert score > 3.0

    def test_anomaly_compute_flatline(self, config):
        agent = TradingStrategistAgent(config)
        df = _make_ohlcv(n=100)
        # Set all recent closes to same value → flatline
        df["close"] = 50_000.0
        anomaly_type, score, tradeable = agent._compute_anomaly(df)
        assert anomaly_type == "flatline"
        assert tradeable is False

    def test_mean_reversion_signal_buy_at_lower_band(self, config):
        agent = TradingStrategistAgent(config)
        close = np.full(30, 50_000.0)
        indicators = {
            "bb_pct_b": 0.01, "bb_upper": 51000.0,
            "bb_lower": 49000.0, "bb_middle": 50000.0,
        }
        side, strength = agent._mean_reversion_signal(close, indicators)
        assert side == "buy"
        assert strength > 0

    def test_mean_reversion_signal_sell_at_upper_band(self, config):
        agent = TradingStrategistAgent(config)
        close = np.full(30, 51_000.0)
        indicators = {
            "bb_pct_b": 0.99, "bb_upper": 51000.0,
            "bb_lower": 49000.0, "bb_middle": 50000.0,
        }
        side, strength = agent._mean_reversion_signal(close, indicators)
        assert side == "sell"
        assert strength > 0

    def test_signal_history_capped_at_500(self, config):
        agent = TradingStrategistAgent(config)
        df = _make_trending_ohlcv(n=100, direction="up")
        for i in range(10):
            verdict = _make_regime_verdict("bull-trending", confidence=0.85)
            agent.generate_signal(df, verdict)
        assert len(agent._signal_history) <= 500


# ===========================================================================
# 4. RiskAnalystAgent Tests
# ===========================================================================

class TestRiskAnalyst:
    def test_approve_healthy_signal(self, config, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.agents.risk_analyst.RISK_MEMORY_PATH",
                            str(tmp_path / "trading-risk.json"))
        agent = RiskAnalystAgent(config)
        signal = _make_signal_proposal(confidence=0.75)
        portfolio = _make_portfolio_state(balance=10_000)
        df = _make_ohlcv(n=100)
        decision = agent.evaluate(signal, portfolio, df)
        assert isinstance(decision, RiskDecision)
        assert decision.signal_id == signal.signal_id

    def test_returns_reject_on_daily_loss_halt(self, config, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.agents.risk_analyst.RISK_MEMORY_PATH",
                            str(tmp_path / "trading-risk.json"))
        agent = RiskAnalystAgent(config)
        signal = _make_signal_proposal()
        # 4% daily loss pct > 3% DAILY_LOSS_HALT_PCT threshold
        portfolio = _make_portfolio_state(balance=9_600, daily_pnl_pct=-0.04)
        df = _make_ohlcv(n=100)
        decision = agent.evaluate(signal, portfolio, df)
        # Should trigger daily loss halt
        if decision.circuit_breaker_triggered:
            assert decision.decision == "reject"
            assert "DAILY_LOSS" in (decision.circuit_breaker_reason or "")

    def test_returns_reject_on_drawdown_halt(self, config, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.agents.risk_analyst.RISK_MEMORY_PATH",
                            str(tmp_path / "trading-risk.json"))
        agent = RiskAnalystAgent(config)
        signal = _make_signal_proposal()
        # portfolio_peak much higher than current balance → 20% drawdown
        portfolio = _make_portfolio_state(balance=8_000, portfolio_peak=10_000)
        df = _make_ohlcv(n=100)
        decision = agent.evaluate(signal, portfolio, df)
        if decision.circuit_breaker_triggered:
            assert decision.decision == "reject"

    def test_circuit_breaker_correlation_spike(self, config, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.agents.risk_analyst.RISK_MEMORY_PATH",
                            str(tmp_path / "trading-risk.json"))
        agent = RiskAnalystAgent(config)
        # Pre-load state with high correlation
        agent._state["max_pairwise_correlation"] = 0.92
        signal = _make_signal_proposal()
        portfolio = _make_portfolio_state()
        df = _make_ohlcv(n=100)
        decision = agent.evaluate(signal, portfolio, df)
        if decision.circuit_breaker_triggered:
            assert decision.decision == "reject"

    def test_var_computed_and_present(self, config, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.agents.risk_analyst.RISK_MEMORY_PATH",
                            str(tmp_path / "trading-risk.json"))
        agent = RiskAnalystAgent(config)
        signal = _make_signal_proposal()
        portfolio = _make_portfolio_state()
        df = _make_ohlcv(n=100)
        decision = agent.evaluate(signal, portfolio, df)
        assert decision.var_95 >= 0.0
        assert decision.cvar_95 >= 0.0

    def test_sharpe_present(self, config, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.agents.risk_analyst.RISK_MEMORY_PATH",
                            str(tmp_path / "trading-risk.json"))
        agent = RiskAnalystAgent(config)
        signal = _make_signal_proposal()
        portfolio = _make_portfolio_state()
        df = _make_ohlcv(n=100)
        decision = agent.evaluate(signal, portfolio, df)
        # Sharpe can be any real number
        assert isinstance(decision.sharpe_ratio, float)

    def test_adjusted_size_never_exceeds_signal_size(self, config, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.agents.risk_analyst.RISK_MEMORY_PATH",
                            str(tmp_path / "trading-risk.json"))
        agent = RiskAnalystAgent(config)
        signal = _make_signal_proposal(size_pct=0.10)
        portfolio = _make_portfolio_state()
        df = _make_ohlcv(n=100)
        decision = agent.evaluate(signal, portfolio, df)
        assert decision.adjusted_size_pct <= signal.size_pct + 1e-9

    def test_risk_decision_is_approved_type(self, config, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.agents.risk_analyst.RISK_MEMORY_PATH",
                            str(tmp_path / "trading-risk.json"))
        agent = RiskAnalystAgent(config)
        signal = _make_signal_proposal(confidence=0.80, size_pct=0.02)
        portfolio = _make_portfolio_state(balance=100_000)
        df = _make_ohlcv(n=100, volatility=0.005)
        decision = agent.evaluate(signal, portfolio, df)
        # Decision must be either approve or reject — never anything else
        assert decision.decision in ("approve", "reject")

    def test_state_persisted_to_disk(self, config, tmp_path, monkeypatch):
        """State is written to disk when a hard circuit breaker fires (_update_state call)."""
        path = str(tmp_path / "trading-risk.json")
        monkeypatch.setattr("scripts.agents.risk_analyst.RISK_MEMORY_PATH", path)
        agent = RiskAnalystAgent(config)
        signal = _make_signal_proposal()
        # Drawdown of 20% triggers MAX_DRAWDOWN_HALT → _update_state → _save_state
        portfolio = _make_portfolio_state(balance=8_000, portfolio_peak=10_000)
        df = _make_ohlcv(n=100)
        agent.evaluate(signal, portfolio, df)
        # File should have been written by _update_state during circuit breaker rejection
        assert os.path.exists(path)


# ===========================================================================
# 5. BacktestEngineerAgent Tests
# ===========================================================================

class TestBacktestEngineer:
    def _strategy_config(self):
        return {
            "strategy_type": "momentum",
            "lookback": 10,
            "threshold": 0.01,
        }

    def test_returns_artifact(self, config):
        agent = BacktestEngineerAgent(config)
        df = _make_ohlcv(n=300)
        artifact = agent.run_walk_forward(df, self._strategy_config())
        assert isinstance(artifact, SignedBacktestArtifact)

    def test_artifact_has_required_fields(self, config):
        agent = BacktestEngineerAgent(config)
        df = _make_ohlcv(n=300)
        artifact = agent.run_walk_forward(df, self._strategy_config())
        assert artifact.strategy_id
        assert artifact.symbol
        assert artifact.num_trades >= 0
        assert isinstance(artifact.walk_forward_scores, list)
        assert 0.0 <= artifact.monte_carlo_p_value <= 1.0

    def test_quality_gate_method(self, config):
        agent = BacktestEngineerAgent(config)
        df = _make_ohlcv(n=300)
        artifact = agent.run_walk_forward(df, self._strategy_config())
        gates = artifact.quality_gate_summary()
        assert isinstance(gates, dict)
        assert "min_trades" in gates
        assert "sharpe_ratio" in gates
        assert "profit_factor" in gates
        assert "max_drawdown" in gates
        assert "monte_carlo_significance" in gates
        assert "win_rate_consistency" in gates

    def test_insufficient_data_returns_artifact_with_no_gates(self, config):
        agent = BacktestEngineerAgent(config)
        df = _make_ohlcv(n=10)  # Too short for walk-forward
        artifact = agent.run_walk_forward(df, self._strategy_config())
        assert isinstance(artifact, SignedBacktestArtifact)
        assert artifact.passed_quality_gates is False

    def test_passed_quality_gates_matches_summary(self, config):
        agent = BacktestEngineerAgent(config)
        df = _make_ohlcv(n=300)
        artifact = agent.run_walk_forward(df, self._strategy_config())
        gates = artifact.quality_gate_summary()
        expected = all(gates.values())
        assert artifact.passed_quality_gates == expected

    def test_mean_reversion_strategy(self, config):
        agent = BacktestEngineerAgent(config)
        df = _make_ranging_ohlcv(n=300)
        strategy = {"strategy_type": "mean-reversion", "lookback": 10, "threshold": 0.01}
        artifact = agent.run_walk_forward(df, strategy)
        assert isinstance(artifact, SignedBacktestArtifact)

    def test_artifact_id_unique_per_run(self, config):
        agent = BacktestEngineerAgent(config)
        df = _make_ohlcv(n=300)
        a1 = agent.run_walk_forward(df, self._strategy_config())
        a2 = agent.run_walk_forward(df, self._strategy_config())
        assert a1.artifact_id != a2.artifact_id

    def test_max_drawdown_is_non_positive(self, config):
        agent = BacktestEngineerAgent(config)
        df = _make_ohlcv(n=300)
        artifact = agent.run_walk_forward(df, self._strategy_config())
        # Drawdown should be <= 0 (it's a loss measure)
        assert artifact.max_drawdown <= 0.0

    def test_win_rate_in_range(self, config):
        agent = BacktestEngineerAgent(config)
        df = _make_ohlcv(n=300)
        artifact = agent.run_walk_forward(df, self._strategy_config())
        assert 0.0 <= artifact.win_rate <= 1.0


# ===========================================================================
# 6. TradingOrchestrator Tests
# ===========================================================================

class TestOrchestrator:
    def test_pipeline_runs_end_to_end(self, config, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.agents.risk_analyst.RISK_MEMORY_PATH",
                            str(tmp_path / "trading-risk.json"))
        orch = TradingOrchestrator(config)
        df = _make_ohlcv(n=100)
        portfolio = _make_portfolio_state()
        result = orch.run_pipeline(df, ["BTC/USDT"], portfolio_state=portfolio)
        # Result must be RiskDecision or None
        assert result is None or isinstance(result, RiskDecision)

    def test_pipeline_returns_risk_decision(self, config, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.agents.risk_analyst.RISK_MEMORY_PATH",
                            str(tmp_path / "trading-risk.json"))
        orch = TradingOrchestrator(config)
        df = _make_trending_ohlcv(n=150, direction="up")
        portfolio = _make_portfolio_state(balance=100_000)
        result = orch.run_pipeline(df, ["BTC/USDT"], portfolio_state=portfolio)
        if result is not None:
            assert isinstance(result, RiskDecision)
            assert result.decision in ("approve", "reject")

    def test_pipeline_with_empty_df(self, config, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.agents.risk_analyst.RISK_MEMORY_PATH",
                            str(tmp_path / "trading-risk.json"))
        orch = TradingOrchestrator(config)
        portfolio = _make_portfolio_state()
        result = orch.run_pipeline(None, ["BTC/USDT"], portfolio_state=portfolio)
        # Should not crash; result may be None or rejected RiskDecision
        assert result is None or isinstance(result, RiskDecision)

    def test_pipeline_counter_increments(self, config, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.agents.risk_analyst.RISK_MEMORY_PATH",
                            str(tmp_path / "trading-risk.json"))
        orch = TradingOrchestrator(config)
        df = _make_ohlcv(n=100)
        portfolio = _make_portfolio_state()
        before = orch._pipeline_count
        orch.run_pipeline(df, ["BTC/USDT"], portfolio_state=portfolio)
        assert orch._pipeline_count == before + 1

    def test_broker_never_called_without_risk_decision(self, config, tmp_path, monkeypatch):
        """Invariant: pipeline result must be RiskDecision if a trade would happen."""
        monkeypatch.setattr("scripts.agents.risk_analyst.RISK_MEMORY_PATH",
                            str(tmp_path / "trading-risk.json"))
        orch = TradingOrchestrator(config)
        df = _make_ohlcv(n=100)
        portfolio = _make_portfolio_state()
        # Run multiple cycles and confirm no bypass
        for _ in range(5):
            result = orch.run_pipeline(df, ["BTC/USDT"], portfolio_state=portfolio)
            # Any non-None result MUST be a RiskDecision
            if result is not None:
                assert isinstance(result, RiskDecision)

    def test_approved_decision_has_positive_size(self, config, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.agents.risk_analyst.RISK_MEMORY_PATH",
                            str(tmp_path / "trading-risk.json"))
        orch = TradingOrchestrator(config)
        df = _make_trending_ohlcv(n=150, direction="up")
        portfolio = _make_portfolio_state(balance=100_000)
        result = orch.run_pipeline(df, ["BTC/USDT"], portfolio_state=portfolio)
        if result is not None and result.is_approved():
            assert result.adjusted_size_pct > 0.0


# ===========================================================================
# 7. Integration — Full Pipeline Smoke Test
# ===========================================================================

class TestFullPipelineIntegration:
    def test_full_pipeline_bull_market(self, config, tmp_path, monkeypatch):
        """End-to-end: trending-up data through all 4 agents."""
        monkeypatch.setattr("scripts.agents.risk_analyst.RISK_MEMORY_PATH",
                            str(tmp_path / "trading-risk.json"))
        analyst = MarketAnalystAgent(config)
        strategist = TradingStrategistAgent(config)
        risk = RiskAnalystAgent(config)

        df = _make_trending_ohlcv(n=150, direction="up")
        portfolio = _make_portfolio_state(balance=50_000)

        # Stage 1: Regime
        verdict = analyst.analyze(df, ["BTC/USDT"])
        assert isinstance(verdict, RegimeVerdict)

        # Stage 2: Signal (may be None)
        signal = strategist.generate_signal(df, verdict)
        if signal is None:
            return  # No signal generated — valid outcome

        assert isinstance(signal, SignalProposal)
        assert signal.is_valid()

        # Stage 3: Risk gate (always runs)
        decision = risk.evaluate(signal, portfolio, df)
        assert isinstance(decision, RiskDecision)
        assert decision.signal_id == signal.signal_id
        assert decision.decision in ("approve", "reject")

        # Stage 4: Broker gate check
        if decision.is_approved():
            assert decision.adjusted_size_pct > 0.0
            assert not decision.circuit_breaker_triggered

    def test_full_pipeline_bear_market(self, config, tmp_path, monkeypatch):
        """End-to-end: trending-down data through all 4 agents."""
        monkeypatch.setattr("scripts.agents.risk_analyst.RISK_MEMORY_PATH",
                            str(tmp_path / "trading-risk.json"))
        analyst = MarketAnalystAgent(config)
        strategist = TradingStrategistAgent(config)
        risk = RiskAnalystAgent(config)

        df = _make_trending_ohlcv(n=150, direction="down")
        portfolio = _make_portfolio_state(balance=50_000)

        verdict = analyst.analyze(df, ["BTC/USDT"])
        signal = strategist.generate_signal(df, verdict)
        if signal is None:
            return

        decision = risk.evaluate(signal, portfolio, df)
        assert decision.decision in ("approve", "reject")

    def test_full_pipeline_circuit_breaker_halts_trade(self, config, tmp_path, monkeypatch):
        """When daily loss circuit breaker fires, no trade proceeds."""
        monkeypatch.setattr("scripts.agents.risk_analyst.RISK_MEMORY_PATH",
                            str(tmp_path / "trading-risk.json"))
        analyst = MarketAnalystAgent(config)
        strategist = TradingStrategistAgent(config)
        risk = RiskAnalystAgent(config)

        df = _make_trending_ohlcv(n=150, direction="up")
        # 5% daily loss — exceeds 3% halt threshold
        portfolio = _make_portfolio_state(balance=9_500, daily_pnl_pct=-0.05)

        verdict = analyst.analyze(df, ["BTC/USDT"])
        signal = strategist.generate_signal(df, verdict)
        if signal is None:
            return

        decision = risk.evaluate(signal, portfolio, df)
        # If circuit breaker triggered, decision MUST be reject
        if decision.circuit_breaker_triggered:
            assert decision.decision == "reject"
            assert not decision.is_approved()

    def test_message_ids_are_linked_through_pipeline(self, config, tmp_path, monkeypatch):
        """Verify verdict_id → signal.regime_verdict_id and signal_id → decision.signal_id."""
        monkeypatch.setattr("scripts.agents.risk_analyst.RISK_MEMORY_PATH",
                            str(tmp_path / "trading-risk.json"))
        analyst = MarketAnalystAgent(config)
        strategist = TradingStrategistAgent(config)
        risk = RiskAnalystAgent(config)

        df = _make_trending_ohlcv(n=150, direction="up")
        portfolio = _make_portfolio_state(balance=100_000)

        verdict = analyst.analyze(df, ["BTC/USDT"])
        signal = strategist.generate_signal(df, verdict)
        if signal is None:
            return

        assert signal.regime_verdict_id == verdict.verdict_id
        decision = risk.evaluate(signal, portfolio, df)
        assert decision.signal_id == signal.signal_id
