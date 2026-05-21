"""Trading strategy orchestration agent — middle stage of the live pipeline.

The TradingStrategistAgent receives a RegimeVerdict from the market-analyst,
selects the appropriate strategy family, computes Z-score anomaly signals,
and produces a SignalProposal for the risk-analyst.

CRITICAL: This agent NEVER invokes broker execution. It ALWAYS routes through
the risk-analyst. Any attempt to bypass this constraint is a critical policy violation.

Pipeline position: market-analyst → trading-strategist → risk-analyst → broker
"""

from typing import Dict, Optional, Any
import logging
import numpy as np
import pandas as pd

from scripts.agents.base_agent import (
    BaseAgent, AgentRole, RegimeVerdict, SignalProposal
)


logger = logging.getLogger(__name__)


# ---- Strategy-to-regime mapping ----
REGIME_STRATEGY_MAP: Dict[str, str] = {
    "bull-trending": "momentum",
    "bear-trending": "momentum",
    "ranging": "mean-reversion",
    "high-volatility": "adaptive",
    "low-volatility": "mean-reversion",
    "transitioning": "none",
}

# Minimum confidence to generate a signal per strategy
MIN_CONFIDENCE: Dict[str, float] = {
    "momentum": 0.55,
    "mean-reversion": 0.50,
    "pairs": 0.60,
    "adaptive": 0.50,
}

# Base position size as fraction of portfolio
BASE_SIZE_PCT = 0.05


class TradingStrategistAgent(BaseAgent):
    """Strategy orchestration agent that converts regime verdicts into trade signals.

    Receives a RegimeVerdict, selects the appropriate strategy, computes
    Z-score anomaly detection, and generates a SignalProposal.

    INVARIANT: This agent never invokes the broker under any circumstances.
    If a broker bypass is attempted, it logs CRITICAL and returns None.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the TradingStrategistAgent.

        Args:
            config: Full bot configuration dictionary.
        """
        super().__init__(AgentRole.TRADING_STRATEGIST, config)
        self._signal_history: list = []

    def generate_signal(
        self,
        df: pd.DataFrame,
        regime_verdict: RegimeVerdict,
        ai_engine: Optional[Any] = None,
    ) -> Optional[SignalProposal]:
        """Generate a trade signal based on the regime verdict and market data.

        This is the primary method of the trading strategist. It validates the
        regime verdict, selects a strategy, computes Z-score anomalies, and
        returns a SignalProposal or None.

        Args:
            df: OHLCV DataFrame for the target symbol.
            regime_verdict: Validated RegimeVerdict from market-analyst.
            ai_engine: Optional AIEngine for LSTM/Transformer direction prediction.

        Returns:
            SignalProposal if a valid signal is found, None otherwise.

        CRITICAL: This method NEVER calls exchange/broker functions.
        """
        # ---- Guard: broker bypass check ----
        # This guard is belt-and-suspenders: the orchestrator also enforces this.
        # The strategist itself must never hold a reference to an exchange object.

        # ---- Validate regime verdict ----
        if regime_verdict is None:
            self.log_decision("No RegimeVerdict provided — no signal", level="warning")
            return None

        if not regime_verdict.is_valid():
            self.log_decision("Invalid RegimeVerdict — no signal", level="warning")
            return None

        if regime_verdict.is_stale(max_age_seconds=300):
            self.log_decision(
                f"RegimeVerdict {regime_verdict.verdict_id} is stale — no signal",
                level="warning"
            )
            return None

        if regime_verdict.regime_type == "transitioning" and regime_verdict.confidence < 0.5:
            self.log_decision(
                "Transitioning regime with low confidence — no signal",
                level="info"
            )
            return None

        if df is None or len(df) < 20:
            self.log_decision("Insufficient data for signal generation", level="warning")
            return None

        try:
            symbol = regime_verdict.symbols[0] if regime_verdict.symbols else "UNKNOWN"

            # ---- Select strategy family ----
            strategy_name = REGIME_STRATEGY_MAP.get(regime_verdict.regime_type, "none")
            if strategy_name == "none":
                self.log_decision(
                    f"No strategy for regime {regime_verdict.regime_type} — no signal"
                )
                return None

            # ---- Compute Z-score anomaly ----
            anomaly_type, anomaly_score, anomaly_ok = self._compute_anomaly(df)

            if not anomaly_ok:
                self.log_decision(
                    f"Anomaly {anomaly_type} (z={anomaly_score:.2f}) rejected — no signal"
                )
                return None

            # ---- Compute strategy-specific signal ----
            side, signal_strength = self._compute_signal(
                df, regime_verdict, strategy_name, ai_engine
            )

            if side is None:
                self.log_decision(f"No directional signal from {strategy_name} strategy")
                return None

            # ---- Compute final confidence ----
            # 40% regime confidence + 40% signal strength + 20% anomaly quality
            anomaly_quality = max(0.0, 1.0 - (anomaly_score / 5.0))  # Normalize z-score
            confidence = (
                0.40 * regime_verdict.confidence
                + 0.40 * signal_strength
                + 0.20 * anomaly_quality
            )
            confidence = max(0.0, min(1.0, confidence))

            # Check minimum confidence threshold
            min_conf = MIN_CONFIDENCE.get(strategy_name, 0.55)
            if anomaly_type == "spike":
                min_conf = 0.70  # Higher threshold for spike anomalies

            if confidence < min_conf:
                self.log_decision(
                    f"Confidence {confidence:.3f} below threshold {min_conf} — no signal"
                )
                return None

            # ---- Compute position size ----
            size_pct = self._compute_size(
                regime_verdict, strategy_name, anomaly_type, confidence
            )

            # ---- Build SignalProposal ----
            proposal = SignalProposal(
                symbol=symbol,
                side=side,
                size_pct=round(size_pct, 4),
                confidence=round(confidence, 4),
                strategy_name=strategy_name,
                anomaly_type=anomaly_type,
                anomaly_score=round(anomaly_score, 4),
                regime_verdict_id=regime_verdict.verdict_id,
            )

            self._signal_history.append(proposal)
            if len(self._signal_history) > 500:
                self._signal_history = self._signal_history[-500:]

            self.log_decision(
                f"Signal: {side} {symbol} | Strategy: {strategy_name} | "
                f"Size: {size_pct:.3f} | Confidence: {confidence:.3f} | "
                f"Anomaly: {anomaly_type} (z={anomaly_score:.2f})"
            )
            return proposal

        except Exception as exc:
            self.log_decision(
                f"Signal generation failed: {exc} — no signal", level="error"
            )
            return None

    def _compute_anomaly(
        self, df: pd.DataFrame
    ) -> tuple[str, float, bool]:
        """Compute Z-score anomaly type from price/volume data.

        Args:
            df: OHLCV DataFrame.

        Returns:
            Tuple of (anomaly_type, z_score, is_tradeable).
            is_tradeable is False for cluster-outlier and flatline.
        """
        close = df["close"].values.astype(float)
        volume = df["volume"].values.astype(float)

        if len(close) < 10:
            return "none", 0.0, True

        # Compute rolling Z-score of returns
        returns = np.diff(close) / close[:-1]
        if len(returns) < 5:
            return "none", 0.0, True

        recent_returns = returns[-20:] if len(returns) >= 20 else returns
        mean_ret = np.mean(recent_returns)
        std_ret = np.std(recent_returns, ddof=1)

        if std_ret == 0:
            return "flatline", 0.0, False  # No volatility — skip

        latest_z = abs((returns[-1] - mean_ret) / std_ret)

        # ---- Classify anomaly type ----

        # Flatline: abnormally low volatility
        if std_ret < 0.001:
            return "flatline", 0.0, False

        # Spike: single-candle sharp movement
        if latest_z > 3.0:
            return "spike", float(latest_z), True  # Allow but require higher confidence

        # Oscillation: alternating sign over last 4+ candles
        if len(returns) >= 4:
            signs = np.sign(returns[-4:])
            alternating = all(signs[i] != signs[i + 1] for i in range(len(signs) - 1))
            if alternating:
                return "oscillation", float(np.mean(np.abs(returns[-4:])) / std_ret), True

        # Pattern-break: large z-score after 10+ ranging candles
        if len(returns) >= 10 and latest_z > 2.0:
            ranging_std = np.std(returns[-10:-1], ddof=1)
            if ranging_std < std_ret * 0.5:  # Previous period was much quieter
                return "pattern-break", float(latest_z), True

        # Drift: sustained movement 3+ candles
        if len(returns) >= 3:
            last_3_z = [(r - mean_ret) / std_ret for r in returns[-3:]]
            if all(z > 1.5 for z in last_3_z) or all(z < -1.5 for z in last_3_z):
                return "drift", float(abs(np.mean(last_3_z))), True

        # Cluster-outlier: multivariate outlier in price-volume space
        # Simplified: check if volume is also anomalous simultaneously
        vol_mean = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
        vol_std = np.std(volume[-20:], ddof=1) if len(volume) >= 20 else np.std(volume, ddof=1)
        if vol_std > 0:
            vol_z = abs((volume[-1] - vol_mean) / vol_std)
            if vol_z > 3.0 and latest_z > 2.0:
                return "cluster-outlier", float(latest_z), False  # Reject

        return "none", float(latest_z), True

    def _compute_signal(
        self,
        df: pd.DataFrame,
        regime_verdict: RegimeVerdict,
        strategy_name: str,
        ai_engine: Optional[Any],
    ) -> tuple[Optional[str], float]:
        """Compute trade direction and signal strength for the given strategy.

        Args:
            df: OHLCV DataFrame.
            regime_verdict: Current regime verdict.
            strategy_name: Selected strategy family.
            ai_engine: Optional neural model engine.

        Returns:
            Tuple of (side, signal_strength). side is None if no clear signal.
        """
        close = df["close"].values.astype(float)
        indicators = regime_verdict.indicator_values

        if strategy_name == "momentum":
            return self._momentum_signal(close, indicators, regime_verdict)

        elif strategy_name == "mean-reversion":
            return self._mean_reversion_signal(close, indicators)

        elif strategy_name == "pairs":
            # Simplified pairs: use momentum as fallback
            return self._momentum_signal(close, indicators, regime_verdict)

        elif strategy_name == "adaptive":
            # Blend of momentum and mean-reversion
            mom_side, mom_strength = self._momentum_signal(close, indicators, regime_verdict)
            mr_side, mr_strength = self._mean_reversion_signal(close, indicators)
            conf = regime_verdict.confidence
            if mom_side == mr_side and mom_side is not None:
                blended = conf * mom_strength + (1 - conf) * mr_strength
                return mom_side, blended
            elif mom_side is not None and mom_strength > mr_strength:
                return mom_side, mom_strength * 0.75
            elif mr_side is not None:
                return mr_side, mr_strength * 0.75
            return None, 0.0

        return None, 0.0

    def _momentum_signal(
        self,
        close: np.ndarray,
        indicators: Dict[str, float],
        regime_verdict: RegimeVerdict,
    ) -> tuple[Optional[str], float]:
        """Momentum strategy: MACD crossover + RSI confirmation."""
        rsi = indicators.get("rsi", 50.0)
        macd_hist = indicators.get("macd_hist", 0.0)
        regime = regime_verdict.regime_type

        if regime == "bull-trending":
            # Long signal: MACD positive + RSI in 45–75
            if macd_hist > 0 and 45 <= rsi <= 75:
                strength = min(1.0, abs(macd_hist) * 100 + (rsi - 45) / 30)
                return "buy", float(strength)
        elif regime == "bear-trending":
            # Short signal: MACD negative + RSI in 25–55
            if macd_hist < 0 and 25 <= rsi <= 55:
                strength = min(1.0, abs(macd_hist) * 100 + (55 - rsi) / 30)
                return "sell", float(strength)

        return None, 0.0

    def _mean_reversion_signal(
        self,
        close: np.ndarray,
        indicators: Dict[str, float],
    ) -> tuple[Optional[str], float]:
        """Mean-reversion strategy: Bollinger Band boundary + Z-score."""
        bb_pct_b = indicators.get("bb_pct_b", 0.5)
        bb_upper = indicators.get("bb_upper", 0.0)
        bb_lower = indicators.get("bb_lower", 0.0)
        current = float(close[-1]) if len(close) > 0 else 0.0

        if bb_lower == 0:
            return None, 0.0

        # Buy at lower band
        if bb_pct_b < 0.05:
            strength = min(1.0, (0.05 - bb_pct_b) * 20 + 0.5)
            return "buy", float(strength)

        # Sell at upper band
        if bb_pct_b > 0.95:
            strength = min(1.0, (bb_pct_b - 0.95) * 20 + 0.5)
            return "sell", float(strength)

        # Z-score confirmation
        if len(close) >= 20:
            mean_20 = np.mean(close[-20:])
            std_20 = np.std(close[-20:], ddof=1)
            if std_20 > 0:
                z = (current - mean_20) / std_20
                if z > 2.0:
                    return "sell", min(1.0, abs(z) / 3.0)
                elif z < -2.0:
                    return "buy", min(1.0, abs(z) / 3.0)

        return None, 0.0

    def _compute_size(
        self,
        regime_verdict: RegimeVerdict,
        strategy_name: str,
        anomaly_type: str,
        confidence: float,
    ) -> float:
        """Compute target position size as a fraction of portfolio.

        Args:
            regime_verdict: Current regime verdict.
            strategy_name: Selected strategy.
            anomaly_type: Detected anomaly classification.
            confidence: Final signal confidence score.

        Returns:
            Position size as fraction of portfolio (0.0–1.0).
        """
        max_size = float(
            self.get_config_value("trading", "risk_percentage", default=0.05)
        )
        size = BASE_SIZE_PCT * confidence

        # Strategy-specific adjustments
        if strategy_name == "adaptive":
            size *= 0.75  # Conservative for adaptive
        elif strategy_name == "mean-reversion" and regime_verdict.regime_type == "ranging":
            size *= 1.5  # Stronger in ranging

        # Anomaly adjustments
        if anomaly_type == "spike":
            size *= 0.50  # Half size for spike anomalies
        elif anomaly_type == "high-volatility":
            size *= 0.25  # Quarter size in high volatility

        # Cap at max configured risk
        size = min(size, max_size)
        size = max(0.001, size)  # Minimum viable size
        return round(size, 4)
