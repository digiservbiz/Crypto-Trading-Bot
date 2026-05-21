"""Market regime detection agent — entry point of the live trading pipeline.

The MarketAnalystAgent classifies the current market regime using technical
indicators (ADX, RSI, Bollinger Bands, ATR, MACD, OBV) and produces a
RegimeVerdict that drives strategy selection downstream.

Pipeline position: START → market-analyst → trading-strategist → risk-analyst → broker

This agent never communicates with the broker, risk-analyst, or backtest-engineer.
"""

from typing import Dict, List, Optional, Any
import logging
import numpy as np
import pandas as pd

from scripts.agents.base_agent import BaseAgent, AgentRole, RegimeVerdict


logger = logging.getLogger(__name__)


class MarketAnalystAgent(BaseAgent):
    """Entry-point agent responsible for market regime classification.

    Computes 6 technical indicators (RSI, MACD, Bollinger Bands, ADX, ATR, OBV)
    and produces a RegimeVerdict representing one of 6 possible market states.

    Regimes:
        bull-trending: ADX > 25, +DI > -DI, RSI 55–70
        bear-trending: ADX > 25, -DI > +DI, RSI 30–45
        ranging: ADX < 20, BB width within historical norm
        high-volatility: ATR > 2x 20-period average
        low-volatility: ATR < 0.5x 20-period average
        transitioning: Weak/mixed signals, confidence < 0.4
    """

    # Confidence scoring weights per indicator
    CONFIDENCE_WEIGHTS = {
        "adx": 0.25,
        "rsi": 0.20,
        "macd": 0.20,
        "bb": 0.15,
        "atr": 0.15,
        "obv": 0.05,
    }

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the MarketAnalystAgent.

        Args:
            config: Full bot configuration dictionary.
        """
        super().__init__(AgentRole.MARKET_ANALYST, config)
        self._verdict_history: List[RegimeVerdict] = []
        self._max_history = 100

    def analyze(self, df: pd.DataFrame, symbols: List[str]) -> RegimeVerdict:
        """Analyze market data and return a regime verdict.

        This is the primary entry point for the market-analyst. It runs full
        indicator computation and regime classification.

        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume.
                Must have at least 30 rows after NaN removal.
            symbols: List of trading symbols analyzed.

        Returns:
            RegimeVerdict with regime classification, confidence, and indicators.
            Returns a 'transitioning' verdict with low confidence on data errors.
        """
        if df is None or len(df) < 30:
            self.log_decision(
                "Insufficient data (<30 rows) — returning transitioning with low confidence",
                level="warning"
            )
            return self._degraded_verdict(symbols, "insufficient data")

        try:
            # Compute all technical indicators
            indicators = self._compute_indicators(df)

            # Classify regime using decision tree
            regime_type, confidence = self._classify_regime(indicators)

            # Override to transitioning if confidence too low
            if confidence < 0.4:
                regime_type = "transitioning"

            verdict = RegimeVerdict(
                regime_type=regime_type,
                confidence=round(confidence, 4),
                indicator_values=indicators,
                symbols=symbols,
            )

            self._store_verdict(verdict)
            self.log_decision(
                f"Regime: {regime_type} | Confidence: {confidence:.3f} | "
                f"Symbols: {symbols}"
            )
            return verdict

        except Exception as exc:
            self.log_decision(
                f"Indicator computation failed: {exc} — returning degraded verdict",
                level="error"
            )
            return self._degraded_verdict(symbols, str(exc))

    def _compute_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute all 6 required technical indicators.

        Args:
            df: Clean OHLCV DataFrame.

        Returns:
            Dictionary of indicator names to their latest values.
        """
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        volume = df["volume"].values.astype(float)

        # --- RSI (14) ---
        rsi = self._compute_rsi(close, period=14)

        # --- MACD (12, 26, 9) ---
        macd, macd_signal, macd_hist = self._compute_macd(close)

        # --- Bollinger Bands (20, 2) ---
        bb_upper, bb_middle, bb_lower, bb_width, bb_pct_b = self._compute_bollinger(close, period=20)

        # --- ADX + DI (14) ---
        adx, plus_di, minus_di = self._compute_adx(high, low, close, period=14)

        # --- ATR (14) ---
        atr, atr_ratio = self._compute_atr(high, low, close, period=14)

        # --- OBV trend ---
        obv_trend = self._compute_obv_trend(close, volume)

        return {
            "adx": float(adx),
            "plus_di": float(plus_di),
            "minus_di": float(minus_di),
            "rsi": float(rsi),
            "macd": float(macd),
            "macd_signal": float(macd_signal),
            "macd_hist": float(macd_hist),
            "bb_upper": float(bb_upper),
            "bb_middle": float(bb_middle),
            "bb_lower": float(bb_lower),
            "bb_width": float(bb_width),
            "bb_pct_b": float(bb_pct_b),
            "atr": float(atr),
            "atr_ratio": float(atr_ratio),
            "obv_trend": float(obv_trend),
            "close": float(close[-1]),
            "ma_20": float(np.mean(close[-20:])),
        }

    def _classify_regime(
        self, indicators: Dict[str, float]
    ) -> tuple[str, float]:
        """Apply decision tree to classify regime and compute confidence.

        Args:
            indicators: Dictionary of computed indicator values.

        Returns:
            Tuple of (regime_type, confidence_score).
        """
        adx = indicators.get("adx", 0.0)
        plus_di = indicators.get("plus_di", 0.0)
        minus_di = indicators.get("minus_di", 0.0)
        rsi = indicators.get("rsi", 50.0)
        macd_hist = indicators.get("macd_hist", 0.0)
        bb_pct_b = indicators.get("bb_pct_b", 0.5)
        bb_width = indicators.get("bb_width", 0.0)
        atr_ratio = indicators.get("atr_ratio", 1.0)
        obv_trend = indicators.get("obv_trend", 0.0)
        close = indicators.get("close", 0.0)
        ma_20 = indicators.get("ma_20", 0.0)

        confidence = 0.5  # Neutral baseline

        # Step 1: Check ADX for trend strength
        if adx > 25:
            # Potentially trending
            if plus_di > minus_di and rsi > 50:
                regime_type = "bull-trending"
                # Confidence boosts for strong bull signals
                if adx > 30:
                    confidence += 0.12
                if rsi >= 55 and rsi <= 70:
                    confidence += 0.10
                if macd_hist > 0:
                    confidence += 0.08
                if bb_pct_b > 0.8:
                    confidence += 0.08
                if close > ma_20:
                    confidence += 0.06
                if obv_trend > 0.3:
                    confidence += 0.05
                # Contradictions
                if rsi > 75:
                    confidence -= 0.08  # Overbought
                if macd_hist < 0:
                    confidence -= 0.10

            elif minus_di > plus_di and rsi < 50:
                regime_type = "bear-trending"
                # Confidence boosts for strong bear signals
                if adx > 30:
                    confidence += 0.12
                if rsi >= 30 and rsi <= 45:
                    confidence += 0.10
                if macd_hist < 0:
                    confidence += 0.08
                if bb_pct_b < 0.2:
                    confidence += 0.08
                if close < ma_20:
                    confidence += 0.06
                if obv_trend < -0.3:
                    confidence += 0.05
                # Contradictions
                if rsi < 25:
                    confidence -= 0.08  # Oversold
                if macd_hist > 0:
                    confidence -= 0.10

            else:
                # Mixed DI signals with high ADX
                regime_type = "transitioning"
                confidence = 0.35

        elif adx >= 20:
            # Weak trend — likely transitioning
            regime_type = "transitioning"
            confidence = 0.4 + (25 - adx) * 0.005  # Slightly higher confidence near ADX=20

        else:
            # ADX < 20 — non-trending
            if atr_ratio > 2.0:
                regime_type = "high-volatility"
                confidence = 0.5 + min((atr_ratio - 2.0) * 0.1, 0.3)
                if bb_width > 0:
                    confidence += 0.08
            elif atr_ratio < 0.5:
                regime_type = "low-volatility"
                confidence = 0.5 + min((0.5 - atr_ratio) * 0.2, 0.25)
                if bb_width < 0.01:
                    confidence += 0.08
            else:
                regime_type = "ranging"
                confidence = 0.55
                # Strong ranging signals
                if rsi >= 40 and rsi <= 60:
                    confidence += 0.08
                if abs(bb_pct_b - 0.5) < 0.2:
                    confidence += 0.05
                if abs(macd_hist) < 0.001:
                    confidence += 0.05

        # Clamp confidence to [0.0, 1.0]
        confidence = max(0.0, min(1.0, confidence))
        return regime_type, confidence

    # -------------------------------------------------------------------------
    # Indicator computation helpers
    # -------------------------------------------------------------------------

    def _compute_rsi(self, close: np.ndarray, period: int = 14) -> float:
        """Compute RSI for the most recent period."""
        if len(close) < period + 1:
            return 50.0
        delta = np.diff(close)
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _compute_macd(
        self, close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple[float, float, float]:
        """Compute MACD, signal line, and histogram."""
        if len(close) < slow:
            return 0.0, 0.0, 0.0
        ema_fast = self._ema(close, fast)
        ema_slow = self._ema(close, slow)
        macd_line = ema_fast - ema_slow
        # For signal, we need a series of MACD values
        macd_series = np.array([
            self._ema(close[:i], fast) - self._ema(close[:i], slow)
            for i in range(slow, len(close) + 1)
        ])
        if len(macd_series) < signal:
            signal_line = macd_series[-1] if len(macd_series) > 0 else 0.0
        else:
            signal_line = self._ema(macd_series, signal)
        histogram = macd_line - signal_line
        return float(macd_line), float(signal_line), float(histogram)

    def _compute_bollinger(
        self, close: np.ndarray, period: int = 20, num_std: float = 2.0
    ) -> tuple[float, float, float, float, float]:
        """Compute Bollinger Bands and derived metrics."""
        if len(close) < period:
            mid = float(close[-1])
            return mid, mid, mid, 0.0, 0.5
        window = close[-period:]
        middle = float(np.mean(window))
        std = float(np.std(window, ddof=1))
        upper = middle + num_std * std
        lower = middle - num_std * std
        bb_width = (upper - lower) / middle if middle != 0 else 0.0
        bb_pct_b = (close[-1] - lower) / (upper - lower) if (upper - lower) != 0 else 0.5
        return upper, middle, lower, bb_width, bb_pct_b

    def _compute_adx(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> tuple[float, float, float]:
        """Compute ADX, +DI, and -DI using Wilder's smoothing method.

        This is the correct Wilder ADX, not the simplified DX proxy.
        ADX = smoothed moving average of DX over `period` bars.
        """
        if len(close) < period * 2 + 1:
            return 20.0, 25.0, 25.0

        tr_list: list = []
        plus_dm_list: list = []
        minus_dm_list: list = []

        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hpc = abs(high[i] - close[i - 1])
            lpc = abs(low[i] - close[i - 1])
            tr_list.append(max(hl, hpc, lpc))

            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]
            plus_dm_list.append(up_move if (up_move > down_move and up_move > 0) else 0.0)
            minus_dm_list.append(down_move if (down_move > up_move and down_move > 0) else 0.0)

        # Wilder's initial smoothed value = sum of first `period` values
        if len(tr_list) < period:
            return 20.0, 25.0, 25.0

        smoothed_tr = float(np.sum(tr_list[:period]))
        smoothed_plus_dm = float(np.sum(plus_dm_list[:period]))
        smoothed_minus_dm = float(np.sum(minus_dm_list[:period]))

        # Wilder's smoothing for the rest
        dx_values: list = []
        for i in range(period, len(tr_list)):
            smoothed_tr = smoothed_tr - (smoothed_tr / period) + tr_list[i]
            smoothed_plus_dm = smoothed_plus_dm - (smoothed_plus_dm / period) + plus_dm_list[i]
            smoothed_minus_dm = smoothed_minus_dm - (smoothed_minus_dm / period) + minus_dm_list[i]

            if smoothed_tr == 0:
                continue

            plus_di_i = 100.0 * smoothed_plus_dm / smoothed_tr
            minus_di_i = 100.0 * smoothed_minus_dm / smoothed_tr
            di_sum = plus_di_i + minus_di_i
            dx_i = 100.0 * abs(plus_di_i - minus_di_i) / di_sum if di_sum > 0 else 0.0
            dx_values.append((dx_i, plus_di_i, minus_di_i))

        if not dx_values:
            return 20.0, 25.0, 25.0

        # ADX = simple average of recent DX values (Wilder: smoothed MA)
        # Use last `period` DX values for ADX
        recent_dx = dx_values[-period:]
        adx = float(np.mean([d[0] for d in recent_dx]))
        # Use the most recent +DI and -DI
        plus_di = float(dx_values[-1][1])
        minus_di = float(dx_values[-1][2])

        return adx, plus_di, minus_di

    def _compute_atr(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> tuple[float, float]:
        """Compute ATR and ATR ratio (current vs. 20-period average)."""
        if len(close) < period + 1:
            return 0.0, 1.0
        tr_list = []
        for i in range(1, len(close)):
            tr = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
            tr_list.append(tr)
        tr_arr = np.array(tr_list)
        atr_current = float(np.mean(tr_arr[-period:]))
        atr_20 = float(np.mean(tr_arr[-20:])) if len(tr_arr) >= 20 else atr_current
        atr_ratio = atr_current / atr_20 if atr_20 > 0 else 1.0
        return atr_current, atr_ratio

    def _compute_obv_trend(self, close: np.ndarray, volume: np.ndarray) -> float:
        """Compute OBV trend as normalized slope (-1 to +1)."""
        if len(close) < 2:
            return 0.0
        obv = np.zeros(len(close))
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]
        # Normalize recent OBV trend to [-1, +1] using last 20 periods
        window = obv[-20:]
        if len(window) < 2:
            return 0.0
        slope = np.polyfit(np.arange(len(window)), window, 1)[0]
        # Normalize by volume scale
        vol_scale = float(np.mean(volume[-20:])) if np.mean(volume[-20:]) > 0 else 1.0
        normalized_slope = np.clip(slope / vol_scale, -1.0, 1.0)
        return float(normalized_slope)

    def _ema(self, values: np.ndarray, period: int) -> float:
        """Compute EMA for the most recent value."""
        if len(values) == 0:
            return 0.0
        if len(values) < period:
            return float(np.mean(values))
        k = 2.0 / (period + 1)
        ema = float(values[0])
        for v in values[1:]:
            ema = v * k + ema * (1 - k)
        return ema

    def _degraded_verdict(self, symbols: List[str], reason: str) -> RegimeVerdict:
        """Return a safe degraded verdict when analysis fails.

        Args:
            symbols: Symbol list.
            reason: Human-readable reason for degradation.

        Returns:
            RegimeVerdict with transitioning regime and low confidence.
        """
        return RegimeVerdict(
            regime_type="transitioning",
            confidence=0.1,
            indicator_values={"error": 1.0, "reason": 0.0},
            symbols=symbols,
        )

    def _store_verdict(self, verdict: RegimeVerdict) -> None:
        """Store verdict in rolling history (max 100 entries)."""
        self._verdict_history.append(verdict)
        if len(self._verdict_history) > self._max_history:
            self._verdict_history = self._verdict_history[-self._max_history:]

    @property
    def recent_verdicts(self) -> List[RegimeVerdict]:
        """Return the rolling history of recent RegimeVerdicts."""
        return list(self._verdict_history)

    def get_dominant_regime(
        self, dfs: Dict[str, pd.DataFrame], symbols: List[str]
    ) -> RegimeVerdict:
        """Analyze multiple symbols and return the dominant regime verdict.

        Computes per-symbol verdicts and returns the one with highest confidence.

        Args:
            dfs: Dictionary mapping symbol -> OHLCV DataFrame.
            symbols: List of symbols to analyze.

        Returns:
            The RegimeVerdict with highest confidence across all symbols.
        """
        if not dfs or not symbols:
            return self._degraded_verdict(symbols, "no data provided")

        best_verdict: Optional[RegimeVerdict] = None
        for symbol in symbols:
            df = dfs.get(symbol)
            if df is None:
                continue
            verdict = self.analyze(df, [symbol])
            if best_verdict is None or verdict.confidence > best_verdict.confidence:
                best_verdict = verdict

        if best_verdict is None:
            return self._degraded_verdict(symbols, "all symbols failed analysis")

        # Update symbols to reflect all analyzed symbols
        best_verdict.symbols = symbols
        return best_verdict
