"""Market regime detection agent — entry point of the live trading pipeline.

The MarketAnalystAgent classifies the current market regime using Ichimoku
Kinko Hyo as the primary signal, supported by ADX (trend strength), ATR
(volatility), and OBV (volume trend).

Ichimoku components used:
    Tenkan-sen  (9-period midprice)  — fast momentum line
    Kijun-sen   (26-period midprice) — slow baseline / support-resistance
    Senkou Span A  = (tenkan + kijun) / 2
    Senkou Span B  = 52-period midprice
    Cloud (Kumo)   = region between Span A and Span B

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

    Uses Ichimoku Kinko Hyo as the primary regime signal:
        bull-trending : price above cloud, Span A > Span B, tenkan > kijun
        bear-trending : price below cloud, Span A < Span B, tenkan < kijun
        ranging       : price inside the cloud
        high-volatility: ATR > 2× 20-period average (overrides cloud position)
        low-volatility : ATR < 0.5× 20-period average
        transitioning  : price just crossed cloud boundary or weak signals
    """

    # Confidence scoring weights per signal source
    CONFIDENCE_WEIGHTS = {
        "ichimoku_cloud": 0.35,   # primary: price vs cloud
        "ichimoku_tk":    0.25,   # TK cross direction
        "adx":            0.20,   # trend strength confirmation
        "atr":            0.12,   # volatility context
        "obv":            0.08,   # volume confirmation
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
        """Compute Ichimoku + supporting indicators.

        Args:
            df: Clean OHLCV DataFrame (minimum 60 bars recommended).

        Returns:
            Dictionary of indicator names to their latest values.
        """
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        volume = df["volume"].values.astype(float)

        # --- Ichimoku Kinko Hyo (primary) ---
        ich = self._compute_ichimoku(high, low, close)

        # --- ADX (14) — trend strength, confirms Ichimoku direction ---
        adx, plus_di, minus_di = self._compute_adx(high, low, close, period=14)

        # --- ATR (14) — volatility regime detection ---
        atr, atr_ratio = self._compute_atr(high, low, close, period=14)

        # --- OBV trend — volume confirmation ---
        obv_trend = self._compute_obv_trend(close, volume)

        # --- Bollinger Bands (kept for ranging/mean-reversion downstream) ---
        bb_upper, bb_middle, bb_lower, bb_width, bb_pct_b = self._compute_bollinger(close, period=20)

        return {
            # Ichimoku
            "tenkan":       ich["tenkan"],
            "kijun":        ich["kijun"],
            "span_a":       ich["span_a"],
            "span_b":       ich["span_b"],
            "cloud_top":    ich["cloud_top"],
            "cloud_bot":    ich["cloud_bot"],
            "price_vs_cloud": ich["price_vs_cloud"],  # +1 above, 0 inside, -1 below
            "tk_cross":     ich["tk_cross"],           # +1 bull, -1 bear, 0 none
            "cloud_bullish": ich["cloud_bullish"],      # 1.0 if Span A > Span B
            # Supporting
            "adx":          float(adx),
            "plus_di":      float(plus_di),
            "minus_di":     float(minus_di),
            "atr":          float(atr),
            "atr_ratio":    float(atr_ratio),
            "obv_trend":    float(obv_trend),
            "bb_upper":     float(bb_upper),
            "bb_middle":    float(bb_middle),
            "bb_lower":     float(bb_lower),
            "bb_width":     float(bb_width),
            "bb_pct_b":     float(bb_pct_b),
            "close":        float(close[-1]),
            "ma_20":        float(np.mean(close[-20:])),
        }

    def _classify_regime(
        self, indicators: Dict[str, float]
    ) -> tuple[str, float]:
        """Classify market regime using Ichimoku as primary signal.

        Decision hierarchy:
          1. Volatility check (ATR) — overrides cloud position
          2. Cloud position (price above / inside / below)
          3. Cloud colour (Span A vs Span B)
          4. TK cross direction
          5. ADX for trend-strength confidence boost

        Args:
            indicators: Dictionary from _compute_indicators().

        Returns:
            Tuple of (regime_type, confidence_score).
        """
        price_vs_cloud  = indicators.get("price_vs_cloud", 0.0)   # +1 above, 0 inside, -1 below
        cloud_bullish   = indicators.get("cloud_bullish", 0.0)     # 1.0 = Span A > Span B
        tk_cross        = indicators.get("tk_cross", 0.0)          # +1 / -1 / 0
        tenkan          = indicators.get("tenkan", 0.0)
        kijun           = indicators.get("kijun", 0.0)
        adx             = indicators.get("adx", 20.0)
        plus_di         = indicators.get("plus_di", 25.0)
        minus_di        = indicators.get("minus_di", 25.0)
        atr_ratio       = indicators.get("atr_ratio", 1.0)
        obv_trend       = indicators.get("obv_trend", 0.0)
        bb_width        = indicators.get("bb_width", 0.0)

        confidence = 0.5  # neutral baseline

        # ── Step 1: volatility override ──────────────────────────────────────
        if atr_ratio > 2.0:
            regime_type = "high-volatility"
            confidence = 0.5 + min((atr_ratio - 2.0) * 0.12, 0.30)
            return regime_type, max(0.0, min(1.0, confidence))

        if atr_ratio < 0.5:
            regime_type = "low-volatility"
            confidence = 0.5 + min((0.5 - atr_ratio) * 0.25, 0.25)
            return regime_type, max(0.0, min(1.0, confidence))

        # ── Step 2: Ichimoku cloud position ──────────────────────────────────
        if price_vs_cloud == 0.0:
            # Price is inside the cloud — market is ranging/transitioning
            if bb_width < 0.015:
                regime_type = "ranging"
                confidence = 0.58
            else:
                regime_type = "transitioning"
                confidence = 0.38
            return regime_type, max(0.0, min(1.0, confidence))

        # ── Step 3: Price above cloud (bullish territory) ────────────────────
        if price_vs_cloud > 0:
            if cloud_bullish > 0:
                # Bullish cloud: Span A > Span B
                regime_type = "bull-trending"
                confidence = 0.55
                # Boosts
                if tenkan > kijun:
                    confidence += 0.12   # TK bullish alignment
                if tk_cross > 0:
                    confidence += 0.08   # Recent bullish TK cross
                if adx > 25:
                    confidence += 0.08   # Trend confirmed by ADX
                if adx > 30:
                    confidence += 0.05
                if plus_di > minus_di:
                    confidence += 0.05
                if obv_trend > 0.3:
                    confidence += 0.04
                # Contradictions
                if tenkan < kijun:
                    confidence -= 0.10   # Bearish TK despite bullish cloud
                if tk_cross < 0:
                    confidence -= 0.08   # Recent bearish cross — possibly transitioning
            else:
                # Bearish cloud above price — unusual, likely transitioning
                regime_type = "transitioning"
                confidence = 0.35
            return regime_type, max(0.0, min(1.0, confidence))

        # ── Step 4: Price below cloud (bearish territory) ────────────────────
        if cloud_bullish <= 0:
            # Bearish cloud: Span A < Span B
            regime_type = "bear-trending"
            confidence = 0.55
            # Boosts
            if tenkan < kijun:
                confidence += 0.12
            if tk_cross < 0:
                confidence += 0.08
            if adx > 25:
                confidence += 0.08
            if adx > 30:
                confidence += 0.05
            if minus_di > plus_di:
                confidence += 0.05
            if obv_trend < -0.3:
                confidence += 0.04
            # Contradictions
            if tenkan > kijun:
                confidence -= 0.10
            if tk_cross > 0:
                confidence -= 0.08
        else:
            # Bullish cloud below price — possible bottoming / transitioning
            regime_type = "transitioning"
            confidence = 0.38

        confidence = max(0.0, min(1.0, confidence))
        return regime_type, confidence

    # -------------------------------------------------------------------------
    # Indicator computation helpers
    # -------------------------------------------------------------------------

    def _compute_ichimoku(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> Dict[str, float]:
        """Compute Ichimoku Kinko Hyo values for the most recent bar.

        Lines:
            Tenkan-sen  (9-period midprice)
            Kijun-sen   (26-period midprice)
            Senkou Span A = (tenkan + kijun) / 2
            Senkou Span B = 52-period midprice

        Returns dict with tenkan, kijun, span_a, span_b, cloud_top, cloud_bot,
        price_vs_cloud (+1/0/-1), tk_cross (+1/-1/0), cloud_bullish (1/0).
        """
        def midprice(period: int) -> float:
            if len(high) < period:
                return float(close[-1])
            return (float(np.max(high[-period:])) + float(np.min(low[-period:]))) / 2.0

        tenkan = midprice(9)
        kijun  = midprice(26)
        span_a = (tenkan + kijun) / 2.0
        span_b = midprice(52)

        cloud_top = max(span_a, span_b)
        cloud_bot = min(span_a, span_b)
        price = float(close[-1])

        if price > cloud_top:
            price_vs_cloud = 1.0
        elif price < cloud_bot:
            price_vs_cloud = -1.0
        else:
            price_vs_cloud = 0.0

        # TK cross: compare current and previous bar
        if len(high) >= 10:
            prev_tenkan = (float(np.max(high[-10:-1])) + float(np.min(low[-10:-1]))) / 2.0
            prev_kijun  = midprice(26)   # kijun changes slowly; use same value as proxy
            prev_above = prev_tenkan > prev_kijun
            curr_above = tenkan > kijun
            if curr_above and not prev_above:
                tk_cross = 1.0   # bullish cross
            elif not curr_above and prev_above:
                tk_cross = -1.0  # bearish cross
            else:
                tk_cross = 0.0
        else:
            tk_cross = 0.0

        return {
            "tenkan":         tenkan,
            "kijun":          kijun,
            "span_a":         span_a,
            "span_b":         span_b,
            "cloud_top":      cloud_top,
            "cloud_bot":      cloud_bot,
            "price_vs_cloud": price_vs_cloud,
            "tk_cross":       tk_cross,
            "cloud_bullish":  1.0 if span_a > span_b else 0.0,
        }

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
