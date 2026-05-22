"""Main trading bot loop with ruflo multi-agent orchestration.

This module runs the primary live trading loop. It uses the TradingOrchestrator
to coordinate the full agent pipeline:
    market-analyst → trading-strategist → risk-analyst → broker

CRITICAL: The broker is ONLY invoked when the risk-analyst returns a
RiskDecision with decision == "approve" and circuit_breaker_triggered == False.
No trade is ever executed without this authorization.

Architecture:
    - Streamlit UI for monitoring
    - Prometheus metrics for observability
    - Telegram notifications for alerts
    - Agent pipeline for all trading decisions
"""

import time
import logging
import json
import os
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
from prometheus_client import Gauge, Counter
from datetime import datetime, timezone

# Graceful torch import (not needed for core bot loop)
try:
    import torch  # noqa: F401
except ImportError:
    torch = None  # type: ignore

from scripts.inference.ai_engine import AIEngine
from scripts.exchange import Exchange
from scripts.notifier import Notifier
from scripts.logger import get_logger
from scripts.agents.orchestrator import TradingOrchestrator
from scripts.agents.base_agent import RiskDecision


logger = get_logger(__name__)

# ---- Prometheus Metrics ----
BALANCE = Gauge("bot_balance", "Current balance in USDT")
TOTAL_PROFIT_LOSS = Gauge("bot_total_profit_loss", "Total profit or loss in USDT")
OPEN_POSITIONS = Gauge("bot_open_positions", "Number of open positions")
TRADES = Gauge("bot_trades", "Trades", ["symbol", "type", "price", "quantity"])
REGIME_GAUGE = Gauge("bot_market_regime", "Current market regime code",
                     ["symbol", "regime"])
PIPELINE_CYCLES = Counter("bot_pipeline_cycles_total", "Total pipeline cycles run")
PIPELINE_APPROVALS = Counter("bot_pipeline_approvals_total", "Approved signals")
PIPELINE_REJECTIONS = Counter("bot_pipeline_rejections_total", "Rejected signals")
CIRCUIT_BREAKER_EVENTS = Counter("bot_circuit_breaker_total", "Circuit breaker triggers",
                                  ["reason"])

# Regime type → numeric code for Prometheus
REGIME_CODES = {
    "bull-trending": 2,
    "bear-trending": -2,
    "ranging": 0,
    "high-volatility": 1,
    "low-volatility": -1,
    "transitioning": 0,
}


def _add_indicators(df: pd.DataFrame) -> None:
    """Add technical indicators to OHLCV DataFrame using pure pandas/numpy.

    Replaces pandas-ta dependency. Appends columns in-place:
        MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
        RSI_14
        BBL_20_2.0, BBM_20_2.0, BBU_20_2.0
        ATRr_14  (ATR ratio vs 20-bar average)
        OBV

    Args:
        df: DataFrame with columns open, high, low, close, volume.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # --- MACD (12, 26, 9) ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["MACD_12_26_9"] = macd_line
    df["MACDs_12_26_9"] = signal_line
    df["MACDh_12_26_9"] = macd_line - signal_line

    # --- RSI (14) ---
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # --- Bollinger Bands (20, 2) ---
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std(ddof=1)
    df["BBL_20_2.0"] = bb_mid - 2 * bb_std
    df["BBM_20_2.0"] = bb_mid
    df["BBU_20_2.0"] = bb_mid + 2 * bb_std

    # --- ATR (14) ---
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    atr_20 = tr.rolling(20).mean()
    df["ATRr_14"] = atr / atr_20.replace(0, float("nan"))

    # --- OBV ---
    direction = np.sign(close.diff()).fillna(0)
    df["OBV"] = (direction * volume).cumsum()


BOT_STATE_PATH = "data/state/bot-state.json"
_pnl_history: list = []          # module-level rolling P&L log


def _write_bot_state(
    running: bool,
    balance: float,
    initial_balance: float,
    daily_pnl_pct: float,
    weekly_pnl_pct: float,
    positions: Dict[str, Any],
    entry_prices: Dict[str, float],
    current_prices: Dict[str, float],
    regimes: Dict[str, Any],
    pipeline_stats: Dict[str, int],
    risk_state: Dict[str, Any],
    last_risk_metrics: Dict[str, float],
    recent_signals: list,
    recent_verdicts: list,
) -> None:
    """Write live bot state to disk so the Streamlit dashboard can read it.

    Called at the end of every trading cycle. Non-blocking: errors are silently
    logged so a disk issue never interrupts the trading loop.
    """
    global _pnl_history

    # Build positions dict with P&L
    pos_out: Dict[str, Any] = {}
    for sym, side in positions.items():
        if side is None:
            continue
        entry = entry_prices.get(sym, 0.0)
        current = current_prices.get(sym, entry)
        pnl_pct = ((current - entry) / entry) if entry > 0 else 0.0
        if side == "sell":
            pnl_pct = -pnl_pct
        pos_out[sym] = {
            "side": side,
            "entry_price": entry,
            "current_price": current,
            "pnl_pct": pnl_pct,
        }

    # Rolling P&L history (max 500 points)
    _pnl_history.append({"timestamp": time.time(), "balance": balance})
    if len(_pnl_history) > 500:
        _pnl_history = _pnl_history[-500:]

    # Infer circuit breaker states from risk_state
    daily_loss = risk_state.get("daily_loss_pct", 0.0)
    weekly_loss = risk_state.get("weekly_loss_pct", 0.0)
    drawdown = risk_state.get("current_drawdown_pct", 0.0)
    max_corr = risk_state.get("max_pairwise_correlation", 0.0)

    state = {
        "running": running,
        "last_update": time.time(),
        "balance": balance,
        "initial_balance": initial_balance,
        "daily_pnl_pct": daily_pnl_pct,
        "weekly_pnl_pct": weekly_pnl_pct,
        "positions": pos_out,
        "regimes": regimes,
        "pipeline": pipeline_stats,
        "circuit_breakers": {
            "daily_loss_halt": abs(daily_loss) >= 0.03,
            "weekly_reduction": abs(weekly_loss) >= 0.05,
            "max_drawdown_halt": drawdown >= 0.15,
            "correlation_spike": max_corr > 0.85,
            "vix_spike": False,          # Updated via atr_ratio in risk analyst
            "concentration_limit": False,
        },
        "last_risk_metrics": last_risk_metrics,
        "recent_signals": recent_signals[-20:],
        "recent_verdicts": recent_verdicts[-20:],
        "pnl_history": _pnl_history[-200:],
    }

    try:
        os.makedirs(os.path.dirname(BOT_STATE_PATH), exist_ok=True)
        with open(BOT_STATE_PATH, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as exc:
        logger.debug("Could not write bot state: %s", exc)


def get_news(symbol: str) -> list:
    """Fetch recent news articles for a cryptocurrency symbol.

    Args:
        symbol: Cryptocurrency symbol (e.g., "BTC").

    Returns:
        List of news article dicts from CryptoCompare, or empty list on error.
    """
    try:
        url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&categories={symbol}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json().get("Data", [])
    except Exception as exc:
        logger.warning("News fetch failed for %s: %s", symbol, exc)
        return []


# Module-level Flair classifier singleton — loaded once, reused across all calls
_flair_classifier: Any = None


def _get_flair_classifier() -> Any:
    """Lazily load the Flair sentiment classifier exactly once (singleton).

    Loading TextClassifier from disk takes 5-10s. Calling it inside a loop
    every 60 seconds would cause unacceptable latency.
    """
    global _flair_classifier
    if _flair_classifier is None:
        try:
            from flair.models import TextClassifier
            _flair_classifier = TextClassifier.load("en-sentiment")
            logger.info("Flair sentiment classifier loaded")
        except Exception as exc:
            logger.warning("Flair classifier unavailable: %s — sentiment disabled", exc)
    return _flair_classifier


def get_sentiment(text: str) -> float:
    """Analyze sentiment of text using the flair library (singleton classifier).

    Args:
        text: News article body or title.

    Returns:
        Sentiment score in [-1.0, +1.0]. Positive = bullish, negative = bearish.
    """
    classifier = _get_flair_classifier()
    if classifier is None:
        return 0.0
    try:
        from flair.data import Sentence
        sentence = Sentence(text[:512])  # Limit length
        classifier.predict(sentence)
        score = sentence.labels[0].score
        direction = 1 if sentence.labels[0].value == "POSITIVE" else -1
        return float(score * direction)
    except Exception as exc:
        logger.debug("Sentiment analysis failed: %s", exc)
        return 0.0


def compute_sentiment_score(symbol: str, config: Dict[str, Any]) -> float:
    """Compute weighted sentiment score from recent news.

    Args:
        symbol: Trading symbol (e.g., "BTC/USDT").
        config: Bot configuration dict.

    Returns:
        Weighted sentiment score in [-1.0, +1.0].
    """
    if not config.get("sentiment_analysis", {}).get("enabled", False):
        return 0.0

    base_symbol = symbol.split("/")[0]
    news = get_news(base_symbol)
    if not news:
        return 0.0

    weighted_sentiments = []
    decay = config["sentiment_analysis"].get("time_decay_factor", 0.95)
    source_weights = config["sentiment_analysis"].get("source_weights", {})

    for article in news:
        try:
            published = article.get("published_on", 0)
            age_hours = (
                datetime.now(timezone.utc)
                - datetime.fromtimestamp(published, tz=timezone.utc)
            ).total_seconds() / 3600
            time_decay = decay ** age_hours
            source_weight = source_weights.get(article.get("source", ""), 0.5)
            sentiment_score = get_sentiment(article.get("body", ""))
            weighted_sentiments.append(sentiment_score * time_decay * source_weight)
        except Exception:
            continue

    return float(sum(weighted_sentiments) / len(weighted_sentiments)) if weighted_sentiments else 0.0


def build_portfolio_state(
    exchange: Exchange,
    initial_balance: float,
    start_of_day_balance: float,
    start_of_week_balance: float,
    portfolio_peak: float,
    open_positions: Dict[str, Any],
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """Build the portfolio state dict for risk evaluation.

    Args:
        exchange: Exchange wrapper instance.
        initial_balance: Balance at bot start.
        start_of_day_balance: Balance at UTC midnight today.
        start_of_week_balance: Balance at start of this week.
        portfolio_peak: All-time peak portfolio balance.
        open_positions: Dict of open position metadata.
        df: Current OHLCV DataFrame for ATR ratio.

    Returns:
        Portfolio state dict expected by RiskAnalystAgent.evaluate().
    """
    try:
        balance_info = exchange.get_balance("USDT")
        current_balance = float(balance_info.get("free", 0.0))
    except Exception:
        current_balance = initial_balance

    daily_pnl_pct = (
        (current_balance - start_of_day_balance) / start_of_day_balance
        if start_of_day_balance > 0 else 0.0
    )
    weekly_pnl_pct = (
        (current_balance - start_of_week_balance) / start_of_week_balance
        if start_of_week_balance > 0 else 0.0
    )

    # Compute ATR ratio for volatility spike detection
    atr_ratio = 1.0
    if df is not None and len(df) >= 20 and "ATRr_14" in df.columns:
        recent_atr = float(df["ATRr_14"].iloc[-1])
        avg_atr = float(df["ATRr_14"].rolling(90).mean().iloc[-1])
        if avg_atr > 0:
            atr_ratio = recent_atr / avg_atr

    peak = max(portfolio_peak, current_balance)

    return {
        "balance": current_balance,
        "daily_pnl_pct": daily_pnl_pct,
        "weekly_pnl_pct": weekly_pnl_pct,
        "portfolio_peak": peak,
        "open_positions": open_positions,
        "atr_ratio": atr_ratio,
    }


def execute_trade(
    exchange: Exchange,
    notifier: Notifier,
    risk_decision: RiskDecision,
    symbol: str,
    side: str,
    current_price: float,
    balance: float,
    dry_run: bool = False,
) -> Optional[float]:
    """Execute a trade ONLY after a valid RiskDecision approval.

    CRITICAL: This function MUST NOT be called without a verified RiskDecision
    with is_approved() == True. The caller (run_bot) enforces this invariant.

    Args:
        exchange: Exchange wrapper.
        notifier: Telegram notifier.
        risk_decision: Approved RiskDecision from risk-analyst.
        symbol: Trading pair.
        side: "buy" or "sell".
        current_price: Current market price.
        balance: Available portfolio balance.
        dry_run: If True, log the trade but don't send real orders.

    Returns:
        Trade amount executed, or None on failure.
    """
    # Safety check — belt and suspenders
    if not risk_decision.is_approved():
        logger.critical(
            "POLICY VIOLATION: execute_trade called without approved RiskDecision "
            "for signal_id=%s — BLOCKED",
            risk_decision.signal_id
        )
        return None

    trade_amount = (balance * risk_decision.adjusted_size_pct) / current_price
    if trade_amount <= 0:
        logger.warning("Computed trade_amount is zero or negative — skipping")
        return None

    msg = (
        f"[{symbol}] {side.upper()} {trade_amount:.6f} @ {current_price:.2f} | "
        f"size={risk_decision.adjusted_size_pct:.3f} | "
        f"VaR={risk_decision.var_95:.4f} | "
        f"signal_id={risk_decision.signal_id}"
    )

    if dry_run:
        logger.info("[DRY RUN] %s", msg)
        notifier.send_message(f"[DRY RUN] {msg}")
        TRADES.labels(symbol, side, str(round(current_price, 2)), str(round(trade_amount, 6))).set(1)
        return trade_amount

    try:
        logger.info("Placing order: %s", msg)
        order = exchange.create_order(symbol, "market", side, trade_amount)
        if order is None:
            logger.error("[%s] Exchange returned None for order — may not have executed", symbol)
            notifier.send_message(f"[WARNING] Order may not have executed for {symbol}: exchange returned None")
            return None
        order_id = order.get("id", "unknown")
        order_status = order.get("status", "unknown")
        logger.info("[%s] Order placed: id=%s status=%s", symbol, order_id, order_status)
        notifier.send_message(
            f"{msg} | order_id={order_id} status={order_status}"
        )
        TRADES.labels(symbol, side, str(round(current_price, 2)), str(round(trade_amount, 6))).set(1)
        return trade_amount
    except Exception as exc:
        logger.error("Trade execution failed for %s: %s", symbol, exc)
        notifier.send_message(f"[ERROR] Trade execution failed for {symbol}: {exc}")
        return None


def close_position_order(
    exchange: Exchange,
    notifier: Notifier,
    symbol: str,
    side: str,
    amount: float,
    current_price: float,
    pnl: float,
    dry_run: bool = False,
) -> bool:
    """Execute a position-closing order on the exchange.

    This is called when trailing stop or take-profit triggers. It places the
    opposing order to close the existing position.

    Args:
        exchange: Exchange wrapper.
        notifier: Telegram notifier.
        symbol: Trading pair.
        side: "sell" to close a long, "buy" to close a short.
        amount: Number of units to close.
        current_price: Current market price.
        pnl: Realized PnL percentage for logging.
        dry_run: If True, log but don't send real orders.

    Returns:
        True if close order was placed successfully (or dry run), False on error.
    """
    msg = (
        f"[{symbol}] CLOSE {side.upper()} {amount:.6f} @ {current_price:.2f} | "
        f"PnL={pnl*100:.2f}%"
    )

    if dry_run:
        logger.info("[DRY RUN] %s", msg)
        notifier.send_message(f"[DRY RUN] {msg}")
        return True

    if amount <= 0:
        logger.warning("[%s] Close order amount is zero — skipping", symbol)
        return False

    try:
        order = exchange.create_order(symbol, "market", side, amount)
        if order is None:
            logger.error("[%s] Close order returned None — position may still be open", symbol)
            notifier.send_message(f"[WARNING] Close order may not have executed for {symbol}")
            return False
        logger.info("[%s] Close order placed: id=%s status=%s", symbol, order.get("id"), order.get("status"))
        notifier.send_message(msg)
        return True
    except Exception as exc:
        logger.error("[%s] Close order failed: %s", symbol, exc)
        notifier.send_message(f"[ERROR] Close order failed for {symbol}: {exc}")
        return False


def run_bot(config: Dict[str, Any]) -> None:
    """Main trading loop with ruflo multi-agent orchestration.

    Initializes all components, then runs the per-symbol trading loop.
    Each iteration runs the full agent pipeline:
        1. Fetch OHLCV data
        2. Run pipeline (market-analyst → strategist → risk-analyst)
        3. Execute trade ONLY if RiskDecision approves

    Args:
        config: Full bot configuration dictionary.
    """
    # ---- Initialize components ----
    ai_engine = AIEngine(config)
    exchange = Exchange(config)
    notifier = Notifier(config)
    orchestrator = TradingOrchestrator(config)

    symbols = config["data"]["symbols"]
    dry_run = config.get("dry_run", True)
    # Column names match _add_indicators() output (pure pandas/numpy, period=20 BB)
    features = [
        "close", "volume", "volatility",
        "MACD_12_26_9", "RSI_14",
        "BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0",
        "ATRr_14", "OBV"
    ]

    # ---- Per-symbol state ----
    positions: Dict[str, Optional[str]] = {s: None for s in symbols}
    entry_prices: Dict[str, float] = {s: 0.0 for s in symbols}
    highest_prices: Dict[str, float] = {s: 0.0 for s in symbols}
    trade_amounts: Dict[str, float] = {s: 0.0 for s in symbols}

    # ---- Portfolio-level state ----
    try:
        initial_balance = float(exchange.get_balance("USDT").get("free", 0.0))
    except Exception:
        initial_balance = 0.0

    start_of_day_balance = initial_balance
    start_of_week_balance = initial_balance
    portfolio_peak = initial_balance
    # Daily and weekly balance reset tracking (for circuit breaker accuracy)
    last_reset_day = datetime.now(timezone.utc).date()
    last_reset_week = datetime.now(timezone.utc).isocalendar()[:2]  # (year, week_num)

    # Per-symbol close price cache for pairwise correlation computation
    symbol_close_cache: Dict[str, list] = {s: [] for s in symbols}
    open_positions_meta: Dict[str, Any] = {}

    # ---- Dashboard state tracking ----
    current_prices: Dict[str, float] = {s: 0.0 for s in symbols}
    regime_cache: Dict[str, Any] = {}       # symbol → {regime, confidence, indicators}
    recent_signals_cache: list = []          # last 20 SignalProposals (as dicts)
    recent_verdicts_cache: list = []         # last 20 RegimeVerdicts (as dicts)
    last_risk_metrics: Dict[str, float] = {}
    pipeline_stats: Dict[str, int] = {
        "cycles": 0, "approvals": 0, "rejections": 0, "cb_events": 0
    }

    logger.info(
        "Bot started | symbols=%s | dry_run=%s | initial_balance=%.2f USDT",
        symbols, dry_run, initial_balance
    )

    while st.session_state.get("bot_running", False):
        try:
            # ---- Update global metrics ----
            try:
                balance_info = exchange.get_balance("USDT")
                current_balance = float(balance_info.get("free", 0.0))
            except Exception:
                current_balance = initial_balance

            portfolio_peak = max(portfolio_peak, current_balance)
            BALANCE.set(current_balance)
            TOTAL_PROFIT_LOSS.set(current_balance - initial_balance)
            OPEN_POSITIONS.set(sum(1 for p in positions.values() if p is not None))
            PIPELINE_CYCLES.inc()
            pipeline_stats["cycles"] += 1

            # ---- Daily/Weekly balance reset for circuit breaker accuracy ----
            now_utc = datetime.now(timezone.utc)
            today = now_utc.date()
            this_week = now_utc.isocalendar()[:2]
            if today != last_reset_day:
                start_of_day_balance = current_balance
                last_reset_day = today
                logger.info("Daily balance reset: %.2f USDT", start_of_day_balance)
            if this_week != last_reset_week:
                start_of_week_balance = current_balance
                last_reset_week = this_week
                logger.info("Weekly balance reset: %.2f USDT", start_of_week_balance)

            # ---- Per-symbol trading loop ----
            for symbol in symbols:
                try:
                    # Fetch and prepare OHLCV data
                    ohlcv = exchange.exchange.fetch_ohlcv(
                        symbol,
                        config["data"]["timeframe"],
                        limit=config["data"]["lookback"]
                    )
                    df = pd.DataFrame(
                        ohlcv,
                        columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )
                    df["returns"] = df["close"].pct_change()
                    df["volatility"] = df["returns"].rolling(20).std()
                    # Compute indicators with pure pandas/numpy (no pandas-ta dependency)
                    _add_indicators(df)
                    df.dropna(inplace=True)
                    # Cache recent close prices for cross-symbol correlation
                    symbol_close_cache[symbol] = df["close"].values[-30:].tolist()

                    if len(df) < 30:
                        logger.warning("[%s] Insufficient data after indicators", symbol)
                        continue

                    current_price = float(df["close"].iloc[-1])
                    current_prices[symbol] = current_price   # track for dashboard

                    # ---- Manage existing position exits ----
                    if positions[symbol] is not None:
                        pnl = (
                            (current_price - entry_prices[symbol]) / entry_prices[symbol]
                            if entry_prices[symbol] > 0 else 0.0
                        )
                        take_profit_pct = config["trading"].get("take_profit_percentage", 0.04)
                        if config["trading"].get("dynamic_take_profit", {}).get("enabled"):
                            vol = float(df["volatility"].iloc[-1])
                            mult = config["trading"]["dynamic_take_profit"].get(
                                "volatility_multiplier", 2
                            )
                            take_profit_pct = vol * mult

                        trailing_pct = config["trading"].get(
                            "trailing_stop", {}
                        ).get("percentage", 0.01)

                        if positions[symbol] == "buy":
                            highest_prices[symbol] = max(
                                highest_prices[symbol], current_price
                            )
                            trailing_stop = highest_prices[symbol] * (1 - trailing_pct)
                            if current_price < trailing_stop or pnl > take_profit_pct:
                                logger.info(
                                    "[%s] Closing LONG position | PnL=%.4f", symbol, pnl
                                )
                                closed = close_position_order(
                                    exchange=exchange,
                                    notifier=notifier,
                                    symbol=symbol,
                                    side="sell",
                                    amount=trade_amounts[symbol],
                                    current_price=current_price,
                                    pnl=pnl,
                                    dry_run=dry_run,
                                )
                                if closed or dry_run:
                                    TRADES.labels(
                                        symbol, "sell",
                                        str(round(current_price, 2)),
                                        str(round(trade_amounts[symbol], 6))
                                    ).set(1)
                                    positions[symbol] = None
                                    open_positions_meta.pop(symbol, None)

                        elif positions[symbol] == "sell":
                            highest_prices[symbol] = min(
                                highest_prices[symbol], current_price
                            )
                            trailing_stop = highest_prices[symbol] * (1 + trailing_pct)
                            if current_price > trailing_stop or pnl < -take_profit_pct:
                                logger.info(
                                    "[%s] Closing SHORT position | PnL=%.4f", symbol, pnl
                                )
                                closed = close_position_order(
                                    exchange=exchange,
                                    notifier=notifier,
                                    symbol=symbol,
                                    side="buy",
                                    amount=trade_amounts[symbol],
                                    current_price=current_price,
                                    pnl=pnl,
                                    dry_run=dry_run,
                                )
                                if closed or dry_run:
                                    TRADES.labels(
                                        symbol, "buy",
                                        str(round(current_price, 2)),
                                        str(round(trade_amounts[symbol], 6))
                                    ).set(1)
                                    positions[symbol] = None
                                    open_positions_meta.pop(symbol, None)

                    # ---- Run multi-agent pipeline for new entries ----
                    if positions[symbol] is None:
                        portfolio_state = build_portfolio_state(
                            exchange=exchange,
                            initial_balance=initial_balance,
                            start_of_day_balance=start_of_day_balance,
                            start_of_week_balance=start_of_week_balance,
                            portfolio_peak=portfolio_peak,
                            open_positions=open_positions_meta,
                            df=df,
                        )

                        risk_decision = orchestrator.run_pipeline(
                            df=df,
                            symbols=[symbol],
                            ai_engine=ai_engine,
                            portfolio_state=portfolio_state,
                        )

                        # ---- Update Prometheus regime metric + dashboard cache ----
                        last_regime = orchestrator._last_regime
                        if last_regime is not None:
                            regime_code = REGIME_CODES.get(last_regime.regime_type, 0)
                            REGIME_GAUGE.labels(symbol=symbol, regime=last_regime.regime_type).set(
                                regime_code
                            )
                            regime_cache[symbol] = {
                                "regime": last_regime.regime_type,
                                "confidence": last_regime.confidence,
                                "indicators": last_regime.indicator_values,
                            }
                            # Cache recent regime verdict for dashboard
                            try:
                                v_dict = {
                                    "regime_type": last_regime.regime_type,
                                    "confidence": last_regime.confidence,
                                    "symbols": last_regime.symbols,
                                    "timestamp": last_regime.timestamp,
                                    "verdict_id": last_regime.verdict_id,
                                    "indicator_values": last_regime.indicator_values,
                                }
                                recent_verdicts_cache.append(v_dict)
                                if len(recent_verdicts_cache) > 50:
                                    recent_verdicts_cache.pop(0)
                            except Exception:
                                pass

                        # Cache last signal for dashboard
                        last_signal = orchestrator._last_signal
                        if last_signal is not None:
                            try:
                                s_dict = {
                                    "symbol": last_signal.symbol,
                                    "side": last_signal.side,
                                    "size_pct": last_signal.size_pct,
                                    "confidence": last_signal.confidence,
                                    "strategy_name": last_signal.strategy_name,
                                    "anomaly_type": last_signal.anomaly_type,
                                    "anomaly_score": last_signal.anomaly_score,
                                    "regime_verdict_id": last_signal.regime_verdict_id,
                                    "signal_id": last_signal.signal_id,
                                    "timestamp": last_signal.timestamp,
                                }
                                recent_signals_cache.append(s_dict)
                                if len(recent_signals_cache) > 50:
                                    recent_signals_cache.pop(0)
                            except Exception:
                                pass

                        if risk_decision is None:
                            continue

                        if risk_decision.circuit_breaker_triggered:
                            CIRCUIT_BREAKER_EVENTS.labels(
                                reason=risk_decision.circuit_breaker_reason or "UNKNOWN"
                            ).inc()
                            pipeline_stats["cb_events"] += 1
                            logger.warning(
                                "[%s] Circuit breaker: %s",
                                symbol, risk_decision.circuit_breaker_reason
                            )

                        # Capture risk metrics for dashboard
                        last_risk_metrics.update({
                            "var_95": risk_decision.var_95,
                            "cvar_95": risk_decision.cvar_95,
                            "sharpe_ratio": risk_decision.sharpe_ratio,
                            "drawdown_pct": getattr(orchestrator._risk_agent, "_state", {})
                                .get("current_drawdown_pct", 0.0),
                        })

                        # ---- Execute trade ONLY if approved ----
                        if risk_decision.is_approved():
                            PIPELINE_APPROVALS.inc()
                            pipeline_stats["approvals"] += 1
                            signal = orchestrator._last_signal
                            side = signal.side if signal else "buy"

                            amount = execute_trade(
                                exchange=exchange,
                                notifier=notifier,
                                risk_decision=risk_decision,
                                symbol=symbol,
                                side=side,
                                current_price=current_price,
                                balance=portfolio_state["balance"],
                                dry_run=dry_run,
                            )

                            if amount is not None:
                                positions[symbol] = side
                                entry_prices[symbol] = current_price
                                highest_prices[symbol] = current_price
                                trade_amounts[symbol] = amount
                                open_positions_meta[symbol] = {
                                    "side": side,
                                    "concentration_pct": risk_decision.adjusted_size_pct,
                                }
                        else:
                            PIPELINE_REJECTIONS.inc()
                            pipeline_stats["rejections"] += 1

                except Exception as exc:
                    logger.error(
                        "Error in trading loop for %s: %s", symbol, exc
                    )
                    notifier.send_message(
                        f"[ERROR] Trading loop error for {symbol}: {exc}"
                    )

            # ---- Compute pairwise correlation for circuit breaker ----
            if len(symbols) > 1:
                try:
                    import itertools
                    max_corr = 0.0
                    sym_pairs = list(itertools.combinations(symbols, 2))
                    for s1, s2 in sym_pairs:
                        c1 = symbol_close_cache.get(s1, [])
                        c2 = symbol_close_cache.get(s2, [])
                        min_len = min(len(c1), len(c2))
                        if min_len > 5:
                            import numpy as _np
                            r1 = _np.diff(c1[-min_len:]) / _np.array(c1[-min_len:-1])
                            r2 = _np.diff(c2[-min_len:]) / _np.array(c2[-min_len:-1])
                            if _np.std(r1) > 0 and _np.std(r2) > 0:
                                corr = abs(float(_np.corrcoef(r1, r2)[0, 1]))
                                if not _np.isnan(corr):
                                    max_corr = max(max_corr, corr)
                    orchestrator.update_portfolio_correlation(max_corr)
                except Exception as _corr_exc:
                    logger.debug("Correlation computation failed: %s", _corr_exc)

            # ---- Write live state for Streamlit dashboard ----
            try:
                risk_agent_state = getattr(
                    getattr(orchestrator, "_risk_agent", None), "_state", {}
                ) or {}
                _write_bot_state(
                    running=True,
                    balance=current_balance,
                    initial_balance=initial_balance,
                    daily_pnl_pct=(current_balance - start_of_day_balance) / start_of_day_balance
                        if start_of_day_balance > 0 else 0.0,
                    weekly_pnl_pct=(current_balance - start_of_week_balance) / start_of_week_balance
                        if start_of_week_balance > 0 else 0.0,
                    positions=positions,
                    entry_prices=entry_prices,
                    current_prices=current_prices,
                    regimes=regime_cache,
                    pipeline_stats=dict(pipeline_stats),
                    risk_state=risk_agent_state,
                    last_risk_metrics=dict(last_risk_metrics),
                    recent_signals=list(recent_signals_cache),
                    recent_verdicts=list(recent_verdicts_cache),
                )
            except Exception as _state_exc:
                logger.debug("State write failed: %s", _state_exc)

            time.sleep(60)

        except Exception as exc:
            logger.error("Main bot loop error: %s", exc)
            notifier.send_message(f"[ERROR] Main bot loop: {exc}")
            time.sleep(60)
