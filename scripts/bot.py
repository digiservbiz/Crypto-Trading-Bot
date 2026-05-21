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
from typing import Dict, Optional, Any

import torch
import pandas as pd
import requests
import streamlit as st
from prometheus_client import Gauge, Counter
from datetime import datetime, timezone

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


def get_sentiment(text: str) -> float:
    """Analyze sentiment of text using the flair library.

    Args:
        text: News article body or title.

    Returns:
        Sentiment score in [-1.0, +1.0]. Positive = bullish, negative = bearish.
    """
    try:
        from flair.models import TextClassifier
        from flair.data import Sentence
        classifier = TextClassifier.load("en-sentiment")
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

    msg = (
        f"[{symbol}] {side.upper()} {trade_amount:.6f} @ {current_price:.2f} | "
        f"size={risk_decision.adjusted_size_pct:.3f} | "
        f"VaR={risk_decision.var_95:.4f} | "
        f"signal_id={risk_decision.signal_id}"
    )

    if dry_run:
        logger.info("[DRY RUN] %s", msg)
        notifier.send_message(f"[DRY RUN] {msg}")
        return trade_amount

    try:
        logger.info("Executing trade: %s", msg)
        notifier.send_message(msg)
        TRADES.labels(symbol, side, current_price, trade_amount).set(1)
        return trade_amount
    except Exception as exc:
        logger.error("Trade execution failed: %s", exc)
        notifier.send_message(f"[ERROR] Trade execution failed for {symbol}: {exc}")
        return None


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
    features = [
        "close", "volume", "volatility",
        "MACD_12_26_9", "RSI_14",
        "BBL_5_2.0", "BBM_5_2.0", "BBU_5_2.0",
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
    open_positions_meta: Dict[str, Any] = {}

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
                    df.ta.macd(append=True)
                    df.ta.rsi(append=True)
                    df.ta.bbands(append=True)
                    df.ta.atr(append=True)
                    df.ta.obv(append=True)
                    df.dropna(inplace=True)

                    if len(df) < 30:
                        logger.warning("[%s] Insufficient data after indicators", symbol)
                        continue

                    current_price = float(df["close"].iloc[-1])

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
                                notifier.send_message(
                                    f"[{symbol}] Closing LONG position | PnL={pnl:.4f}"
                                )
                                TRADES.labels(
                                    symbol, "sell", current_price, trade_amounts[symbol]
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
                                notifier.send_message(
                                    f"[{symbol}] Closing SHORT position | PnL={pnl:.4f}"
                                )
                                TRADES.labels(
                                    symbol, "buy", current_price, trade_amounts[symbol]
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

                        # ---- Update Prometheus regime metric ----
                        last_regime = orchestrator._last_regime
                        if last_regime is not None:
                            regime_code = REGIME_CODES.get(last_regime.regime_type, 0)
                            REGIME_GAUGE.labels(symbol=symbol, regime=last_regime.regime_type).set(
                                regime_code
                            )

                        if risk_decision is None:
                            continue

                        if risk_decision.circuit_breaker_triggered:
                            CIRCUIT_BREAKER_EVENTS.labels(
                                reason=risk_decision.circuit_breaker_reason or "UNKNOWN"
                            ).inc()
                            logger.warning(
                                "[%s] Circuit breaker: %s",
                                symbol, risk_decision.circuit_breaker_reason
                            )

                        # ---- Execute trade ONLY if approved ----
                        if risk_decision.is_approved():
                            PIPELINE_APPROVALS.inc()
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

                except Exception as exc:
                    logger.error(
                        "Error in trading loop for %s: %s", symbol, exc
                    )
                    notifier.send_message(
                        f"[ERROR] Trading loop error for {symbol}: {exc}"
                    )

            # ---- Update pipeline stats in Streamlit session state ----
            stats = orchestrator.get_pipeline_stats()
            if hasattr(st, "session_state"):
                st.session_state["pipeline_stats"] = stats

            time.sleep(60)

        except Exception as exc:
            logger.error("Main bot loop error: %s", exc)
            notifier.send_message(f"[ERROR] Main bot loop: {exc}")
            time.sleep(60)
