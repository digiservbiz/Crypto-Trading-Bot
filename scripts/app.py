"""Full trading dashboard — Streamlit UI for the Crypto Trading Bot.

Tabs:
    📊 Overview  — balance, P&L, regime cards, circuit breakers, pipeline stats
    💼 Positions — open positions table with live P&L
    📡 Signals   — recent signal proposals and regime verdicts
    📈 Backtest  — run backtests, view quality gates and equity curve
    🧠 Training  — train LSTM / Transformer models
    ⚙️  Config    — edit key configuration values

State is read from data/state/bot-state.json written by the bot on every cycle.
"""

import json
import os
import subprocess
import sys
import threading
import time
import yaml
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
from prometheus_client import start_http_server

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crypto Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constants ────────────────────────────────────────────────────────────────
BOT_STATE_PATH = "data/state/bot-state.json"
BOT_STOP_FILE  = "data/state/bot.stop"
RISK_MEMORY_PATH = "data/memory/trading-risk.json"
SIGNALS_MEMORY_PATH = "data/memory/trading-signals.json"
ANALYSIS_MEMORY_PATH = "data/memory/trading-analysis.json"
BACKTESTS_MEMORY_PATH = "data/memory/trading-backtests.json"
CONFIG_PATH = "config.yaml"

REGIME_COLORS = {
    "bull-trending":  "#00c853",
    "bear-trending":  "#d50000",
    "ranging":        "#2979ff",
    "high-volatility":"#ff6d00",
    "low-volatility": "#aa00ff",
    "transitioning":  "#90a4ae",
    "unknown":        "#546e7a",
}

REGIME_ICONS = {
    "bull-trending":  "🟢",
    "bear-trending":  "🔴",
    "ranging":        "🔵",
    "high-volatility":"🟠",
    "low-volatility": "🟣",
    "transitioning":  "⚪",
    "unknown":        "⚫",
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_json(path: str, default: Any = None) -> Any:
    """Load a JSON file; return default on any error."""
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return default if default is not None else {}


def _load_config() -> dict:
    try:
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _bot_state() -> dict:
    """Return live bot state or a safe empty stub."""
    default = {
        "running": False,
        "last_update": None,
        "balance": 0.0,
        "initial_balance": 0.0,
        "daily_pnl_pct": 0.0,
        "weekly_pnl_pct": 0.0,
        "positions": {},
        "regimes": {},
        "pipeline": {"cycles": 0, "approvals": 0, "rejections": 0, "cb_events": 0},
        "circuit_breakers": {
            "daily_loss_halt": False,
            "weekly_reduction": False,
            "max_drawdown_halt": False,
            "correlation_spike": False,
            "vix_spike": False,
            "concentration_limit": False,
        },
        "recent_signals": [],
        "recent_verdicts": [],
    }
    state = _load_json(BOT_STATE_PATH, default)
    # Merge missing keys from default
    for k, v in default.items():
        state.setdefault(k, v)
    return state


def _bot_actually_running() -> bool:
    """True if the bot process is alive and recently updated its state file.

    Uses two signals:
    1. bot-state.json says running=True
    2. last_update is within the last 150 seconds (2.5 × 60s cycle)

    This is reliable across browser refreshes because it reads from disk,
    not from Streamlit session state.
    """
    state = _bot_state()
    if not state.get("running", False):
        return False
    last = state.get("last_update")
    if last is None:
        return False
    return (time.time() - float(last)) < 150


def _start_bot(config_path: str, dry_run: bool) -> None:
    """Spawn the bot as a standalone subprocess."""
    # Remove any leftover stop sentinel
    if os.path.exists(BOT_STOP_FILE):
        os.remove(BOT_STOP_FILE)

    cmd = [sys.executable, "-m", "scripts.bot", "--config", config_path]
    if dry_run:
        cmd.append("--dry-run")

    proc = subprocess.Popen(
        cmd,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    # Store PID so dashboard can show it; not needed for control
    st.session_state["bot_pid"] = proc.pid
    st.session_state["bot_running"] = True


def _stop_bot() -> None:
    """Write the stop sentinel — the bot will exit after its current cycle."""
    os.makedirs(os.path.dirname(BOT_STOP_FILE), exist_ok=True)
    with open(BOT_STOP_FILE, "w") as f:
        f.write(str(time.time()))
    st.session_state["bot_running"] = False


def _pnl_color(pnl: float) -> str:
    if pnl > 0:
        return "normal"
    if pnl < 0:
        return "inverse"
    return "off"


def _fmt_pct(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.2f}%"


def _fmt_usd(v: float) -> str:
    return f"${v:,.2f}"


def _ago(ts: Optional[float]) -> str:
    if ts is None:
        return "—"
    delta = time.time() - ts
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta / 60)}m ago"
    return f"{int(delta / 3600)}h ago"


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark metric cards */
[data-testid="metric-container"] {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 10px;
    padding: 12px 16px;
}
/* Regime badge */
.regime-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
    color: white;
    margin: 2px 0;
}
/* Status dot */
.status-dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-right: 6px;
}
.dot-green { background: #00c853; box-shadow: 0 0 6px #00c853; }
.dot-red   { background: #d50000; }
/* CB row */
.cb-row { display: flex; align-items: center; gap: 8px; padding: 4px 0; font-size: 0.9rem; }
.cb-ok  { color: #00c853; font-weight: 700; }
.cb-err { color: #ff5252; font-weight: 700; }
/* Tab headers */
div[data-testid="stHorizontalBlock"] { gap: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
state = _bot_state()
running = _bot_actually_running()   # truth from state file, survives browser refresh
last_update = state.get("last_update")

col_title, col_status, col_refresh = st.columns([4, 2, 1])
with col_title:
    st.title("📈 Crypto Trading Bot")
with col_status:
    dot_cls = "dot-green" if running else "dot-red"
    status_text = "LIVE" if running else "STOPPED"
    st.markdown(
        f'<div style="margin-top:28px">'
        f'<span class="status-dot {dot_cls}"></span>'
        f'<b style="font-size:1rem">{status_text}</b>'
        f' &nbsp;·&nbsp; <span style="color:#888;font-size:0.8rem">updated {_ago(last_update)}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
with col_refresh:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_positions, tab_signals, tab_backtest, tab_training, tab_config = st.tabs([
    "📊 Overview",
    "💼 Positions",
    "📡 Signals",
    "📈 Backtest",
    "🧠 Training",
    "⚙️ Config",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:

    # ── KPI Metrics Row ──────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)

    balance = state.get("balance", 0.0)
    initial = state.get("initial_balance", balance or 1.0)
    daily_pnl = state.get("daily_pnl_pct", 0.0)
    weekly_pnl = state.get("weekly_pnl_pct", 0.0)
    total_pnl = (balance - initial) / initial if initial > 0 else 0.0
    n_positions = sum(1 for p in state.get("positions", {}).values() if p)
    pipeline = state.get("pipeline", {})
    cycles = pipeline.get("cycles", 0)
    approvals = pipeline.get("approvals", 0)
    approval_rate = (approvals / cycles * 100) if cycles > 0 else 0.0

    c1.metric("💰 Balance", _fmt_usd(balance),
              delta=_fmt_usd(balance - initial) if initial > 0 else None)
    c2.metric("📅 Daily P&L", _fmt_pct(daily_pnl),
              delta=_fmt_pct(daily_pnl), delta_color=_pnl_color(daily_pnl))
    c3.metric("📆 Weekly P&L", _fmt_pct(weekly_pnl),
              delta=_fmt_pct(weekly_pnl), delta_color=_pnl_color(weekly_pnl))
    c4.metric("🔁 Pipeline Cycles", f"{cycles:,}",
              delta=f"{approval_rate:.1f}% approved")
    c5.metric("📂 Open Positions", n_positions)

    st.markdown("---")

    # ── Market Regimes ───────────────────────────────────────────────────────
    left, right = st.columns([3, 2])

    with left:
        st.subheader("🌐 Market Regimes")
        regimes = state.get("regimes", {})
        config_data = _load_config()
        symbols = config_data.get("data", {}).get("symbols", ["BTC/USDT", "ETH/USDT"])

        if not regimes:
            st.info("No regime data yet — start the bot to begin analysis.")
        else:
            for sym in symbols:
                info = regimes.get(sym, {})
                regime = info.get("regime", "unknown")
                conf = info.get("confidence", 0.0)
                color = REGIME_COLORS.get(regime, "#546e7a")
                icon = REGIME_ICONS.get(regime, "⚫")
                adx = info.get("adx", None)
                rsi = info.get("rsi", None)

                sc1, sc2, sc3 = st.columns([2, 3, 2])
                with sc1:
                    st.markdown(f"**{sym}**")
                with sc2:
                    st.markdown(
                        f'<span class="regime-badge" style="background:{color}">'
                        f'{icon} {regime.upper()}</span>',
                        unsafe_allow_html=True
                    )
                with sc3:
                    st.progress(conf, text=f"Conf: {conf:.0%}")

                if adx is not None or rsi is not None:
                    ind_cols = st.columns(4)
                    indicators = info.get("indicators", {})
                    for i, (name, key) in enumerate([
                        ("ADX", "adx"), ("RSI", "rsi"),
                        ("MACD", "macd_hist"), ("ATR×", "atr_ratio")
                    ]):
                        val = indicators.get(key, info.get(key))
                        if val is not None:
                            ind_cols[i].metric(name, f"{val:.2f}")
                st.markdown("")

    with right:
        st.subheader("🔒 Circuit Breakers")
        cbs = state.get("circuit_breakers", {})
        risk_state = _load_json(RISK_MEMORY_PATH, {})
        cb_log = risk_state.get("circuit_breaker_log", [])

        cb_definitions = [
            ("daily_loss_halt",   "Daily Loss Halt",     "≥3% daily loss"),
            ("weekly_reduction",  "Weekly Reduction",    "≥5% weekly loss"),
            ("max_drawdown_halt", "Max Drawdown Halt",   "≥15% from peak (manual reset)"),
            ("correlation_spike", "Correlation Spike",   ">0.85 pairwise correlation"),
            ("vix_spike",         "Volatility Spike",    ">2× 90-day ATR average"),
            ("concentration_limit","Concentration Limit","Single asset >10% portfolio"),
        ]

        for key, label, condition in cb_definitions:
            triggered = cbs.get(key, False)
            cls = "cb-err" if triggered else "cb-ok"
            icon_cb = "🚨" if triggered else "✅"
            st.markdown(
                f'<div class="cb-row">'
                f'<span class="{cls}">{icon_cb}</span>'
                f'<b>{label}</b>'
                f'<span style="color:#888;font-size:0.78rem"> — {condition}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        if cb_log:
            st.markdown("**Recent triggers:**")
            for entry in cb_log[-3:]:
                ts = datetime.fromtimestamp(entry.get("timestamp", 0)).strftime("%H:%M:%S")
                st.markdown(
                    f'<div style="color:#ff5252;font-size:0.8rem">⚡ {ts} — {entry.get("reason","")}</div>',
                    unsafe_allow_html=True
                )

    st.markdown("---")

    # ── Pipeline Stats ───────────────────────────────────────────────────────
    st.subheader("🤖 Agent Pipeline Stats")
    p1, p2, p3, p4, p5 = st.columns(5)
    cb_events = pipeline.get("cb_events", 0)
    rejections = pipeline.get("rejections", 0)

    p1.metric("Total Cycles", f"{cycles:,}")
    p2.metric("Approvals", f"{approvals:,}",
              delta=f"{approval_rate:.1f}%")
    p3.metric("Rejections", f"{rejections:,}")
    p4.metric("Circuit Breaker Events", f"{cb_events:,}")
    rejection_rate = (rejections / cycles * 100) if cycles > 0 else 0.0
    p5.metric("Rejection Rate", f"{rejection_rate:.1f}%")

    # Approval vs rejection bar
    if cycles > 0:
        chart_data = pd.DataFrame({
            "Category": ["✅ Approved", "❌ Rejected", "⚡ CB Events"],
            "Count": [approvals, rejections, cb_events]
        }).set_index("Category")
        st.bar_chart(chart_data)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — POSITIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_positions:
    st.subheader("💼 Open Positions")

    positions_data = state.get("positions", {})
    open_pos = {k: v for k, v in positions_data.items() if v and isinstance(v, dict)}

    if not open_pos:
        st.info("No open positions currently.")
    else:
        rows = []
        for sym, info in open_pos.items():
            side = info.get("side", "—")
            entry = info.get("entry_price", 0.0)
            current = info.get("current_price", entry)
            pnl_pct = info.get("pnl_pct", 0.0)
            size_pct = info.get("size_pct", 0.0)
            pnl_usd = balance * size_pct * pnl_pct if balance > 0 else 0.0

            rows.append({
                "Symbol": sym,
                "Side": "🟢 BUY" if side == "buy" else "🔴 SELL",
                "Entry Price": f"${entry:,.2f}",
                "Current Price": f"${current:,.2f}",
                "P&L %": f"{'+'if pnl_pct>=0 else ''}{pnl_pct*100:.2f}%",
                "P&L USD": f"{'+'if pnl_usd>=0 else ''}${pnl_usd:,.2f}",
                "Size": f"{size_pct*100:.1f}%",
            })

        df_pos = pd.DataFrame(rows)
        st.dataframe(df_pos, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── P&L History chart ────────────────────────────────────────────────────
    st.subheader("📉 P&L History")
    pnl_history = state.get("pnl_history", [])

    if len(pnl_history) < 2:
        st.info("P&L history builds up as the bot runs cycles. Check back after a few minutes.")
    else:
        df_pnl = pd.DataFrame(pnl_history)
        df_pnl["time"] = pd.to_datetime(df_pnl["timestamp"], unit="s")
        df_pnl = df_pnl.set_index("time")
        st.line_chart(df_pnl[["balance"]], height=250)

    # ── Risk Metrics (latest from risk-analyst) ───────────────────────────────
    st.subheader("📐 Risk Metrics")
    risk_metrics = state.get("last_risk_metrics", {})
    if risk_metrics:
        rm1, rm2, rm3, rm4 = st.columns(4)
        rm1.metric("VaR (95%)", f"{risk_metrics.get('var_95', 0)*100:.3f}%",
                   help="Value at Risk — max expected loss at 95% confidence")
        rm2.metric("CVaR (95%)", f"{risk_metrics.get('cvar_95', 0)*100:.3f}%",
                   help="Conditional VaR — expected loss beyond VaR threshold")
        rm3.metric("Sharpe Ratio", f"{risk_metrics.get('sharpe_ratio', 0):.3f}",
                   help="Rolling 30-day annualized Sharpe")
        rm4.metric("Drawdown", f"{risk_metrics.get('drawdown_pct', 0)*100:.2f}%")
    else:
        st.info("Risk metrics appear here after the first trade signal is evaluated.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_signals:
    st.subheader("📡 Recent Signal Proposals")

    recent_signals = state.get("recent_signals", [])
    signals_mem = _load_json(SIGNALS_MEMORY_PATH, {"signals": []})
    all_signals = recent_signals + signals_mem.get("signals", [])
    all_signals = sorted(all_signals, key=lambda x: x.get("timestamp", 0), reverse=True)[:30]

    if not all_signals:
        st.info("No signals generated yet.")
    else:
        sig_rows = []
        for sig in all_signals:
            ts = datetime.fromtimestamp(sig.get("timestamp", 0)).strftime("%H:%M:%S")
            side = sig.get("side", "—")
            sig_rows.append({
                "Time": ts,
                "Symbol": sig.get("symbol", "—"),
                "Side": "🟢 BUY" if side == "buy" else "🔴 SELL",
                "Strategy": sig.get("strategy_name", "—"),
                "Anomaly": sig.get("anomaly_type", "none"),
                "Z-Score": f"{sig.get('anomaly_score', 0):.2f}",
                "Confidence": f"{sig.get('confidence', 0)*100:.1f}%",
                "Size": f"{sig.get('size_pct', 0)*100:.2f}%",
                "Verdict ID": sig.get("regime_verdict_id", "—"),
            })

        st.dataframe(pd.DataFrame(sig_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🌐 Recent Regime Verdicts")

    recent_verdicts = state.get("recent_verdicts", [])
    analysis_mem = _load_json(ANALYSIS_MEMORY_PATH, {"verdicts": []})
    all_verdicts = recent_verdicts + analysis_mem.get("verdicts", [])
    all_verdicts = sorted(all_verdicts, key=lambda x: x.get("timestamp", 0), reverse=True)[:20]

    if not all_verdicts:
        st.info("No regime verdicts yet.")
    else:
        v_rows = []
        for v in all_verdicts:
            ts = datetime.fromtimestamp(v.get("timestamp", 0)).strftime("%H:%M:%S")
            regime = v.get("regime_type", "unknown")
            color = REGIME_COLORS.get(regime, "#546e7a")
            icon = REGIME_ICONS.get(regime, "⚫")
            inds = v.get("indicator_values", {})
            v_rows.append({
                "Time": ts,
                "Symbols": ", ".join(v.get("symbols", [])),
                "Regime": f"{icon} {regime}",
                "Confidence": f"{v.get('confidence', 0)*100:.1f}%",
                "ADX": f"{inds.get('adx', 0):.1f}",
                "RSI": f"{inds.get('rsi', 0):.1f}",
                "MACD Hist": f"{inds.get('macd_hist', 0):.4f}",
                "ATR×": f"{inds.get('atr_ratio', 1):.2f}",
                "ID": v.get("verdict_id", "—"),
            })
        st.dataframe(pd.DataFrame(v_rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════
with tab_backtest:
    st.subheader("📈 Backtesting")

    config_data = _load_config()
    bt_col1, bt_col2, bt_col3 = st.columns(3)
    with bt_col1:
        bt_symbol = st.selectbox(
            "Symbol",
            config_data.get("data", {}).get("symbols", ["BTC/USDT"]),
            key="bt_symbol"
        )
    with bt_col2:
        bt_strategy = st.selectbox(
            "Strategy",
            ["momentum", "mean-reversion", "adaptive"],
            key="bt_strategy"
        )
    with bt_col3:
        bt_lookback = st.number_input("Lookback bars", value=20, min_value=5, max_value=100)

    if st.button("▶️ Run Backtest", type="primary"):
        with st.spinner(f"Running {bt_strategy} backtest on {bt_symbol}…"):
            try:
                from scripts.backtest import legacy_backtest
                bt_config = dict(config_data)
                results = legacy_backtest(bt_config)
                st.session_state["backtest_results"] = results
                st.success("Backtest complete!")
            except Exception as e:
                st.error(f"Backtest failed: {e}")

    # ── Results ──────────────────────────────────────────────────────────────
    if "backtest_results" in st.session_state:
        r = st.session_state["backtest_results"]
        st.markdown("### Results")

        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
        total_ret = r.get("total_return", 0)
        rc1.metric("Total Return", _fmt_pct(total_ret),
                   delta_color="normal" if total_ret >= 0 else "inverse")
        rc2.metric("Sharpe Ratio", f"{r.get('sharpe_ratio', 0):.3f}")
        rc3.metric("Max Drawdown", f"{r.get('max_drawdown', 0)*100:.2f}%")
        rc4.metric("Win Rate", f"{r.get('win_rate', 0)*100:.1f}%")
        rc5.metric("Profit Factor", f"{r.get('profit_factor', 0):.2f}")

        # Quality gates
        st.markdown("### Quality Gates")
        gates = r.get("quality_gates", {})
        if gates:
            gc = st.columns(6)
            gate_labels = {
                "min_trades": "≥30 Trades",
                "win_rate_consistency": "Win Rate Consistency",
                "monte_carlo_significance": "Monte Carlo (p<0.05)",
                "max_drawdown": "Max DD <15%",
                "profit_factor": "Profit Factor >1.5",
                "sharpe_ratio": "Sharpe >1.0",
            }
            for i, (key, label) in enumerate(gate_labels.items()):
                passed = gates.get(key, False)
                gc[i].metric(label, "✅ PASS" if passed else "❌ FAIL")

        # Equity curve
        curve = r.get("cumulative_returns")
        if curve is not None:
            st.markdown("### Equity Curve")
            if isinstance(curve, list):
                curve = pd.Series(curve)
            st.line_chart(curve, height=300)

    # ── Saved artifacts ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💾 Saved Backtest Artifacts")
    bt_mem = _load_json(BACKTESTS_MEMORY_PATH, {"artifacts": []})
    artifacts = bt_mem.get("artifacts", [])
    if not artifacts:
        st.info("No signed backtest artifacts yet. Run a backtest with a strategy that passes all 6 quality gates.")
    else:
        a_rows = []
        for a in sorted(artifacts, key=lambda x: x.get("timestamp", 0), reverse=True)[:10]:
            gates = a.get("quality_gate_summary", {})
            passed_count = sum(1 for v in gates.values() if v)
            a_rows.append({
                "ID": a.get("artifact_id", "—"),
                "Symbol": a.get("symbol", "—"),
                "Strategy": a.get("strategy_id", "—"),
                "Period": a.get("period", "—"),
                "Return": _fmt_pct(a.get("total_return", 0)),
                "Sharpe": f"{a.get('sharpe_ratio', 0):.2f}",
                "Max DD": f"{a.get('max_drawdown', 0)*100:.2f}%",
                "Win Rate": f"{a.get('win_rate', 0)*100:.1f}%",
                "Gates": f"{passed_count}/6",
                "Signed": "✅" if a.get("passed_quality_gates") else "❌",
            })
        st.dataframe(pd.DataFrame(a_rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
with tab_training:
    st.subheader("🧠 Model Training")

    config_data = _load_config()
    tc1, tc2 = st.columns(2)
    with tc1:
        train_model_type = st.selectbox("Model Type", ["lstm", "transformer"], key="train_model")
    with tc2:
        train_symbol = st.selectbox(
            "Symbol",
            config_data.get("data", {}).get("symbols", ["BTC/USDT"]),
            key="train_symbol"
        )

    st.caption("⚠️ Training requires `torch` and `pytorch-lightning`. Install with: `pip install torch pytorch-lightning`")

    if st.button("🚀 Start Training", type="primary"):
        with st.spinner(f"Training {train_model_type.upper()} on {train_symbol}…"):
            try:
                from scripts.training.train_sequential import train as train_sequential
                train_cfg = dict(config_data)
                train_cfg["models"]["model_type"] = train_model_type
                train_sequential(train_cfg)
                st.success(f"✅ {train_model_type.upper()} model trained successfully!")
            except ImportError as e:
                st.error(f"Missing dependency: {e}\n\nRun: `pip install torch pytorch-lightning`")
            except Exception as e:
                st.error(f"Training failed: {e}")

    st.markdown("---")

    # ── Saved models ─────────────────────────────────────────────────────────
    st.subheader("📦 Saved Models")
    models_dir = config_data.get("inference", {}).get("models_dir", "models")
    saved = []
    if os.path.isdir(models_dir):
        for root, dirs, files in os.walk(models_dir):
            for f in files:
                fpath = os.path.join(root, f)
                mtime = os.path.getmtime(fpath)
                size_kb = os.path.getsize(fpath) / 1024
                saved.append({
                    "File": os.path.relpath(fpath, models_dir),
                    "Size": f"{size_kb:.1f} KB",
                    "Modified": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M"),
                })

    if saved:
        st.dataframe(pd.DataFrame(saved), use_container_width=True, hide_index=True)
    else:
        st.info("No model checkpoints found. Train a model above.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
with tab_config:
    st.subheader("⚙️ Configuration")

    config_data = _load_config()

    # ── Exchange ─────────────────────────────────────────────────────────────
    with st.expander("🔌 Exchange", expanded=True):
        ex = config_data.get("exchange", {})
        ec1, ec2, ec3 = st.columns(3)
        new_exchange = ec1.selectbox("Exchange", ["binance", "bybit", "okx", "kraken", "kucoin"],
                                     index=["binance","bybit","okx","kraken","kucoin"].index(
                                         ex.get("name", "binance")) if ex.get("name","binance") in
                                         ["binance","bybit","okx","kraken","kucoin"] else 0)
        ec2.text_input("API Key", value="", type="password",
                       placeholder="Set via EXCHANGE_API_KEY env var (recommended)")
        ec3.text_input("Secret Key", value="", type="password",
                       placeholder="Set via EXCHANGE_SECRET_KEY env var (recommended)")
        st.caption("🔐 Never save API keys to config.yaml. Use environment variables instead.")

    # ── Symbols ──────────────────────────────────────────────────────────────
    with st.expander("📊 Symbols & Data"):
        data_cfg = config_data.get("data", {})
        syms_str = st.text_input("Symbols (comma-separated)",
                                  value=", ".join(data_cfg.get("symbols", ["BTC/USDT", "ETH/USDT"])))
        sc1, sc2 = st.columns(2)
        new_tf = sc1.selectbox("Timeframe", ["1m","5m","15m","30m","1h","4h","1d"],
                               index=["1m","5m","15m","30m","1h","4h","1d"].index(
                                   data_cfg.get("timeframe","5m")))
        new_lookback = sc2.number_input("Lookback bars", value=data_cfg.get("lookback", 60),
                                        min_value=30, max_value=500)

    # ── Risk ─────────────────────────────────────────────────────────────────
    with st.expander("🛡️ Risk Management"):
        trd = config_data.get("trading", {})
        rc1, rc2, rc3 = st.columns(3)
        new_risk = rc1.number_input("Risk % per trade",
                                    value=trd.get("risk_percentage", 0.01) * 100,
                                    min_value=0.1, max_value=10.0, step=0.1,
                                    format="%.1f") / 100
        new_sl = rc2.number_input("Stop Loss %",
                                   value=trd.get("stop_loss_percentage", 0.02) * 100,
                                   min_value=0.1, max_value=20.0, step=0.1,
                                   format="%.1f") / 100
        new_tp = rc3.number_input("Take Profit %",
                                   value=trd.get("take_profit_percentage", 0.04) * 100,
                                   min_value=0.1, max_value=50.0, step=0.1,
                                   format="%.1f") / 100

    # ── Dry Run ──────────────────────────────────────────────────────────────
    with st.expander("🧪 Dry Run Mode"):
        dry_run_active = config_data.get("dry_run", True)
        new_dry_run = st.toggle("Enable Dry Run (no real orders)", value=dry_run_active)
        if new_dry_run:
            st.success("✅ Dry run ON — bot will simulate trades without real money")
        else:
            st.warning("⚠️ Dry run OFF — bot will place REAL orders on the exchange!")

    if st.button("💾 Save Configuration", type="primary"):
        try:
            config_data.setdefault("exchange", {})["name"] = new_exchange
            config_data.setdefault("data", {})["symbols"] = [
                s.strip() for s in syms_str.split(",") if s.strip()
            ]
            config_data["data"]["timeframe"] = new_tf
            config_data["data"]["lookback"] = int(new_lookback)
            config_data.setdefault("trading", {})["risk_percentage"] = new_risk
            config_data["trading"]["stop_loss_percentage"] = new_sl
            config_data["trading"]["take_profit_percentage"] = new_tp
            config_data["dry_run"] = new_dry_run

            with open(CONFIG_PATH, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            st.success("✅ Configuration saved to config.yaml")
        except Exception as e:
            st.error(f"Failed to save: {e}")

    st.markdown("---")

    # ── Raw YAML view ─────────────────────────────────────────────────────────
    with st.expander("📄 Raw config.yaml"):
        try:
            with open(CONFIG_PATH) as f:
                st.code(f.read(), language="yaml")
        except Exception:
            st.warning("Could not load config.yaml")


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Bot Controls
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🤖 Bot Controls")
    st.markdown("---")

    config_data = _load_config()

    if "metrics_started" not in st.session_state:
        st.session_state.metrics_started = False

    # ── Dry run toggle ───────────────────────────────────────────────────────
    dry_run = st.toggle("🧪 Dry Run Mode", value=True, key="sidebar_dry_run")
    if dry_run:
        st.caption("✅ Simulating trades — no real orders")
    else:
        st.caption("⚠️ LIVE MODE — real orders will be placed!")

    st.markdown("---")

    # ── Start/Stop ───────────────────────────────────────────────────────────
    running = _bot_actually_running()
    if not running:
        if st.button("▶️ Start Bot", type="primary", use_container_width=True):
            if not st.session_state.metrics_started:
                try:
                    metrics_thread = threading.Thread(
                        target=start_http_server, args=(8000,), daemon=True
                    )
                    metrics_thread.start()
                    st.session_state.metrics_started = True
                except Exception:
                    pass  # Port may already be in use

            _start_bot(CONFIG_PATH, dry_run)
            st.success("✅ Bot started!")
            st.rerun()
    else:
        if st.button("⏹ Stop Bot", type="secondary", use_container_width=True):
            _stop_bot()
            st.warning("⏹ Stop signal sent — bot will halt after current cycle.")
            st.rerun()

    st.markdown("---")

    # ── Auto-refresh ─────────────────────────────────────────────────────────
    auto_refresh = st.toggle("🔄 Auto-refresh", value=st.session_state.bot_running,
                              key="auto_refresh")
    refresh_secs = st.slider("Interval (seconds)", 5, 60, 15, key="refresh_interval")

    if auto_refresh:
        time.sleep(refresh_secs)
        st.rerun()

    st.markdown("---")

    # ── Info panel ───────────────────────────────────────────────────────────
    st.markdown("**📊 Metrics**: [Prometheus](http://localhost:9090)  |  [Grafana](http://localhost:3000)")
    st.markdown("---")
    st.caption(f"Last state: {_ago(last_update)}")
    st.caption("Bot state: `data/state/bot-state.json`")
    st.caption("Memory: `data/memory/*.json`")
