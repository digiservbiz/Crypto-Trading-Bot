"""Risk gating agent — the blocking gate before any broker execution.

The RiskAnalystAgent is the last line of defense before any trade is executed.
It evaluates every SignalProposal against VaR/CVaR thresholds, Kelly criterion
position sizing, Sharpe ratio requirements, and all circuit breakers.

A RiskDecision with decision == "approve" is the ONLY authorization for broker execution.
Any trade attempted without this authorization is a critical policy violation.

Pipeline position: trading-strategist → risk-analyst → broker (ONLY if approved)
"""

from typing import Dict, Optional, Any, List
import json
import logging
import math
import os
import time
import numpy as np
import pandas as pd

from scripts.agents.base_agent import (
    BaseAgent, AgentRole, SignalProposal, RiskDecision
)


logger = logging.getLogger(__name__)

# Memory file path for circuit breaker state persistence
RISK_MEMORY_PATH = "data/memory/trading-risk.json"

# Risk thresholds
VAR_THRESHOLD = 0.02        # VaR must be <= 2% of portfolio
CVAR_THRESHOLD = 0.03       # CVaR must be <= 3% of portfolio
SHARPE_FULL_SIZE = 1.5      # Sharpe above this → full size
SHARPE_HALF_SIZE = 1.0      # Sharpe above this → 50% size; below → reject
MAX_DRAWDOWN_THRESHOLD = 0.15   # Max drawdown: 15%
DAILY_LOSS_HALT_PCT = 0.03      # 3% daily loss → halt
WEEKLY_LOSS_REDUCTION_PCT = 0.05  # 5% weekly loss → 50% size reduction
CORRELATION_THRESHOLD = 0.85     # Max pairwise correlation
CONCENTRATION_LIMIT = 0.10       # Max single-asset portfolio concentration
VOLATILITY_SPIKE_RATIO = 2.0     # ATR ratio triggering size reduction
KELLY_FRACTION = 0.25            # Quarter-Kelly for safety


class RiskAnalystAgent(BaseAgent):
    """Blocking gate agent that evaluates every trade signal before execution.

    Computes VaR, CVaR, Kelly criterion, Sharpe ratio, and enforces all
    circuit breakers. Returns a RiskDecision with approve/reject and
    the approved adjusted position size.

    INVARIANT: A trade MUST NOT be executed without an approved RiskDecision.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the RiskAnalystAgent.

        Args:
            config: Full bot configuration dictionary.
        """
        super().__init__(AgentRole.RISK_ANALYST, config)
        self._state = self._load_state()

    def evaluate(
        self,
        signal_proposal: SignalProposal,
        portfolio_state: Dict[str, Any],
        df: pd.DataFrame,
    ) -> RiskDecision:
        """Evaluate a signal proposal and return a risk decision.

        This is the primary method. It runs all checks in order:
        1. Circuit breakers (hard stops)
        2. VaR/CVaR assessment
        3. Kelly criterion sizing
        4. Sharpe ratio check
        5. Correlation check

        Args:
            signal_proposal: The SignalProposal from trading-strategist.
            portfolio_state: Dict with keys: balance, positions, daily_pnl,
                weekly_pnl, portfolio_peak, open_positions.
            df: OHLCV DataFrame for historical simulation.

        Returns:
            RiskDecision with decision, adjusted size, and all risk metrics.
        """
        if signal_proposal is None:
            return self._reject_decision("", "NULL_SIGNAL", "No signal proposal provided")

        signal_id = signal_proposal.signal_id

        try:
            # ---- Extract portfolio state ----
            balance = float(portfolio_state.get("balance", 0.0))
            daily_pnl_pct = float(portfolio_state.get("daily_pnl_pct", 0.0))
            weekly_pnl_pct = float(portfolio_state.get("weekly_pnl_pct", 0.0))
            portfolio_peak = float(portfolio_state.get("portfolio_peak", balance))
            open_positions = portfolio_state.get("open_positions", {})
            atr_ratio = float(portfolio_state.get("atr_ratio", 1.0))

            if balance <= 0:
                return self._reject_decision(
                    signal_id, "INVALID_BALANCE", "Portfolio balance is zero or negative"
                )

            # ---- Step 1: Circuit breakers (hard stops) ----
            cb_triggered, cb_reason = self._check_circuit_breakers(
                daily_pnl_pct=daily_pnl_pct,
                weekly_pnl_pct=weekly_pnl_pct,
                balance=balance,
                portfolio_peak=portfolio_peak,
                signal_proposal=signal_proposal,
                open_positions=open_positions,
                atr_ratio=atr_ratio,
            )

            if cb_triggered and cb_reason in (
                "DAILY_LOSS_HALT", "MAX_DRAWDOWN_HALT", "CONCENTRATION_LIMIT",
                "CORRELATION_SPIKE"
            ):
                self._update_state({"last_rejection": cb_reason, "last_check": time.time()})
                return self._reject_decision(signal_id, cb_triggered, cb_reason)

            # ---- Step 2: Compute returns for VaR/CVaR ----
            returns = self._compute_returns(df)

            # ---- Step 3: Compute VaR and CVaR ----
            position_value = balance * signal_proposal.size_pct
            var_95 = self._compute_var(returns, position_value, balance)
            cvar_95 = self._compute_cvar(returns, position_value, balance)

            # ---- Step 4: Compute Sharpe ratio ----
            sharpe = self._compute_sharpe(returns)

            # ---- Step 5: Kelly criterion position sizing ----
            kelly_size = self._kelly_position(
                signal_proposal.confidence, sharpe, signal_proposal.size_pct
            )

            # ---- Step 6: Apply soft circuit breakers to size ----
            adjusted_size = kelly_size

            if cb_reason == "WEEKLY_REDUCTION":
                adjusted_size *= 0.5
                self.log_decision(
                    f"Weekly reduction active — size halved to {adjusted_size:.4f}"
                )

            if cb_reason == "VIX_SPIKE":
                adjusted_size *= 0.25
                self.log_decision(
                    f"Volatility spike — size reduced 75% to {adjusted_size:.4f}"
                )

            # ---- Step 7: Risk threshold checks ----
            if var_95 > VAR_THRESHOLD:
                reason = f"VAR_EXCEEDED: VaR {var_95:.3f} > threshold {VAR_THRESHOLD}"
                self.log_decision(reason, level="warning")
                return self._full_decision(
                    signal_id, "reject", 0.0, var_95, cvar_95, sharpe,
                    True, reason
                )

            if cvar_95 > CVAR_THRESHOLD:
                reason = f"CVAR_EXCEEDED: CVaR {cvar_95:.3f} > threshold {CVAR_THRESHOLD}"
                self.log_decision(reason, level="warning")
                return self._full_decision(
                    signal_id, "reject", 0.0, var_95, cvar_95, sharpe,
                    True, reason
                )

            if sharpe < SHARPE_HALF_SIZE:
                reason = f"SHARPE_TOO_LOW: Sharpe {sharpe:.3f} < {SHARPE_HALF_SIZE}"
                self.log_decision(reason, level="warning")
                return self._full_decision(
                    signal_id, "reject", 0.0, var_95, cvar_95, sharpe,
                    True, reason
                )

            # Sharpe between 1.0 and 1.5 → half size
            if sharpe < SHARPE_FULL_SIZE:
                adjusted_size *= 0.5
                self.log_decision(
                    f"Sharpe {sharpe:.3f} in [1.0,1.5] — size halved to {adjusted_size:.4f}"
                )

            # ---- Step 8: Final approval ----
            adjusted_size = max(0.001, min(adjusted_size, 0.10))  # Cap at 10%

            self.log_decision(
                f"APPROVED signal_id={signal_id} | "
                f"VaR={var_95:.4f} CVaR={cvar_95:.4f} Sharpe={sharpe:.3f} | "
                f"Size={adjusted_size:.4f}"
            )

            self._update_state({
                "last_approval": signal_id,
                "last_check": time.time(),
                "last_var": var_95,
                "last_sharpe": sharpe,
            })

            return self._full_decision(
                signal_id, "approve", adjusted_size, var_95, cvar_95, sharpe,
                False, None
            )

        except Exception as exc:
            self.log_decision(
                f"Risk evaluation error: {exc} — defaulting to reject", level="error"
            )
            return self._reject_decision(signal_id, True, f"EVALUATION_ERROR: {exc}")

    def _check_circuit_breakers(
        self,
        daily_pnl_pct: float,
        weekly_pnl_pct: float,
        balance: float,
        portfolio_peak: float,
        signal_proposal: SignalProposal,
        open_positions: Dict[str, Any],
        atr_ratio: float,
    ) -> tuple[bool, Optional[str]]:
        """Check all circuit breakers and return (triggered, reason).

        Returns (False, None) if no breakers triggered.
        Returns (True, reason_string) if a breaker is triggered.
        For soft breakers (weekly reduction, VIX spike) returns (True, reason)
        but the caller treats this as a size-reduction rather than full rejection.
        """
        # CB1: Daily loss halt
        if daily_pnl_pct <= -DAILY_LOSS_HALT_PCT:
            reason = f"DAILY_LOSS_HALT: {daily_pnl_pct*100:.2f}% daily loss"
            self.log_decision(reason, level="warning")
            self._log_circuit_breaker_event(reason)
            return True, "DAILY_LOSS_HALT"

        # CB4: Max drawdown
        if portfolio_peak > 0:
            drawdown = (balance - portfolio_peak) / portfolio_peak
            self._state["current_drawdown_pct"] = drawdown
            if drawdown <= -MAX_DRAWDOWN_THRESHOLD:
                reason = (
                    f"MAX_DRAWDOWN_HALT: {drawdown*100:.2f}% drawdown "
                    f"(threshold: {MAX_DRAWDOWN_THRESHOLD*100:.1f}%)"
                )
                self.log_decision(reason, level="warning")
                self._log_circuit_breaker_event(reason)
                return True, "MAX_DRAWDOWN_HALT"

        # CB3: Correlation spike
        if open_positions and len(open_positions) > 1:
            max_corr = self._state.get("max_pairwise_correlation", 0.0)
            if max_corr > CORRELATION_THRESHOLD:
                reason = f"CORRELATION_SPIKE: max correlation {max_corr:.3f} > {CORRELATION_THRESHOLD}"
                self.log_decision(reason, level="warning")
                return True, "CORRELATION_SPIKE"

        # CB6: Concentration limit
        if open_positions:
            asset = signal_proposal.symbol.split("/")[0]
            current_concentration = open_positions.get(asset, {}).get(
                "concentration_pct", 0.0
            )
            if current_concentration + signal_proposal.size_pct > CONCENTRATION_LIMIT:
                reason = (
                    f"CONCENTRATION_LIMIT: {asset} would reach "
                    f"{(current_concentration + signal_proposal.size_pct)*100:.1f}%"
                )
                self.log_decision(reason, level="warning")
                return True, "CONCENTRATION_LIMIT"

        # CB2: Weekly reduction (soft — returns True but caller does size reduction)
        if weekly_pnl_pct <= -WEEKLY_LOSS_REDUCTION_PCT:
            reason = f"WEEKLY_REDUCTION: {weekly_pnl_pct*100:.2f}% weekly loss"
            self.log_decision(reason + " — applying 50% size reduction", level="warning")
            return True, "WEEKLY_REDUCTION"

        # CB5: VIX/volatility spike (soft — size reduction)
        if atr_ratio > VOLATILITY_SPIKE_RATIO:
            reason = f"VIX_SPIKE: ATR ratio {atr_ratio:.2f} > {VOLATILITY_SPIKE_RATIO}"
            self.log_decision(reason + " — applying 75% size reduction", level="warning")
            return True, "VIX_SPIKE"

        return False, None

    def _compute_returns(self, df: pd.DataFrame) -> np.ndarray:
        """Extract daily-equivalent returns from OHLCV data."""
        if df is None or len(df) < 2:
            return np.array([-0.01, 0.0, 0.01])  # Minimal fallback
        close = df["close"].values.astype(float)
        returns = np.diff(close) / close[:-1]
        return returns[np.isfinite(returns)]

    def _compute_var(
        self, returns: np.ndarray, position_value: float, portfolio_value: float
    ) -> float:
        """Compute Value at Risk at 95% confidence using historical simulation.

        VaR represents the maximum loss not exceeded at the given confidence level.
        Returns as a fraction of total portfolio value.
        """
        if len(returns) < 10:
            return 0.02  # Conservative default
        var_return = -float(np.percentile(returns, 5))
        var_amount = var_return * position_value
        return var_amount / portfolio_value if portfolio_value > 0 else 0.02

    def _compute_cvar(
        self, returns: np.ndarray, position_value: float, portfolio_value: float
    ) -> float:
        """Compute Conditional VaR (Expected Shortfall) at 95% confidence.

        CVaR is the mean loss given that VaR is exceeded.
        Returns as a fraction of total portfolio value.
        """
        if len(returns) < 10:
            return 0.03  # Conservative default
        threshold = float(np.percentile(returns, 5))
        tail_returns = returns[returns < threshold]
        if len(tail_returns) == 0:
            return abs(threshold) * position_value / portfolio_value
        cvar_return = -float(np.mean(tail_returns))
        cvar_amount = cvar_return * position_value
        return cvar_amount / portfolio_value if portfolio_value > 0 else 0.03

    def _compute_sharpe(self, returns: np.ndarray, risk_free_rate: float = 0.05) -> float:
        """Compute annualized Sharpe ratio from return series.

        Uses last 30-day window if available, adjusted to annual frequency.
        """
        if len(returns) < 5:
            return 1.0  # Neutral default
        window = returns[-min(30, len(returns)):]
        mean_ret = float(np.mean(window))
        std_ret = float(np.std(window, ddof=1))
        if std_ret == 0:
            return 0.0
        # Annualize (assuming 288 5-minute bars per day, 365 days)
        candle_periods_per_year = 365 * 24 * 12  # 5-minute bars
        # Use sqrt(n_periods) for annualization
        n_per_day = len(returns) / max(len(window), 1) if len(window) > 0 else 288
        annualize_factor = math.sqrt(candle_periods_per_year / max(n_per_day, 1))
        daily_rf = risk_free_rate / 365
        sharpe = (mean_ret - daily_rf) / std_ret * annualize_factor
        return float(np.clip(sharpe, -10.0, 10.0))

    def _kelly_position(
        self, win_prob: float, sharpe: float, proposed_size: float
    ) -> float:
        """Compute Kelly criterion position size.

        Uses quarter-Kelly for conservatism.

        Args:
            win_prob: Estimated probability of winning trade (signal confidence).
            sharpe: Current Sharpe ratio (used to estimate win/loss ratio).
            proposed_size: The size originally proposed by the strategist.

        Returns:
            Kelly-adjusted position size as fraction of portfolio.
        """
        p = max(0.01, min(0.99, win_prob))
        q = 1.0 - p
        # Estimate b (win/loss ratio) from Sharpe; rough approximation
        b = max(0.1, 1.0 + sharpe * 0.5)
        kelly_full = (p * b - q) / b
        kelly_quarter = KELLY_FRACTION * kelly_full
        # Use the minimum of Kelly size and proposed size
        kelly_quarter = max(0.001, kelly_quarter)
        return min(kelly_quarter, proposed_size)

    def _log_circuit_breaker_event(self, reason: str) -> None:
        """Append a circuit breaker event to the persistent log."""
        log = self._state.get("circuit_breaker_log", [])
        log.append({"reason": reason, "timestamp": time.time()})
        self._state["circuit_breaker_log"] = log[-50:]  # Keep last 50
        self._save_state()

    def _load_state(self) -> Dict[str, Any]:
        """Load persistent circuit breaker state from disk."""
        os.makedirs(os.path.dirname(RISK_MEMORY_PATH), exist_ok=True)
        if os.path.exists(RISK_MEMORY_PATH):
            try:
                with open(RISK_MEMORY_PATH, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning("Could not load risk state — starting fresh")
        return {
            "daily_loss_pct": 0.0,
            "weekly_loss_pct": 0.0,
            "current_drawdown_pct": 0.0,
            "max_pairwise_correlation": 0.0,
            "circuit_breaker_log": [],
        }

    def _save_state(self) -> None:
        """Persist circuit breaker state to disk."""
        try:
            os.makedirs(os.path.dirname(RISK_MEMORY_PATH), exist_ok=True)
            with open(RISK_MEMORY_PATH, "w") as f:
                json.dump(self._state, f, indent=2, default=str)
        except IOError as exc:
            logger.error("Could not save risk state: %s", exc)

    def _update_state(self, updates: Dict[str, Any]) -> None:
        """Update and persist state."""
        self._state.update(updates)
        self._save_state()

    def _reject_decision(
        self,
        signal_id: str,
        cb_triggered: Any,
        reason: str,
    ) -> RiskDecision:
        """Create a rejection RiskDecision."""
        self.log_decision(f"REJECTED signal_id={signal_id}: {reason}", level="warning")
        return RiskDecision(
            signal_id=signal_id,
            decision="reject",
            adjusted_size_pct=0.0,
            var_95=0.0,
            cvar_95=0.0,
            sharpe_ratio=0.0,
            circuit_breaker_triggered=bool(cb_triggered),
            circuit_breaker_reason=reason,
        )

    def _full_decision(
        self,
        signal_id: str,
        decision: str,
        adjusted_size: float,
        var_95: float,
        cvar_95: float,
        sharpe: float,
        cb_triggered: bool,
        cb_reason: Optional[str],
    ) -> RiskDecision:
        """Create a complete RiskDecision with all metrics."""
        return RiskDecision(
            signal_id=signal_id,
            decision=decision,
            adjusted_size_pct=round(adjusted_size, 4),
            var_95=round(var_95, 4),
            cvar_95=round(cvar_95, 4),
            sharpe_ratio=round(sharpe, 4),
            circuit_breaker_triggered=cb_triggered,
            circuit_breaker_reason=cb_reason,
        )

    def get_risk_state(self) -> Dict[str, Any]:
        """Return current persistent risk state for dashboard/monitoring."""
        return dict(self._state)

    def update_correlation(self, max_correlation: float) -> None:
        """Update the maximum pairwise correlation in state.

        Called by orchestrator after computing portfolio correlations.

        Args:
            max_correlation: Current maximum pairwise correlation between holdings.
        """
        self._update_state({"max_pairwise_correlation": max_correlation})
