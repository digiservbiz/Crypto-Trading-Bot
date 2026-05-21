"""Main pipeline coordinator for the ruflo multi-agent trading system.

The TradingOrchestrator coordinates the full live trading pipeline:
    market-analyst → trading-strategist → risk-analyst → broker (ONLY if approved)

It enforces the non-negotiable rule: NO broker execution without a RiskDecision
with decision == "approve" and circuit_breaker_triggered == False.

The orchestrator maintains circuit breaker state tracking across the trading loop
and logs each agent decision in structured format for audit trails.
"""

from typing import Dict, List, Optional, Any
import logging
import time

import pandas as pd

from scripts.agents.base_agent import (
    AgentRole, RegimeVerdict, SignalProposal, RiskDecision
)
from scripts.agents.market_analyst import MarketAnalystAgent
from scripts.agents.trading_strategist import TradingStrategistAgent
from scripts.agents.risk_analyst import RiskAnalystAgent


logger = logging.getLogger(__name__)


class TradingOrchestrator:
    """Coordinator for the multi-agent trading pipeline.

    Manages agent instances and enforces the message-passing protocol between
    market-analyst → trading-strategist → risk-analyst.

    INVARIANT: Broker execution only occurs when the orchestrator receives a
    RiskDecision with is_approved() == True. Any violation is a critical defect.

    Usage:
        orchestrator = TradingOrchestrator(config)
        decision = orchestrator.run_pipeline(df, symbols, ai_engine, portfolio_state)
        if decision and decision.is_approved():
            # Execute trade with decision.adjusted_size_pct
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the TradingOrchestrator with all three live-pipeline agents.

        Args:
            config: Full bot configuration dictionary from config.yaml.
        """
        self.config = config
        self.logger = logging.getLogger("orchestrator")

        # Instantiate live pipeline agents
        self.market_analyst = MarketAnalystAgent(config)
        self.trading_strategist = TradingStrategistAgent(config)
        self.risk_analyst = RiskAnalystAgent(config)

        # Pipeline execution state
        self._pipeline_count = 0
        self._approved_count = 0
        self._rejected_count = 0
        self._circuit_breaker_halts = 0
        self._last_regime: Optional[RegimeVerdict] = None
        self._last_signal: Optional[SignalProposal] = None
        self._last_decision: Optional[RiskDecision] = None

        self.logger.info(
            "TradingOrchestrator initialized with agents: %s",
            [a.value for a in AgentRole if a != AgentRole.BACKTEST_ENGINEER]
        )

    def run_pipeline(
        self,
        df: pd.DataFrame,
        symbols: List[str],
        ai_engine: Optional[Any] = None,
        portfolio_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[RiskDecision]:
        """Execute the full multi-agent pipeline for one trading cycle.

        Pipeline stages:
        1. market-analyst: Compute regime verdict from OHLCV data
        2. trading-strategist: Generate signal proposal from regime verdict
        3. risk-analyst: Evaluate signal and return approve/reject decision

        Args:
            df: OHLCV DataFrame for the primary symbol.
            symbols: List of symbols being analyzed.
            ai_engine: Optional AIEngine for neural model predictions.
            portfolio_state: Current portfolio state for risk assessment.
                Keys: balance, daily_pnl_pct, weekly_pnl_pct, portfolio_peak,
                      open_positions, atr_ratio

        Returns:
            RiskDecision if pipeline completes (may be approve or reject).
            None if pipeline aborts early (no regime, no signal, etc.).
        """
        self._pipeline_count += 1
        cycle_id = f"cycle-{self._pipeline_count:04d}"

        self.logger.info(
            "[%s] === Pipeline start | symbols=%s ===", cycle_id, symbols
        )

        if portfolio_state is None:
            portfolio_state = self._default_portfolio_state()

        # ---- Stage 1: Market Analyst — Regime Detection ----
        regime_verdict = self._run_market_analyst(df, symbols, cycle_id)

        if regime_verdict is None:
            self.logger.warning(
                "[%s] Pipeline aborted at market-analyst stage", cycle_id
            )
            return None

        self._last_regime = regime_verdict

        # ---- Stage 2: Trading Strategist — Signal Generation ----
        signal_proposal = self._run_trading_strategist(
            df, regime_verdict, ai_engine, cycle_id
        )

        if signal_proposal is None:
            self.logger.info(
                "[%s] No signal generated (regime=%s, confidence=%.3f) — pipeline complete",
                cycle_id, regime_verdict.regime_type, regime_verdict.confidence
            )
            return None

        self._last_signal = signal_proposal

        # ---- Stage 3: Risk Analyst — Risk Gate ----
        risk_decision = self._run_risk_analyst(
            signal_proposal, portfolio_state, df, cycle_id
        )

        self._last_decision = risk_decision

        # ---- Log pipeline outcome ----
        if risk_decision.is_approved():
            self._approved_count += 1
            self.logger.info(
                "[%s] === Pipeline APPROVED | signal_id=%s | side=%s | "
                "size=%.4f | VaR=%.4f | Sharpe=%.3f ===",
                cycle_id,
                risk_decision.signal_id,
                signal_proposal.side,
                risk_decision.adjusted_size_pct,
                risk_decision.var_95,
                risk_decision.sharpe_ratio,
            )
        else:
            self._rejected_count += 1
            if risk_decision.circuit_breaker_triggered:
                self._circuit_breaker_halts += 1
            self.logger.info(
                "[%s] === Pipeline REJECTED | reason=%s | cb=%s ===",
                cycle_id,
                risk_decision.circuit_breaker_reason,
                risk_decision.circuit_breaker_triggered,
            )

        return risk_decision

    def _run_market_analyst(
        self,
        df: pd.DataFrame,
        symbols: List[str],
        cycle_id: str,
    ) -> Optional[RegimeVerdict]:
        """Execute market-analyst stage with error isolation.

        Args:
            df: OHLCV DataFrame.
            symbols: Symbol list.
            cycle_id: Pipeline cycle identifier for logging.

        Returns:
            RegimeVerdict or None on failure.
        """
        try:
            start_time = time.time()
            verdict = self.market_analyst.analyze(df, symbols)
            elapsed_ms = (time.time() - start_time) * 1000

            self.logger.info(
                "[%s][market-analyst] regime=%s confidence=%.3f elapsed=%.1fms",
                cycle_id, verdict.regime_type, verdict.confidence, elapsed_ms
            )

            if not verdict.is_valid():
                self.logger.warning(
                    "[%s][market-analyst] Verdict failed validation", cycle_id
                )
                return None

            return verdict

        except Exception as exc:
            self.logger.error(
                "[%s][market-analyst] Unexpected error: %s", cycle_id, exc
            )
            return None

    def _run_trading_strategist(
        self,
        df: pd.DataFrame,
        regime_verdict: RegimeVerdict,
        ai_engine: Optional[Any],
        cycle_id: str,
    ) -> Optional[SignalProposal]:
        """Execute trading-strategist stage with error isolation.

        CRITICAL: This stage must never attempt broker execution.

        Args:
            df: OHLCV DataFrame.
            regime_verdict: Validated RegimeVerdict from market-analyst.
            ai_engine: Optional neural model engine.
            cycle_id: Pipeline cycle identifier.

        Returns:
            SignalProposal or None if no signal generated.
        """
        try:
            start_time = time.time()
            proposal = self.trading_strategist.generate_signal(
                df=df,
                regime_verdict=regime_verdict,
                ai_engine=ai_engine,
            )
            elapsed_ms = (time.time() - start_time) * 1000

            if proposal is not None:
                self.logger.info(
                    "[%s][trading-strategist] signal_id=%s side=%s strategy=%s "
                    "confidence=%.3f anomaly=%s elapsed=%.1fms",
                    cycle_id,
                    proposal.signal_id,
                    proposal.side,
                    proposal.strategy_name,
                    proposal.confidence,
                    proposal.anomaly_type,
                    elapsed_ms,
                )
            else:
                self.logger.info(
                    "[%s][trading-strategist] No signal (elapsed=%.1fms)",
                    cycle_id, elapsed_ms
                )

            return proposal

        except Exception as exc:
            self.logger.error(
                "[%s][trading-strategist] Unexpected error: %s", cycle_id, exc
            )
            return None

    def _run_risk_analyst(
        self,
        signal_proposal: SignalProposal,
        portfolio_state: Dict[str, Any],
        df: pd.DataFrame,
        cycle_id: str,
    ) -> RiskDecision:
        """Execute risk-analyst stage with error isolation.

        If this stage throws an unexpected exception, it defaults to reject
        to ensure safety. A RiskDecision is ALWAYS returned from this method.

        Args:
            signal_proposal: SignalProposal from trading-strategist.
            portfolio_state: Current portfolio state.
            df: OHLCV DataFrame for historical simulation.
            cycle_id: Pipeline cycle identifier.

        Returns:
            RiskDecision (always — defaults to reject on errors).
        """
        try:
            start_time = time.time()
            decision = self.risk_analyst.evaluate(
                signal_proposal=signal_proposal,
                portfolio_state=portfolio_state,
                df=df,
            )
            elapsed_ms = (time.time() - start_time) * 1000

            self.logger.info(
                "[%s][risk-analyst] decision=%s signal_id=%s cb=%s elapsed=%.1fms",
                cycle_id,
                decision.decision,
                decision.signal_id,
                decision.circuit_breaker_triggered,
                elapsed_ms,
            )

            return decision

        except Exception as exc:
            self.logger.error(
                "[%s][risk-analyst] Unexpected error: %s — defaulting to reject",
                cycle_id, exc
            )
            from scripts.agents.base_agent import RiskDecision
            return RiskDecision(
                signal_id=signal_proposal.signal_id if signal_proposal else "",
                decision="reject",
                adjusted_size_pct=0.0,
                var_95=0.0,
                cvar_95=0.0,
                sharpe_ratio=0.0,
                circuit_breaker_triggered=True,
                circuit_breaker_reason=f"ORCHESTRATOR_ERROR: {exc}",
            )

    def _default_portfolio_state(self) -> Dict[str, Any]:
        """Return a safe default portfolio state when none is provided."""
        return {
            "balance": 0.0,
            "daily_pnl_pct": 0.0,
            "weekly_pnl_pct": 0.0,
            "portfolio_peak": 0.0,
            "open_positions": {},
            "atr_ratio": 1.0,
        }

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Return pipeline execution statistics.

        Returns:
            Dict with counts of cycles, approvals, rejections, and CB halts.
        """
        return {
            "total_cycles": self._pipeline_count,
            "approved": self._approved_count,
            "rejected": self._rejected_count,
            "circuit_breaker_halts": self._circuit_breaker_halts,
            "approval_rate": (
                self._approved_count / self._pipeline_count
                if self._pipeline_count > 0 else 0.0
            ),
            "last_regime": (
                self._last_regime.regime_type if self._last_regime else None
            ),
            "last_decision": (
                self._last_decision.decision if self._last_decision else None
            ),
        }

    def get_risk_state(self) -> Dict[str, Any]:
        """Expose the risk-analyst's current persistent state for monitoring."""
        return self.risk_analyst.get_risk_state()

    def update_portfolio_correlation(self, max_correlation: float) -> None:
        """Update the maximum pairwise portfolio correlation in the risk agent.

        Args:
            max_correlation: Current maximum pairwise correlation.
        """
        self.risk_analyst.update_correlation(max_correlation)
