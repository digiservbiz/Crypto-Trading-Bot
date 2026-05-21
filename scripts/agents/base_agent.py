"""Base agent class for the ruflo multi-agent trading pipeline.

This module defines the shared dataclasses and base class used by all agents
in the crypto trading pipeline: market-analyst → trading-strategist →
risk-analyst → broker, with backtest-engineer as an orthogonal research lane.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import logging
import time
import uuid


logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Enumeration of all agent roles in the ruflo pipeline."""
    MARKET_ANALYST = "market-analyst"
    TRADING_STRATEGIST = "trading-strategist"
    RISK_ANALYST = "risk-analyst"
    BACKTEST_ENGINEER = "backtest-engineer"


@dataclass
class RegimeVerdict:
    """Output message from the market-analyst agent.

    Classifies the current market regime based on technical indicators.
    Sent exclusively to the trading-strategist.

    Regime types:
        bull-trending: Strong upward directional movement (ADX > 25, +DI > -DI)
        bear-trending: Strong downward directional movement (ADX > 25, -DI > +DI)
        ranging: Low-directional oscillating price (ADX < 20)
        high-volatility: Large rapid price swings (ATR > 2x average)
        low-volatility: Compressed quiet price action (ATR < 0.5x average)
        transitioning: Regime change in progress, unclear direction
    """
    regime_type: str  # bull-trending | bear-trending | ranging | high-volatility | low-volatility | transitioning
    confidence: float  # 0.0–1.0 indicator agreement score
    indicator_values: Dict[str, float]  # adx, rsi, atr, macd, bb_width, obv_trend, etc.
    symbols: List[str]  # Symbols analyzed in this verdict
    timestamp: float = field(default_factory=time.time)
    verdict_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    VALID_REGIMES = frozenset([
        "bull-trending", "bear-trending", "ranging",
        "high-volatility", "low-volatility", "transitioning"
    ])

    def is_valid(self) -> bool:
        """Return True if this verdict has valid content."""
        return (
            self.regime_type in self.VALID_REGIMES
            and 0.0 <= self.confidence <= 1.0
            and len(self.symbols) > 0
        )

    def is_stale(self, max_age_seconds: float = 300.0) -> bool:
        """Return True if this verdict is older than max_age_seconds."""
        return (time.time() - self.timestamp) > max_age_seconds


@dataclass
class SignalProposal:
    """Output message from the trading-strategist agent.

    Encodes a proposed trade signal. MUST be validated by the risk-analyst
    before any broker execution. The strategist never invokes the broker.

    Anomaly types:
        spike: Sudden sharp price movement (|z| > 3.0, single candle)
        drift: Gradual directional drift (z > 1.5 sustained 3+ candles)
        flatline: Abnormally low volatility (std < 0.001)
        oscillation: Rapid buy/sell oscillation
        pattern-break: Breakout from consolidation
        cluster-outlier: Statistical outlier in multivariate space
        none: No anomaly detected
    """
    symbol: str  # Trading pair (e.g., "BTC/USDT")
    side: str  # "buy" | "sell"
    size_pct: float  # Percentage of portfolio to allocate (0.0–1.0)
    confidence: float  # 0.0–1.0 signal confidence
    strategy_name: str  # momentum | mean-reversion | pairs | adaptive
    anomaly_type: str  # Z-score anomaly classification
    anomaly_score: float  # Raw Z-score magnitude
    regime_verdict_id: str  # Links back to RegimeVerdict.verdict_id
    timestamp: float = field(default_factory=time.time)
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def is_valid(self) -> bool:
        """Return True if this signal has valid content."""
        return (
            self.symbol
            and self.side in ("buy", "sell")
            and 0.0 <= self.size_pct <= 1.0
            and 0.0 <= self.confidence <= 1.0
        )


@dataclass
class RiskDecision:
    """Output message from the risk-analyst agent — the blocking gate.

    This is the ONLY authorization that permits broker execution.
    A trade MUST NOT proceed without:
        decision == "approve"
        circuit_breaker_triggered == False
        signal_id matching the active SignalProposal

    Circuit breaker reasons:
        DAILY_LOSS_HALT: Realized loss >= 3% of start-of-day balance
        WEEKLY_REDUCTION: Realized loss >= 5% of start-of-week balance
        CORRELATION_SPIKE: Pairwise correlation > 0.85
        MAX_DRAWDOWN_HALT: Portfolio drawdown >= 15% (manual reset required)
        VIX_SPIKE: Volatility > 2x 90-day historical average
        CONCENTRATION_LIMIT: Single asset would exceed 10% of portfolio
        VAR_EXCEEDED: VaR > 2% or CVaR > 3%
        SHARPE_TOO_LOW: Rolling Sharpe ratio < 1.0
    """
    signal_id: str  # References SignalProposal.signal_id
    decision: str  # "approve" | "reject"
    adjusted_size_pct: float  # Kelly-adjusted, circuit-breaker-modified size
    var_95: float  # Value at Risk at 95% confidence
    cvar_95: float  # Conditional Value at Risk at 95% confidence
    sharpe_ratio: float  # Rolling 30-day Sharpe ratio
    circuit_breaker_triggered: bool  # True if any circuit breaker fired
    circuit_breaker_reason: Optional[str]  # Which breaker and why
    timestamp: float = field(default_factory=time.time)

    def is_approved(self) -> bool:
        """Return True only when the decision is fully approved for execution."""
        return (
            self.decision == "approve"
            and not self.circuit_breaker_triggered
            and self.adjusted_size_pct > 0.0
        )


@dataclass
class SignedBacktestArtifact:
    """Output from the backtest-engineer agent.

    Stored to the trading-backtests memory namespace. NOT broadcast to any
    live pipeline agent. Represents a fully validated strategy with all
    6 quality gates evaluated.

    Quality gates (all must pass for passed_quality_gates == True):
        1. Minimum trades: >= 30 in test period
        2. Win-rate consistency: variance < 15% across walk-forward windows
        3. Monte Carlo significance: p-value < 0.05
        4. Max drawdown: < 15% across all windows
        5. Profit factor: > 1.5
        6. Sharpe ratio: > 1.0 (annualized)
    """
    strategy_id: str  # Strategy identifier
    symbol: str  # Trading pair
    period: str  # Date range (e.g., "2023-01-01/2024-01-01")
    total_return: float  # Net return over test period
    sharpe_ratio: float  # Annualized Sharpe ratio
    sortino_ratio: float  # Sortino ratio (downside deviation)
    max_drawdown: float  # Maximum peak-to-trough drawdown (negative)
    win_rate: float  # Fraction of trades that were profitable
    profit_factor: float  # Gross profit / gross loss
    num_trades: int  # Total trades in test period
    walk_forward_scores: List[float]  # Per-window Sharpe ratios
    monte_carlo_p_value: float  # Statistical significance p-value
    passed_quality_gates: bool  # True only if ALL 6 gates passed
    timestamp: float = field(default_factory=time.time)
    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def quality_gate_summary(self) -> Dict[str, bool]:
        """Return a dict showing which quality gates passed or failed."""
        win_rate_variance = (
            max(self.walk_forward_scores) - min(self.walk_forward_scores)
            if len(self.walk_forward_scores) > 1 else 0.0
        )
        return {
            "min_trades": self.num_trades >= 30,
            "win_rate_consistency": win_rate_variance < 0.15,
            "monte_carlo_significance": self.monte_carlo_p_value < 0.05,
            "max_drawdown": abs(self.max_drawdown) < 0.15,
            "profit_factor": self.profit_factor > 1.5,
            "sharpe_ratio": self.sharpe_ratio > 1.0,
        }


class BaseAgent:
    """Base class for all ruflo pipeline agents.

    Provides common initialization, logging, and configuration access.
    Subclasses should implement their primary task method (analyze, generate_signal,
    evaluate, or run_walk_forward).
    """

    def __init__(self, role: AgentRole, config: Dict[str, Any]) -> None:
        """Initialize the base agent.

        Args:
            role: The AgentRole enum value identifying this agent.
            config: Full bot configuration dictionary from config.yaml.
        """
        self.role = role
        self.config = config
        self.name = role.value
        self.logger = logging.getLogger(f"agents.{self.name}")
        self.logger.info("Agent initialized: %s", self.name)

    def log_decision(self, message: str, level: str = "info") -> None:
        """Log a structured agent decision message.

        Args:
            message: Human-readable decision description.
            level: Logging level (info, warning, error, critical).
        """
        log_fn = getattr(self.logger, level, self.logger.info)
        log_fn("[%s] %s", self.name.upper(), message)

    def get_config_value(self, *keys: str, default: Any = None) -> Any:
        """Safely retrieve a nested config value.

        Args:
            *keys: Sequence of keys to traverse in the config dict.
            default: Value to return if key path is not found.

        Returns:
            The config value or default.
        """
        value = self.config
        for key in keys:
            if not isinstance(value, dict):
                return default
            value = value.get(key, default)
            if value is None:
                return default
        return value
