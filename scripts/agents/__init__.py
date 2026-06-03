"""Multi-agent trading pipeline for the ruflo crypto trading bot.

This package implements the 4-agent ruflo pipeline:

Live Trading Pipeline (sequential):
    market-analyst → trading-strategist → risk-analyst → broker

Research Lane (isolated):
    backtest-engineer → SignedBacktestArtifact → trading-backtests namespace

Agents:
    MarketAnalystAgent: Regime detection (bull/bear/ranging/volatile/transitioning)
    TradingStrategistAgent: Strategy selection and Z-score signal generation
    RiskAnalystAgent: VaR/CVaR/circuit-breaker gate (NEVER bypass this)
    BacktestEngineerAgent: Walk-forward validation and Monte Carlo testing

Message types:
    RegimeVerdict: market-analyst → trading-strategist
    SignalProposal: trading-strategist → risk-analyst
    RiskDecision: risk-analyst → orchestrator (approve/reject)
    SignedBacktestArtifact: backtest-engineer → trading-backtests namespace

Entry point:
    TradingOrchestrator: Coordinates the full live pipeline
"""

from scripts.agents.base_agent import (
    AgentRole,
    BaseAgent,
    RegimeVerdict,
    SignalProposal,
    RiskDecision,
    SignedBacktestArtifact,
)
from scripts.agents.market_analyst import MarketAnalystAgent
from scripts.agents.trading_strategist import TradingStrategistAgent
from scripts.agents.risk_analyst import RiskAnalystAgent
from scripts.agents.backtest_engineer import BacktestEngineerAgent
from scripts.agents.orchestrator import TradingOrchestrator

__all__ = [
    # Enums and base classes
    "AgentRole",
    "BaseAgent",
    # Message types
    "RegimeVerdict",
    "SignalProposal",
    "RiskDecision",
    "SignedBacktestArtifact",
    # Agent implementations
    "MarketAnalystAgent",
    "TradingStrategistAgent",
    "RiskAnalystAgent",
    "BacktestEngineerAgent",
    # Pipeline coordinator
    "TradingOrchestrator",
]
