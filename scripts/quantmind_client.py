"""
QuantMind integration client for the Crypto Trading Bot.

Provides research-backed sentiment, position sizing, model selection,
and strategy validation by reading from a local research cache built
nightly by research_updater.py.
"""
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SYMBOL_KEYWORDS: dict[str, list[str]] = {
    'BTC': ['bitcoin', 'btc', 'cryptocurrency', 'crypto'],
    'ETH': ['ethereum', 'eth', 'smart contract', 'defi'],
}

_POSITIVE_WORDS = {
    'outperform', 'profitable', 'gain', 'alpha', 'upward',
    'bullish', 'improve', 'advantage', 'positive', 'strong',
    'significant', 'robust', 'effective', 'superior',
}
_NEGATIVE_WORDS = {
    'underperform', 'loss', 'downward', 'bearish', 'decline',
    'weak', 'fail', 'negative', 'risk', 'limitation',
    'ineffective', 'poor', 'unstable', 'volatile',
}


class QuantMindClient:
    """
    Reads from data/research_cache.json (written by research_updater.py)
    and exposes four capabilities consumed by the trading bot:
      1. Research sentiment per symbol  (#1)
      2. Position-size multiplier       (#3)
      3. Model architecture suggestion  (#5)
      4. Strategy parameter validation  (#4)
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        qm = config.get('quantmind', {})
        self.cache_path = Path(qm.get('cache_path', 'data/research_cache.json'))
        self._cache: dict = {'cards': [], 'updated_at': None}
        self._load_cache()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_research_sentiment(self, symbol: str) -> float:
        """
        Returns a research sentiment score in [-1.0, 1.0] for the given
        symbol, derived from key findings in cached paper cards.
        Returns 0.0 when the cache is empty or no relevant papers found.
        """
        keywords = self._keywords_for(symbol)
        cards = self._relevant_cards(keywords)
        if not cards:
            return 0.0

        scores: list[float] = []
        for card in cards:
            for section in card.get('sections', []):
                score = self._score_text(section.get('summary', ''), keywords)
                if score is not None:
                    scores.append(score)
            root_score = self._score_text(card.get('summary', ''), keywords)
            if root_score is not None:
                scores.append(root_score)

        return round(sum(scores) / len(scores), 4) if scores else 0.0

    def get_research_signal_multiplier(self, symbol: str) -> float:
        """
        Returns a position-size multiplier in [0.5, 1.5].
        Neutral research → 1.0 (no change to size).
        Strong bullish research → up to 1.5×; bearish → down to 0.5×.
        """
        sentiment = self.get_research_sentiment(symbol)
        return round(1.0 + (sentiment * 0.5), 4)

    def get_recommended_model(self, volatility: float, symbol: str) -> str:
        """
        Returns 'lstm' or 'transformer' based on what recent research
        says about model performance in the current volatility regime.
        Falls back to the config default if the cache is empty.
        """
        default = self.config['models'].get('model_type', 'lstm')
        keywords = self._keywords_for(symbol) + ['lstm', 'transformer', 'attention', 'recurrent']
        cards = self._relevant_cards(keywords)
        if not cards:
            return default

        transformer_votes = 0
        lstm_votes = 0
        for card in cards:
            text = (card.get('summary', '') + ' ' +
                    ' '.join(s.get('summary', '') for s in card.get('sections', []))).lower()
            transformer_votes += text.count('transformer') + text.count('attention mechanism')
            lstm_votes += text.count('lstm') + text.count('recurrent')

        vol_threshold = self.config['models']['model_selection']['volatility_threshold']

        if volatility > vol_threshold:
            # High volatility: Transformers handle regime shifts better per literature
            return 'transformer' if transformer_votes >= lstm_votes else 'lstm'
        # Low volatility: LSTM works well for stable sequences
        return 'lstm' if lstm_votes >= transformer_votes else 'transformer'

    def validate_strategy(self, strategy_params: dict) -> dict:
        """
        Checks strategy parameters against cached research findings.
        Returns {'alignment_score': float [0-1], 'notes': list[str]}.
        """
        cards = self._relevant_cards(['risk', 'stop loss', 'momentum', 'trading strategy', 'factor'])
        notes: list[str] = []
        scores: list[float] = []

        rp = strategy_params.get('risk_percentage', 0.01)
        sl = strategy_params.get('stop_loss_percentage', 0.02)
        tp = strategy_params.get('take_profit_percentage', 0.04)
        rrr = tp / sl if sl > 0 else 0

        for card in cards:
            text = (card.get('summary', '') + ' ' +
                    ' '.join(s.get('summary', '') for s in card.get('sections', []))).lower()

            if 'conservative' in text or 'low risk' in text:
                if rp <= 0.02:
                    scores.append(1.0)
                    notes.append(f"Risk {rp:.1%} aligns with conservative research guidance.")
                else:
                    scores.append(0.4)
                    notes.append(f"Research favors lower risk; current {rp:.1%} may be aggressive.")

            if 'stop loss' in text or 'drawdown control' in text:
                if 0.01 <= sl <= 0.05:
                    scores.append(1.0)
                    notes.append(f"Stop-loss {sl:.1%} is within research-validated 1-5% range.")
                else:
                    scores.append(0.5)
                    notes.append(f"Stop-loss {sl:.1%} outside typical 1-5% range in literature.")

            if 'reward' in text and 'risk' in text:
                if rrr >= 1.5:
                    scores.append(1.0)
                    notes.append(f"Reward/risk ratio {rrr:.1f}× meets research benchmarks (≥1.5×).")
                else:
                    scores.append(0.3)
                    notes.append(f"Reward/risk ratio {rrr:.1f}× below recommended ≥1.5×.")

        if not notes:
            notes.append("No directly matching research found; proceeding with current parameters.")

        alignment_score = round(sum(scores) / len(scores), 4) if scores else 0.5
        return {'alignment_score': alignment_score, 'notes': notes}

    def cache_age_hours(self) -> Optional[float]:
        """Returns how many hours ago the cache was last updated, or None."""
        updated_at = self._cache.get('updated_at')
        if not updated_at:
            return None
        from datetime import datetime, timezone
        try:
            ts = datetime.fromisoformat(updated_at).replace(tzinfo=timezone.utc)
            delta = datetime.now(tz=timezone.utc) - ts
            return round(delta.total_seconds() / 3600, 2)
        except Exception:
            return None

    def refresh_cache(self) -> None:
        """Reload the research cache from disk."""
        self._load_cache()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _load_cache(self) -> None:
        if self.cache_path.exists():
            try:
                with open(self.cache_path) as f:
                    self._cache = json.load(f)
                n = len(self._cache.get('cards', []))
                age = self.cache_age_hours()
                age_str = f"{age:.1f}h ago" if age is not None else "unknown age"
                logger.info(f"[QuantMind] Loaded {n} research cards (updated {age_str}).")
            except Exception as exc:
                logger.warning(f"[QuantMind] Could not load research cache: {exc}")
                self._cache = {'cards': [], 'updated_at': None}
        else:
            logger.info("[QuantMind] No research cache found. Run research_updater.py to build one.")
            self._cache = {'cards': [], 'updated_at': None}

    def _keywords_for(self, symbol: str) -> list[str]:
        base = symbol.split('/')[0].upper()
        return _SYMBOL_KEYWORDS.get(base, [base.lower()])

    def _relevant_cards(self, keywords: list[str]) -> list[dict]:
        kw_lower = [k.lower() for k in keywords]
        result = []
        for card in self._cache.get('cards', []):
            text = (card.get('summary', '') + ' ' +
                    ' '.join(s.get('summary', '') for s in card.get('sections', []))).lower()
            if any(kw in text for kw in kw_lower):
                result.append(card)
        return result

    def _score_text(self, text: str, keywords: list[str]) -> Optional[float]:
        """
        Lexical sentiment score for a text snippet.
        Returns a float in [-1, 1] or None if the text isn't relevant.
        """
        lower = text.lower()
        if not any(kw.lower() in lower for kw in keywords):
            return None
        pos = sum(1 for w in _POSITIVE_WORDS if w in lower)
        neg = sum(1 for w in _NEGATIVE_WORDS if w in lower)
        if pos + neg == 0:
            return None
        return (pos - neg) / (pos + neg)
