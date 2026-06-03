"""
Daily research ingestion pipeline  (#2).

Searches arXiv for recent crypto/quant papers, processes them through
quant-mind's paper_flow, and writes the extracted knowledge to
data/research_cache.json for the trading bot to consume.

Usage:
  python -m scripts.research_updater                          # defaults
  python -m scripts.research_updater --max-papers 20         # more papers
  python -m scripts.research_updater --days-back 14          # wider window

Cron example (run at midnight):
  0 0 * * * cd /app && python -m scripts.research_updater >> logs/research.log 2>&1
"""
import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import arxiv
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

CRYPTO_QUERIES = [
    'cryptocurrency trading machine learning',
    'bitcoin price prediction deep learning',
    'ethereum quantitative strategy',
    'crypto momentum factor alpha',
    'digital asset volatility forecasting',
    'crypto sentiment analysis trading',
    'algorithmic trading cryptocurrency neural',
]


def _search_arxiv(queries: list[str], max_results: int, days_back: int) -> list[str]:
    """Return de-duplicated arXiv IDs (without version suffix) from recent papers."""
    client = arxiv.Client()
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days_back)
    ids: set[str] = set()

    for query in queries:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        try:
            for result in client.results(search):
                if result.published >= cutoff:
                    raw_id = result.entry_id.split('/')[-1]   # e.g. "2401.12345v2"
                    ids.add(raw_id.split('v')[0])              # strip version → "2401.12345"
        except Exception as exc:
            logger.warning(f"arXiv search failed for '{query}': {exc}")

    return sorted(ids)


async def _ingest_papers(arxiv_ids: list[str], concurrency: int) -> list[dict]:
    """
    Run quant-mind paper_flow on each arXiv ID concurrently.
    Returns a list of serialised card dicts ready for the JSON cache.
    """
    from quantmind.flows.paper import paper_flow, PaperFlowCfg
    from quantmind.flows.batch import batch_run

    # quant-mind accepts plain strings as ArxivIdentifier-compatible input
    # (the flow dispatches on the value shape internally)
    inputs = [{'arxiv_id': id_} for id_ in arxiv_ids]
    cfg = PaperFlowCfg(extract_methodology=True, extract_limitations=True)

    processed = 0

    def on_progress(done: int, total: int) -> None:
        nonlocal processed
        processed = done
        logger.info(f"  [{done}/{total}] papers processed …")

    result = await batch_run(
        paper_flow,
        inputs,
        cfg=cfg,
        concurrency=concurrency,
        on_error='skip',
        on_progress=on_progress,
    )

    logger.info(
        f"Batch complete: {result.success_count} succeeded, "
        f"{result.failure_count} failed in {result.duration_seconds:.1f}s"
    )

    cards: list[dict] = []
    for _, paper in result.successes:
        card = _serialise_paper(paper)
        if card:
            cards.append(card)

    return cards


def _serialise_paper(paper) -> dict | None:
    """
    Convert a quant-mind Paper (TreeKnowledge) to a flat dict for the cache.
    Walks the tree and collects {title, summary} for every node.
    """
    try:
        sections: list[dict] = []
        for node in paper.walk_dfs():
            sections.append({
                'title': node.title or '',
                'summary': node.summary or '',
            })

        root_summary = paper.root().summary if paper.root() else ''

        return {
            'arxiv_id': getattr(paper, 'arxiv_id', None),
            'authors': list(getattr(paper, 'authors', [])),
            'asset_classes': list(getattr(paper, 'asset_classes', [])),
            'summary': root_summary,
            'sections': sections,
            'ingested_at': datetime.utcnow().isoformat(),
        }
    except Exception as exc:
        logger.warning(f"Failed to serialise paper: {exc}")
        return None


def _merge_into_cache(cache_path: Path, new_cards: list[dict]) -> dict:
    """Load existing cache, merge new cards (keyed by arxiv_id), and return merged dict."""
    existing: dict = {'cards': [], 'updated_at': None}
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                existing = json.load(f)
        except Exception as exc:
            logger.warning(f"Could not read existing cache ({exc}); starting fresh.")

    by_id: dict[str, dict] = {
        c['arxiv_id']: c
        for c in existing.get('cards', [])
        if c.get('arxiv_id')
    }
    for card in new_cards:
        if card.get('arxiv_id'):
            by_id[card['arxiv_id']] = card

    return {
        'updated_at': datetime.utcnow().isoformat(),
        'cards': list(by_id.values()),
    }


async def run(config: dict, max_papers: int, days_back: int) -> None:
    qm_cfg = config.get('quantmind', {})
    cache_path = Path(qm_cfg.get('cache_path', 'data/research_cache.json'))
    concurrency = qm_cfg.get('ingestion_concurrency', 3)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Searching arXiv (last {days_back} days, up to {max_papers} results/query)…")
    arxiv_ids = _search_arxiv(CRYPTO_QUERIES, max_results=max_papers, days_back=days_back)
    logger.info(f"Found {len(arxiv_ids)} unique papers: {arxiv_ids}")

    if not arxiv_ids:
        logger.warning("No papers found. Cache unchanged.")
        return

    logger.info(f"Ingesting via quant-mind (concurrency={concurrency})…")
    new_cards = await _ingest_papers(arxiv_ids, concurrency)
    logger.info(f"Extracted {len(new_cards)} paper cards.")

    cache = _merge_into_cache(cache_path, new_cards)
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)

    logger.info(
        f"Cache saved to {cache_path} "
        f"({len(cache['cards'])} total cards, updated {cache['updated_at']})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Daily quant-mind research ingestion')
    parser.add_argument('--config', default='config.yaml', help='Path to config.yaml')
    parser.add_argument('--max-papers', type=int, default=15,
                        help='Max arXiv results per query (default: 15)')
    parser.add_argument('--days-back', type=int, default=7,
                        help='Only ingest papers published within N days (default: 7)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    asyncio.run(run(config, args.max_papers, args.days_back))


if __name__ == '__main__':
    main()
