"""Relevance matching + retrieval metrics.

A retrieved result is relevant to a query if it covers >= `min_coverage` of
ANY one of that query's labeled spans (same page, char-span overlap). Metrics
operate on a single ranked list of results per query:

  - MRR: mean of 1/rank-of-first-relevant (0 if none in the list)
  - Hit@k: fraction of queries with >=1 relevant result in the top k

Both are deterministic — no LLM judge — which is the whole point: this mini
evaluates RETRIEVAL, and ranks against hand-labeled spans need no model.
"""

from __future__ import annotations

from search_lab.eval.golden import GoldenQuery, RelevantSpan
from search_lab.search.models import SearchResult


def covers_span(result: SearchResult, span: RelevantSpan, min_coverage: float) -> bool:
    """True if `result` covers >= min_coverage of `span` (same page)."""
    if result.page != span.page:
        return False
    overlap = min(result.char_end, span.char_end) - max(
        result.char_start, span.char_start
    )
    if overlap <= 0:
        return False
    gold_len = span.char_end - span.char_start
    if gold_len <= 0:  # defensive; loader already forbids this
        return False
    return (overlap / gold_len) >= min_coverage


def is_relevant(result: SearchResult, query: GoldenQuery, min_coverage: float) -> bool:
    """Relevant if the result covers ANY one of the query's labeled spans."""
    return any(covers_span(result, span, min_coverage) for span in query.relevant)


def first_relevant_rank(
    results: list[SearchResult], query: GoldenQuery, min_coverage: float
) -> int | None:
    """1-based rank of the first relevant result, or None if none are relevant.

    Uses the result's own `rank` field rather than list position, so it's
    correct even if a caller passes a reordered/sliced list.
    """
    for r in results:
        if is_relevant(r, query, min_coverage):
            return r.rank
    return None


def reciprocal_rank(
    results: list[SearchResult], query: GoldenQuery, min_coverage: float
) -> float:
    """1/rank of first relevant result; 0.0 if none found."""
    rank = first_relevant_rank(results, query, min_coverage)
    return 0.0 if rank is None else 1.0 / rank


def hit_at_k(
    results: list[SearchResult], query: GoldenQuery, k: int, min_coverage: float
) -> bool:
    """True if a relevant result appears within the top k by rank."""
    rank = first_relevant_rank(results, query, min_coverage)
    return rank is not None and rank <= k


def mrr(
    per_query_results: dict[str, list[SearchResult]],
    queries: list[GoldenQuery],
    min_coverage: float,
) -> float:
    """Mean reciprocal rank across all queries."""
    if not queries:
        return 0.0
    total = sum(
        reciprocal_rank(per_query_results[q.qid], q, min_coverage) for q in queries
    )
    return total / len(queries)


def hit_rate_at_k(
    per_query_results: dict[str, list[SearchResult]],
    queries: list[GoldenQuery],
    k: int,
    min_coverage: float,
) -> float:
    """Fraction of queries with a relevant result in the top k."""
    if not queries:
        return 0.0
    hits = sum(hit_at_k(per_query_results[q.qid], q, k, min_coverage) for q in queries)
    return hits / len(queries)
