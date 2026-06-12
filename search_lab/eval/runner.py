"""Eval runner: score retrieval configurations against the golden set.

A configuration is (chunking strategy x search mode x reranker). For each
config we run all golden queries through retrieve [-> rerank], collect one
ranked result list per query, then compute MRR + Hit@k. Results print as a
comparison table with the baseline delta highlighted.

The config list is assembled explicitly by the caller (see __main__) — nothing
is hardcoded here, so adding a config or a second embedding model is one line.
"""

from __future__ import annotations

from dataclasses import dataclass

from search_lab.eval.golden import GoldenQuery
from search_lab.eval.metrics import hit_rate_at_k, mrr
from search_lab.rerank.base import Reranker
from search_lab.search.models import SearchResult
from search_lab.search.modes import SearchMode
from search_lab.search.retriever import Retriever


@dataclass
class EvalConfig:
    """One cell of the comparison grid."""

    name: str  # human label for the table, e.g. "fixed/hybrid/cohere"
    retriever: Retriever  # already bound to a chunking strategy's collection + index
    mode: SearchMode
    reranker: Reranker | None = None  # None = retrieval-only (no rerank stage)


@dataclass
class EvalResult:
    """Scored metrics for one config."""

    name: str
    mrr: float
    hits: dict[int, float]  # k -> hit rate


def run_config(
    config: EvalConfig,
    queries: list[GoldenQuery],
    k_values: list[int],
    min_coverage: float,
    top_k: int,
) -> EvalResult:
    """Run every query through one config and score it."""
    per_query: dict[str, list[SearchResult]] = {}
    for q in queries:
        results = config.retriever.search(q.query, mode=config.mode, top_k=top_k)
        if config.reranker is not None:
            results = config.reranker.rerank(q.query, results, top_k=top_k)
        per_query[q.qid] = results

    return EvalResult(
        name=config.name,
        mrr=mrr(per_query, queries, min_coverage),
        hits={k: hit_rate_at_k(per_query, queries, k, min_coverage) for k in k_values},
    )


def run_eval(
    configs: list[EvalConfig],
    queries: list[GoldenQuery],
    k_values: list[int] = [1, 3, 5],
    min_coverage: float = 0.5,
    top_k: int = 10,
) -> list[EvalResult]:
    """Score every config. First config in the list is treated as the baseline."""
    return [run_config(c, queries, k_values, min_coverage, top_k) for c in configs]


def format_table(
    results: list[EvalResult], k_values: list[int], min_coverage: float
) -> str:
    """Render results as a fixed-width comparison table.

    The first result is the baseline; every other row shows its MRR delta
    against it (acceptance criterion: report the delta vs baseline).
    """
    if not results:
        return "(no results)"

    baseline = results[0]
    headers = ["config", "MRR", *[f"Hit@{k}" for k in k_values], "dMRR"]
    rows: list[list[str]] = []
    for i, r in enumerate(results):
        delta = r.mrr - baseline.mrr
        delta_str = "baseline" if i == 0 else f"{delta:+.3f}"
        rows.append(
            [
                r.name,
                f"{r.mrr:.3f}",
                *[f"{r.hits[k]:.3f}" for k in k_values],
                delta_str,
            ]
        )

    widths = [
        max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)
    ]
    sep = "  "

    def fmt(cols: list[str]) -> str:
        return sep.join(c.ljust(widths[i]) for i, c in enumerate(cols))

    lines = [
        f"min_coverage={min_coverage}  (relevance threshold)",
        fmt(headers),
        sep.join("-" * w for w in widths),
        *(fmt(row) for row in rows),
    ]
    return "\n".join(lines)
