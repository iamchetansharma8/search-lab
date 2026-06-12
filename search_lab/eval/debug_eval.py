"""Per-query eval debug: for one config, print where each query's relevant
chunk ranked, and WHY a result did or didn't count as relevant (coverage %).

This answers "is a 0-score a bad label or a real retrieval miss?" by showing,
for every golden query:
  - the top-N retrieved chunks with their (page, char span)
  - which labeled span(s) they were tested against
  - the best coverage achieved, and whether it cleared the threshold

Run:
    uv run python -m search_lab.eval.debug_eval
"""

from __future__ import annotations

import chromadb

from search_lab.embed.models import EmbedModelSpec
from search_lab.embed.store import collection_name_for
from search_lab.eval.__main__ import (
    CHROMA_PATH,
    GOLDEN_PATH,
    HF_NAME,
    MIN_COVERAGE,
    STORE_NAME,
    TOP_K,
)
from search_lab.eval.golden import load_golden_set
from search_lab.eval.metrics import is_relevant
from search_lab.search.index_opensearch import get_client, index_name_for
from search_lab.search.modes import SearchMode
from search_lab.search.retriever import Retriever


def best_coverage(result, query, min_coverage):
    """Return (best_cov, gold_span) over all the query's spans, for display."""
    best = 0.0
    best_span = None
    for span in query.relevant:
        if result.page != span.page:
            continue
        overlap = min(result.char_end, span.char_end) - max(
            result.char_start, span.char_start
        )
        if overlap <= 0:
            continue
        gold_len = span.char_end - span.char_start
        cov = overlap / gold_len if gold_len > 0 else 0.0
        if cov > best:
            best, best_span = cov, span
    return best, best_span


def main() -> None:
    queries = load_golden_set(GOLDEN_PATH)
    embed_spec = EmbedModelSpec(HF_NAME)
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    os_client = get_client()

    # Debug the dense baseline on the fixed collection.
    strategy, mode = "fixed", SearchMode.DENSE
    retriever = Retriever(
        collection=chroma.get_collection(
            name=collection_name_for(STORE_NAME, strategy, HF_NAME)
        ),
        embed_spec=embed_spec,
        os_client=os_client,
        os_index=index_name_for(STORE_NAME, strategy),
    )

    print(f"\n=== DEBUG: {strategy}/{mode.value} | min_coverage={MIN_COVERAGE} ===\n")

    for q in queries:
        results = retriever.search(q.query, mode=mode, top_k=TOP_K)
        first_hit = next(
            (r.rank for r in results if is_relevant(r, q, MIN_COVERAGE)), None
        )

        print(f"[{q.qid}] {q.query}")
        print(
            "     labeled spans: "
            + ", ".join(f"p{s.page}:{s.char_start}-{s.char_end}" for s in q.relevant)
        )
        verdict = (
            f"first relevant at rank {first_hit}"
            if first_hit
            else f">>> NO RELEVANT HIT in top {TOP_K} <<<"
        )
        print(f"     {verdict}")

        # Show top 5 with coverage so you can see near-misses.
        for r in results[:5]:
            cov, span = best_coverage(r, q, MIN_COVERAGE)
            mark = "OK " if cov >= MIN_COVERAGE else "   "
            note = ""
            if span is None and any(r.page == s.page for s in q.relevant):
                note = " (same page, no char overlap)"
            elif span is None:
                note = f" (page {r.page}, not a labeled page)"
            print(
                f"       {mark} rank{r.rank} {r.id} p{r.page}:{r.char_start}-{r.char_end} "
                f"cov={cov:.2f}{note}"
            )
        print()


if __name__ == "__main__":
    main()
