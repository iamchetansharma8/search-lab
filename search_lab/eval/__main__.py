"""Eval entry point: assemble the config grid and print the comparison table.

Run:
    uv run python -m search_lab.eval

Builds one Retriever per chunking strategy (each bound to that strategy's
Chroma collection + OpenSearch index), then assembles an explicit list of
EvalConfigs across (strategy x mode x reranker). The FIRST config is the
baseline — every other row's MRR delta is measured against it.

Rerankers are constructed once and reused across configs. CohereReranker reads
COHERE_API_KEY from the environment, so load_dotenv() runs before it is built;
if the key is absent, Cohere configs are skipped (local + retrieval-only still
run) rather than crashing the whole eval.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from dotenv import load_dotenv

from search_lab.embed.models import EmbedModelSpec
from search_lab.embed.store import collection_name_for
from search_lab.eval.golden import load_golden_set
from search_lab.eval.runner import EvalConfig, format_table, run_eval
from search_lab.rerank.cohere_reranker import CohereReranker
from search_lab.rerank.local import LocalReranker
from search_lab.search.index_opensearch import get_client, index_name_for
from search_lab.search.modes import SearchMode
from search_lab.search.retriever import Retriever

# --- fixed run parameters -------------------------------------------------
STORE_NAME = "dpdp"
HF_NAME = "all-MiniLM-L6-v2"
CHROMA_PATH = "chroma"
GOLDEN_PATH = Path(__file__).parent / "golden_set.json"

K_VALUES = [1, 3, 5]
MIN_COVERAGE = 0.5
TOP_K = 10


def build_retriever(
    strategy: str, embed_spec: EmbedModelSpec, chroma, os_client
) -> Retriever:
    """One Retriever bound to a chunking strategy's collection + OS index."""
    collection = chroma.get_collection(
        name=collection_name_for(STORE_NAME, strategy, HF_NAME)
    )
    return Retriever(
        collection=collection,
        embed_spec=embed_spec,
        os_client=os_client,
        os_index=index_name_for(STORE_NAME, strategy),
    )


def main() -> None:
    load_dotenv()  # must precede CohereReranker() — it reads COHERE_API_KEY

    queries = load_golden_set(GOLDEN_PATH)

    embed_spec = EmbedModelSpec(HF_NAME)
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    os_client = get_client()

    retrievers = {
        s: build_retriever(s, embed_spec, chroma, os_client)
        for s in ("fixed", "recursive")
    }

    local = LocalReranker()
    try:
        cohere = CohereReranker(throttle_s=6.5)  # trial key: 10 calls/min
    except ValueError as e:
        print(f"[warn] Cohere disabled: {e}")
        cohere = None

    # --- the config grid ---------------------------------------------------
    # Explicit, ordered list. First entry = baseline. Edit freely to add or
    # drop cells; nothing here is generated implicitly.
    configs: list[EvalConfig] = [
        # baseline: fixed chunking, dense retrieval, no rerank
        EvalConfig("fixed/dense/none", retrievers["fixed"], SearchMode.DENSE),
        EvalConfig("fixed/sparse/none", retrievers["fixed"], SearchMode.SPARSE),
        EvalConfig("fixed/hybrid/none", retrievers["fixed"], SearchMode.HYBRID),
        EvalConfig("fixed/hybrid/local", retrievers["fixed"], SearchMode.HYBRID, local),
        EvalConfig("recursive/hybrid/none", retrievers["recursive"], SearchMode.HYBRID),
        EvalConfig(
            "recursive/hybrid/local", retrievers["recursive"], SearchMode.HYBRID, local
        ),
    ]
    # Cohere disabled during authoring to avoid trial-key rate limit (10/min).
    # Re-enable for the final 20-question run.
    if cohere is not None:
        configs.append(
            EvalConfig(
                "fixed/hybrid/cohere", retrievers["fixed"], SearchMode.HYBRID, cohere
            )
        )

    results = run_eval(configs, queries, K_VALUES, MIN_COVERAGE, TOP_K)
    print()
    print(format_table(results, K_VALUES, MIN_COVERAGE))


if __name__ == "__main__":
    main()
