"""Local cross-encoder reranker (zero-spend, runs on the sentence-transformers
stack). Cross-encoder reads (query, doc) jointly → far better relevance than
the bi-encoder retrieval, but too slow for the whole corpus — so it only
reorders the candidate set retrieval already narrowed down.
"""

from sentence_transformers import CrossEncoder

from search_lab.rerank.base import Reranker
from search_lab.search.models import SearchResult

# Purpose-built MS MARCO passage reranker. Raw-logit scores, higher = better.
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


class LocalReranker(Reranker):
    name = "local"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        # Load once, reuse across calls (like EmbedModelSpec).
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        if not results:
            return []

        # 1. Build pairs, aligned by index with `results`.
        pairs = [(query, r.text) for r in results]

        # 2. Batch-score all pairs at once -> numpy array of logits.
        scores = self.model.predict(pairs)

        # 3. Pair each result with its score, sort by score desc.
        scored = sorted(
            zip(results, scores), key=lambda rs: rs[1], reverse=True
        )

        # 4. Take top_k, emit fresh SearchResults with new score/rank, mode carried forward.
        reranked = []
        for i, (r, score) in enumerate(scored[:top_k]):
            reranked.append(
                SearchResult(
                    id=r.id,
                    text=r.text,
                    score=float(score),  # numpy float → Python float
                    rank=i + 1,
                    page=r.page,
                    char_start=r.char_start,
                    char_end=r.char_end,
                    strategy=r.strategy,
                    mode=r.mode,
                    reranker=self.name,
                )
            )
        return reranked