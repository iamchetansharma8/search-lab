"""Cohere Rerank reranker (hosted cross-encoder, trial key = free tier).
Same Reranker interface as the local cross-encoder — swappable at eval time.
"""

import os

import cohere

from search_lab.rerank.base import Reranker
from search_lab.search.models import SearchResult

DEFAULT_MODEL = "rerank-v3.5"


class CohereReranker(Reranker):
    name = "cohere"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        # Entry point is responsible for load_dotenv(); we only read env here.
        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY not set in environment")
        self.client = cohere.ClientV2(api_key=api_key)
        self.model_name = model_name

    def rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        if not results:
            return []

        documents = [r.text for r in results]

        # Cohere returns results already sorted desc by relevance and trimmed
        # to top_n. Each item carries .index (into our `documents`/`results`)
        # and .relevance_score (0-1, higher = better).
        response = self.client.rerank(
            model=self.model_name,
            query=query,
            documents=documents,
            top_n=top_k,
        )

        reranked: list[SearchResult] = []
        for i, item in enumerate(response.results):
            src = results[item.index]  # map back to original result
            reranked.append(
                SearchResult(
                    id=src.id,
                    text=src.text,
                    score=item.relevance_score,  # already a float, 0-1
                    rank=i + 1,  # fresh 1..top_k (response is pre-sorted)
                    page=src.page,
                    char_start=src.char_start,
                    strategy=src.strategy,
                    mode=src.mode,  # carried forward
                    reranker=self.name,  # "cohere"
                )
            )
        return reranked
