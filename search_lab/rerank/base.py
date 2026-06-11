"""Reranker interface — the swappable seam for the rerank stage.

A reranker takes a query + candidate results (from any retrieval mode) and
returns a re-scored, re-ordered, trimmed list. Local cross-encoder and Cohere
both implement this, so the eval can swap them like embedding models.
"""

from abc import ABC, abstractmethod

from search_lab.search.models import SearchResult


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        """Re-score `results` for `query`, return the top_k reordered."""
        ...
