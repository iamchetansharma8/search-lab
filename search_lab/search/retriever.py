from dataclasses import dataclass
from typing import Any

from search_lab.embed.models import EmbedModelSpec
from search_lab.search.models import SearchResult
from search_lab.search.modes import SearchMode


@dataclass
class Retriever:
    collection: Any
    embed_spec: EmbedModelSpec
    os_client: Any
    os_index: str

    def search(
        self, query: str, mode: SearchMode, top_k: int = 10
    ) -> list[SearchResult]:
        match mode:
            case SearchMode.DENSE:
                return self._dense(query, top_k)
            case SearchMode.SPARSE:
                return self._sparse(query, top_k)
            case SearchMode.HYBRID:
                return self._hybrid(query, top_k)

    def _dense(self, query: str, top_k: int) -> list[SearchResult]:
        query_embedding = self.embed_spec.model.encode(
            [query], normalize_embeddings=True
        ).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        search_results: list[SearchResult] = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            search_results.append(
                SearchResult(
                    id=results["ids"][0][i],
                    text=results["documents"][0][i],
                    score=1 - results["distances"][0][i],
                    rank=i + 1,
                    page=meta["page_number"],
                    char_start=meta["char_start"],
                    strategy=meta["strategy"],
                    mode=SearchMode.DENSE,
                )
            )
        return search_results

def query_collection(
    collection: Any,
    query: str,
    embed_spec: EmbedModelSpec,
    k: int = 3,
) -> list[dict]:
    query_embedding = embed_spec.model.encode(
        [query], normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    # query() batches by input query; we sent one, so index [0]
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({"text": doc, "metadata": meta, "distance": dist})
    return hits
