from dataclasses import dataclass
from typing import Any

from search_lab.embed.models import EmbedModelSpec
from search_lab.search.models import SearchResult
from search_lab.search.modes import SearchMode

RRF_K = 60  # Reciprocal Rank Fusion constant (standard default from the RRF paper).
# Dampens top-rank dominance: rank 1 contributes 1/(60+1), etc.
@dataclass
class Retriever:
    collection: Any
    embed_spec: EmbedModelSpec
    os_client: Any
    os_index: str

    def search(
        self, query: str, mode: SearchMode, top_k: int = 10, fetch_depth: int = 20
    ) -> list[SearchResult]:
        match mode:
            case SearchMode.DENSE:
                return self._dense(query, top_k)
            case SearchMode.SPARSE:
                return self._sparse(query, top_k)
            case SearchMode.HYBRID:
                return self._hybrid(query, top_k, fetch_depth)

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
                    char_end=meta["char_end"],
                    strategy=meta["strategy"],
                    mode=SearchMode.DENSE,
                )
            )
        return search_results

    def _sparse(self, query: str, top_k: int) -> list[SearchResult]:
        # Sparse needs the OpenSearch handles; a dense-only Retriever is built
        # with these as None, so fail loudly rather than AttributeError on None.
        if self.os_client is None or self.os_index is None:
            raise ValueError("sparse search requires os_client and os_index")
    
        # `match` on the analyzed `text` field => BM25 scoring (OpenSearch default).
        res = self.os_client.search(
            index=self.os_index,
            body={"query": {"match": {"text": query}}, "size": top_k},
        )
    
        # OpenSearch returns a list of per-hit dicts (not parallel arrays like
        # Chroma), so we enumerate directly — no [0] batch unwrap needed.
        search_results: list[SearchResult] = []
        for i, hit in enumerate(res["hits"]["hits"]):
            src = hit["_source"]
            search_results.append(
                SearchResult(
                    id=hit["_id"],
                    text=src["text"],
                    score=hit["_score"],  # raw BM25, already higher-is-better
                    rank=i + 1,
                    page=src["page_number"],  # OS key -> SearchResult.page
                    char_start=src["char_start"],
                    char_end=src["char_end"],
                    strategy=src["strategy"],
                    mode=SearchMode.SPARSE,
                )
            )
        return search_results

    def _hybrid(
    self, query: str, top_k: int, fetch_depth: int = 20
) -> list[SearchResult]:
        # Pull a deeper slice from each mode than the final top_k, so a chunk
        # ranked low by one method but high by the other still reaches fusion.
        dense_hits = self._dense(query, fetch_depth)
        sparse_hits = self._sparse(query, fetch_depth)

        # Accumulate RRF score per chunk id, and keep one SearchResult per id to
        # carry its text/provenance forward. Same chunk from both lists is two
        # objects sharing an id — we fuse on id, not object identity.
        rrf_scores: dict[str, float] = {}
        by_id: dict[str, SearchResult] = {}
        for hits in (dense_hits, sparse_hits):
            for hit in hits:
                rrf_scores[hit.id] = rrf_scores.get(hit.id, 0.0) + 1.0 / (RRF_K + hit.rank)
                by_id.setdefault(hit.id, hit)  # first sighting wins; chunk data is identical

        # Sort ids by fused score (desc), take final top_k.
        ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        # Re-emit as HYBRID results: score = RRF score, rank = fresh 1..top_k.
        results: list[SearchResult] = []
        for i, cid in enumerate(ranked_ids):
            src = by_id[cid]
            results.append(
                SearchResult(
                    id=src.id,
                    text=src.text,
                    score=rrf_scores[cid],  # RRF score is what produced this ranking
                    rank=i + 1,
                    page=src.page,
                    char_start=src.char_start,
                    char_end=src.char_end,
                    strategy=src.strategy,
                    mode=SearchMode.HYBRID,
                )
            )
        return results