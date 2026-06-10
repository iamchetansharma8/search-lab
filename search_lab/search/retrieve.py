from typing import Any

from search_lab.embed.models import EmbedModelSpec


def query_collection(
    collection: Any,
    query: str,
    embed_model_spec: EmbedModelSpec,
    k: int = 3,
) -> list[dict]:
    query_embedding = embed_model_spec.model.encode(
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
