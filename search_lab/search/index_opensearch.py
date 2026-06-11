"""Build the OpenSearch (BM25) index for a chunk strategy.

Reads chunks from the corresponding Chroma collection (single source of
truth) and bulk-indexes them into a per-strategy OpenSearch index, mirroring
the Chroma orchestration. Sparse retrieval searches the exact same units as
dense, keeping the dense/sparse/hybrid comparison apples-to-apples.
"""

from opensearchpy import OpenSearch, helpers

from search_lab.ingest.models import Chunk

# BM25 is OpenSearch's default similarity for `text` fields, so no explicit
# similarity config is needed. Keys mirror Chroma's metadata exactly.
MAPPING = {
    "mappings": {
        "properties": {
            "text": {"type": "text"},  # analyzed -> BM25
            "page_number": {"type": "integer"},
            "char_start": {"type": "integer"},
            "strategy": {"type": "keyword"},
        }
    }
}


def get_client() -> OpenSearch:
    """Local single-node client. Security plugin is disabled, so plain HTTP."""
    return OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],
        use_ssl=False,
    )


def index_name_for(store_name: str, strategy: str) -> str:
    # No hf_name tag: a lexical index has no embedding model.
    return f"{store_name}_{strategy}"


def create_index(client: OpenSearch, name: str) -> None:
    """Drop-and-recreate for deterministic, idempotent rebuilds.

    Destructive by design: any existing index of this name is deleted first.
    """
    if client.indices.exists(index=name):
        client.indices.delete(index=name)
    client.indices.create(index=name, body=MAPPING)


def index_chunks(client: OpenSearch, name: str, chunks: list[Chunk]) -> int:
    """Bulk-index chunks; chunk.id becomes the OpenSearch _id. Returns count."""
    actions = [
        {
            "_index": name,
            "_id": chunk.id,
            "_source": {
                "text": chunk.text,
                "page_number": chunk.page,
                "char_start": chunk.char_start,
                "strategy": chunk.strategy,
            },
        }
        for chunk in chunks
    ]
    helpers.bulk(client, actions)
    # Force refresh so docs are searchable immediately (OpenSearch refreshes
    # asynchronously by default, which would race a sanity query right after).
    client.indices.refresh(index=name)
    return len(actions)


def orchestrate_opensearch_index_for_strategy(
    client: OpenSearch,
    chunks: list[Chunk],
    strategy: str,
    store_name: str,
) -> int:
    """Mirror of orchestrate_chroma_store_for_strategy, for the BM25 index."""
    name = index_name_for(store_name, strategy)
    create_index(client, name)
    return index_chunks(client, name, chunks)

if __name__ == "__main__":
    import chromadb

    from search_lab.embed.store import chunks_from_collection, collection_name_for

    STORE_NAME = "dpdp"
    HF_NAME = "all-MiniLM-L6-v2"
    STRATEGIES = ["fixed", "recursive"]

    chroma = chromadb.PersistentClient(path="chroma")
    client = get_client()

    for strategy in STRATEGIES:
        collection = chroma.get_collection(
            name=collection_name_for(STORE_NAME, strategy, HF_NAME)
        )
        chunks = chunks_from_collection(collection)
        count = orchestrate_opensearch_index_for_strategy(
            client, chunks, strategy, STORE_NAME
        )
        print(f"indexed {count} chunks into '{index_name_for(STORE_NAME, strategy)}'")