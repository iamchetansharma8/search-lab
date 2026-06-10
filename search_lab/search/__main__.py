from search_lab.embed.models import EmbedModelSpec
from search_lab.embed.store import chroma_client, collection_name_for
from search_lab.search.retrieve import query_collection


def main():
    spec = EmbedModelSpec("all-MiniLM-L6-v2")
    client = chroma_client("chroma")

    query = "What are the rights of a data principal?"

    for strategy in ("fixed", "recursive"):
        name = collection_name_for("dpdp", strategy, spec.hf_name)
        collection = client.get_collection(name=name)
        hits = query_collection(collection, query, spec, k=3)

        print(f"\n=== {name} ===")
        print(f"query: {query!r}")
        for i, h in enumerate(hits, 1):
            m = h["metadata"]
            preview = h["text"][:80].replace("\n", " ")
            print(
                f"  {i}. dist={h['distance']:.4f} "
                f"p{m['page_number']} start={m['char_start']} [{m['strategy']}]"
            )
            print(f"     {preview!r}")


if __name__ == "__main__":
    main()
