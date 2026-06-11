# search_lab/search/__main__.py
import chromadb

from search_lab.embed.models import EmbedModelSpec
from search_lab.embed.store import collection_name_for
from search_lab.search.index_opensearch import get_client, index_name_for
from search_lab.search.modes import SearchMode
from search_lab.search.retriever import Retriever

os_client = get_client()
CHROMA_PATH = "chroma"
HF_NAME = "all-MiniLM-L6-v2"
STRATEGIES = ["fixed", "recursive"]
QUERY = "rights of a data principal"


def main() -> None:
    spec = EmbedModelSpec(hf_name=HF_NAME)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    for strategy in STRATEGIES:
        collection = client.get_collection(
            name=collection_name_for("dpdp", strategy, HF_NAME)
        )
        retriever = Retriever(
            collection=collection,
            embed_spec=spec,
            os_client=os_client,  # not needed for dense
            os_index=index_name_for("dpdp", strategy),
        )
        print(f"\n=== {strategy} | DENSE ===")
        hits = retriever.search(QUERY, SearchMode.DENSE, top_k=5)
        for h in hits:
            print(
                f"#{h.rank}  score={h.score:.4f}  p{h.page}  start={h.char_start}  [{h.strategy}]"
            )
            print(f"     {h.text[:120]}...")
        
        print(f"\n=== {strategy} | SPARSE ===")
        for h in retriever.search(QUERY, SearchMode.SPARSE, top_k=5):
            print(
                f"#{h.rank}  score={h.score:.4f}  p{h.page}  [{h.strategy}]  {h.text[:80]}..."
            )
        print(f"\n=== {strategy} | HYBRID ===")
        
        for h in retriever.search(QUERY, SearchMode.HYBRID, top_k=5):
            print(
                f"#{h.rank}  score={h.score:.4f}  p{h.page}  [{h.strategy}]  {h.text[:80]}..."
            )

if __name__ == "__main__":
    main()
