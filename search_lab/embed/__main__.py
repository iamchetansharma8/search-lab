from search_lab.embed.models import EmbedModelSpec
from search_lab.embed.store import orchestrate_chroma_store_for_strategy
from search_lab.ingest.chunkers import fixed_size_chunker, recursive_chunker
from search_lab.ingest.loader import load_pdf


def main():
    spec = EmbedModelSpec("all-MiniLM-L6-v2")
    pages = load_pdf("data/dpdp_act_2023.pdf")

    fixed = fixed_size_chunker(
        pages, 240, 35, spec.model.tokenizer, spec.content_budget
    )
    recursive = recursive_chunker(
        pages, 240, 35, spec.length_function, spec.content_budget
    )

    print(f"fixed: {len(fixed)} chunks | recursive: {len(recursive)} chunks")

    for strategy, chunks in (("fixed", fixed), ("recursive", recursive)):
        stored = orchestrate_chroma_store_for_strategy(
            chunks=chunks,
            embed_model_spec=spec,
            strategy=strategy,
            store_name="dpdp",
        )
        print(f"sent {len(chunks)} {strategy} chunks | collection.count()={stored}")


if __name__ == "__main__":
    main()
