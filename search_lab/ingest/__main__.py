from search_lab.embed.models import EmbedModelSpec
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

    # Provenance spot-check: does char_start actually point at the chunk text?
    for label, chunks in (("fixed", fixed), ("recursive", recursive)):
        print(f"\n--- {label} (first 3) ---")
        for c in chunks[:3]:
            page_text = next(t for p, t in pages if p == c.page)
            slice_preview = page_text[c.char_start : c.char_start + 40].replace(
                "\n", " "
            )
            chunk_preview = c.text[:40].replace("\n", " ")
            match = (
                "OK"
                if page_text[c.char_start :].startswith(c.text[:20])
                else "MISMATCH"
            )
            print(
                f"  {c.id} p{c.page} start={c.char_start} tok={spec.length_function(c.text)} [{match}]"
            )
            print(f"      from page : {slice_preview!r}")
            print(f"      chunk text: {chunk_preview!r}")


if __name__ == "__main__":
    main()
