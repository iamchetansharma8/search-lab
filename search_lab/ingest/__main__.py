from search_lab.ingest.chunkers import fixed_size_chunker, recursive_chunker
from search_lab.ingest.loader import load_pdf


def main():
    pages = load_pdf("data/dpdp_act_2023.pdf")
    fixed = fixed_size_chunker(pages, 800, 100)
    recursive = recursive_chunker(pages, 800, 100)
    print(f"fixed: {len(fixed)} chunks | recursive: {len(recursive)} chunks")
    for c in fixed[:3]:
        print(f"  {c.id} p{c.page} start={c.char_start} len={len(c)}")
    for c in recursive[:3]:
        print(f"  {c.id} p{c.page} start={c.char_start} len={len(c)}")


if __name__ == "__main__":
    main()
    