"""
Golden-set helper: locate a phrase in the DPDP PDF and print its
per-page character span, so you can paste (page, char_start, char_end)
into eval/golden_set.json.

Offsets are measured per-page, matching the chunker's char_start basis.
The page text comes from the SAME loader the chunkers use, so a span
printed here lines up exactly with retrieved chunks' provenance.

Usage:
    python find_span.py --page 11 --phrase "rights of a data principal"
    python find_span.py --phrase "data principal"          # search all pages
    python find_span.py --page 11 --phrase "..." --ignore-case
"""

from __future__ import annotations

import argparse

from search_lab.ingest.loader import load_pdf  # absolute import (run as module)

# Adjust if your PDF lives elsewhere.
DEFAULT_PDF = "data/dpdp_act_2023.pdf"


def find_in_page(text: str, phrase: str, ignore_case: bool) -> list[tuple[int, int]]:
    """Return [(char_start, char_end), ...] for every occurrence of phrase."""
    hay = text.lower() if ignore_case else text
    needle = phrase.lower() if ignore_case else phrase

    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        i = hay.find(needle, start)
        if i == -1:
            break
        spans.append((i, i + len(phrase)))
        start = i + 1  # allow overlapping matches
    return spans


def main() -> None:
    ap = argparse.ArgumentParser(description="Find a phrase's per-page char span.")
    ap.add_argument("--phrase", required=True, help="Text you know is in the answer.")
    ap.add_argument(
        "--page",
        type=int,
        default=None,
        help="1-indexed page to search. Omit to search all pages.",
    )
    ap.add_argument("--pdf", default=DEFAULT_PDF, help="Path to the PDF.")
    ap.add_argument(
        "--ignore-case", action="store_true", help="Case-insensitive search."
    )
    ap.add_argument(
        "--context",
        type=int,
        default=40,
        help="Chars of surrounding context to print (default 40).",
    )
    args = ap.parse_args()

    pages = load_pdf(args.pdf)  # list[(page_no, text)], 1-indexed page_no

    found_any = False
    for page_no, text in pages:
        if args.page is not None and page_no != args.page:
            continue

        for char_start, char_end in find_in_page(text, args.phrase, args.ignore_case):
            found_any = True
            lo = max(0, char_start - args.context)
            hi = min(len(text), char_end + args.context)
            snippet = (
                text[lo:char_start]
                + "["
                + text[char_start:char_end]
                + "]"
                + text[char_end:hi]
            )
            snippet = snippet.replace("\n", " ")
            print(f"page={page_no}  char_start={char_start}  char_end={char_end}")
            print(f"    …{snippet}…")
            print()

    if not found_any:
        scope = f"page {args.page}" if args.page is not None else "any page"
        print(f"No match for {args.phrase!r} on {scope}.")
        print("Try --ignore-case, a shorter phrase, or check the page number.")


if __name__ == "__main__":
    main()
