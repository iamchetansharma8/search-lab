from typing import Callable

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter

from search_lab.ingest.errors import InvalidParamError
from search_lab.ingest.models import Chunk

logger = structlog.get_logger()

def _validate_chunk_params(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise InvalidParamError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise InvalidParamError(f"overlap must be non-negative, got {overlap}")
    if overlap >= chunk_size:
        raise InvalidParamError(
            f"overlap ({overlap}) must be < chunk_size ({chunk_size})"
        )

def fixed_size_chunker(
    pages: list[tuple[int, str]],
    chunk_size: int,
    overlap: int,
    tokenizer,
    content_budget: int,
) -> list[Chunk]:
    """Fixed-size chunking in TOKEN space, with char-offset provenance.

    Slices the page's token-id list into windows of `chunk_size` tokens
    (stride = chunk_size - overlap). Each chunk's text is recovered by
    slicing the ORIGINAL page_text via the tokenizer's offset mapping —
    so chunk_text is verbatim source and char_start is exact.
    """
    _validate_chunk_params(chunk_size, overlap)
    # Self-protection: a window can never exceed what the model can embed.
    if chunk_size > content_budget:
        raise InvalidParamError(
            f"chunk_size {chunk_size} exceeds content_budget {content_budget}"
        )

    chunks: list[Chunk] = []
    i = 1
    stride = chunk_size - overlap

    for page_number, page_text in pages:
        # Encode once per page. offsets[k] = (char_start, char_end) of token k
        # in page_text. add_special_tokens=False so ids/offsets are content-only.
        # Note: transformers prints "Token indices sequence length is longer
        # than the specified maximum (N > 256)" here — it encodes the whole
        # page at once and assumes the full sequence goes to the model. It does
        # not: we slice `ids` into <=chunk_size windows below before any embed
        # call, so no indexing error occurs. Warning is expected and harmless.
        enc = tokenizer(
            page_text, add_special_tokens=False, return_offsets_mapping=True
        )
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        n_tokens = len(ids)

        start_index = 0
        while start_index < n_tokens:
            end_index = start_index + chunk_size  # exclusive, in TOKEN units
            window = ids[start_index:end_index]  # may be short on final window

            # Provenance: first token's char start .. last token's char end.
            # Use len(window) (not chunk_size) so the final short window is correct.
            last = start_index + len(window) - 1
            char_start = offsets[start_index][0]
            char_end = offsets[last][1]

            # Verbatim slice of the ORIGINAL text — not a decode of the ids.
            chunk_text = page_text[char_start:char_end]

            chunks.append(
                Chunk(
                    id=f"fixed-{i:04d}",
                    text=chunk_text,
                    page=page_number,
                    strategy="fixed",
                    char_start=char_start,
                )
            )
            i += 1

            if end_index >= n_tokens:  # this window hit the end of the page
                break
            start_index += stride  # advance in TOKEN units

        logger.debug("chunked page", page_number=page_number)

    logger.info("chunking complete", total_pages=len(pages), total_chunks=len(chunks))
    return chunks

def recursive_chunker(
    pages: list[tuple[int, str]],
    chunk_size: int,
    overlap: int,
    length_function: Callable[[str], int],
    content_budget: int,
) -> list[Chunk]:
    """Recursive (separator-aware) chunking, token-sized via length_function.

    LangChain decides split points on natural separators; we compute char_start
    ourselves (its add_start_index returns -1 when overlap/whitespace breaks its
    internal search). Each chunk is checked against content_budget — overflow is
    rare (char-level fallback usually prevents it) but logged if it ever happens.
    """
    _validate_chunk_params(chunk_size, overlap)
    chunks: list[Chunk] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=length_function,
        # add_start_index dropped: its offset math subtracts token-unit overlap from a
        # char-unit length, so text.find() starts past the real position and returns -1
        # whenever length_function is token-based and overlap > 0 (our exact config).
    )
    i = 1
    for page_number, page_text in pages:
        pieces = splitter.split_text(page_text)

        # Locate each piece in the original page to recover char_start.
        # Search forward from a cursor so repeated text matches the right spot;
        # account for overlap by stepping the cursor back by `overlap`-ish room.
        search_from = 0
        for piece in pieces:
            char_start = page_text.find(piece, search_from)
            if char_start == -1:
                # Fallback: piece text was altered by the splitter (whitespace
                # at separator boundaries). Search from the very start as a
                # last resort; log so we know provenance may be approximate.
                char_start = page_text.find(piece)
                logger.warning(
                    "recursive char_start fallback",
                    page_number=page_number,
                    chunk_index=i,
                )
            if char_start == -1:
                # Still not found — provenance genuinely unknown. Record -1
                # and flag loudly rather than silently shipping a wrong offset.
                logger.error(
                    "recursive char_start not found",
                    page_number=page_number,
                    chunk_index=i,
                )
                char_start = -1

            # Token-budget guard. Should essentially never fire with default
            # separators (char-level split prevents it), but a custom separator
            # set or pathological unit could overflow → log, don't crash.
            tok_len = length_function(piece)
            if tok_len > content_budget:
                logger.warning(
                    "recursive chunk exceeds content_budget",
                    page_number=page_number,
                    chunk_index=i,
                    token_len=tok_len,
                    content_budget=content_budget,
                )

            chunks.append(
                Chunk(
                    id=f"recursive-{i:04d}",
                    text=piece,
                    page=page_number,
                    strategy="recursive",
                    char_start=char_start,
                )
            )
            i += 1

            # Advance cursor past this piece's start so the next find() doesn't
            # re-match earlier text. Move to char_start+1 (not +len) because
            # overlap means the next piece begins *before* this one ends.
            if char_start >= 0:
                search_from = char_start + 1

        logger.debug("chunked page", page_number=page_number)

    logger.info("chunking complete", total_pages=len(pages), total_chunks=len(chunks))
    return chunks