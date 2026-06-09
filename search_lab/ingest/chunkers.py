from typing import Callable

import structlog

from search_lab.ingest.errors import InvalidParamError
from search_lab.ingest.models import Chunk

logger = structlog.get_logger()

def fixed_size_chunker(
    pages: list[tuple[int, str]],
    chunk_size: int,
    overlap: int,
    length_function: Callable[[str], int] = len,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise InvalidParamError(f"failed to chunk, value of chunk_size should be positive, chunk_size={chunk_size}")
    if overlap < 0:
        raise InvalidParamError(f"failed to chunk, value of overlap should not be negative, overlap={overlap}")
    if overlap >= chunk_size:
        raise InvalidParamError(
            f'''failed to chunk, value of overlap should be less than chunk_size,
            chunk_size={chunk_size}, overlap={overlap}'''
        )
    chunks: list[Chunk] = []
    i = 1
    for page_number, page_text in pages:
        page_len = length_function(page_text)   # compute once, not per-iteration
        start_index = 0
        while start_index < page_len:
            end_index = start_index + chunk_size
            chunk_text = page_text[start_index:end_index]
            chunks.append(
                Chunk(id=f"fixed-{i:04d}", text=chunk_text, page=page_number, strategy="fixed", char_start=start_index)
            )
            i += 1
            if end_index >= page_len:        # this window reached the end — done with page
                break
            start_index += chunk_size - overlap   # advance by stride
        logger.debug("chunked page", page_number=page_number)
    logger.info("chunking complete", total_pages=len(pages), total_chunks=len(chunks))
    return chunks