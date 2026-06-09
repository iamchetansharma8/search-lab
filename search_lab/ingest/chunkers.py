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
    length_function: Callable[[str], int] = len,
) -> list[Chunk]:
    _validate_chunk_params(chunk_size, overlap)
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

def recursive_chunker(
    pages: list[tuple[int, str]],
    chunk_size: int,
    overlap: int,
    length_function: Callable[[str], int] = len,
) -> list[Chunk]:
    _validate_chunk_params(chunk_size, overlap)
    chunks: list[Chunk] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=length_function,
        add_start_index=True,
    )
    i = 1
    for page_number, page_text in pages:
        docs = splitter.create_documents([page_text])
        for doc in docs:
            chunk_text = doc.page_content
            char_start = doc.metadata["start_index"]
            chunks.append(
                Chunk(
                    id=f"recursive-{i:04d}",
                    text=chunk_text,
                    page=page_number,
                    strategy="recursive",
                    char_start=char_start,
                )
            )
            i += 1
        logger.debug("chunked page", page_number=page_number)
    logger.info("chunking complete", total_pages=len(pages), total_chunks=len(chunks))
    return chunks