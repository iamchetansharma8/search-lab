"""PDF loading: turn a PDF file into per-page text with provenance intact."""

from pathlib import Path

import structlog
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from search_lab.ingest.errors import PdfLoadError

logger = structlog.get_logger()


def load_pdf(path: str | Path) -> list[tuple[int, str]]:
    """Extract text from a PDF, one (page_number, text) entry per page.

    Page numbers are 1-indexed and track the true PDF position.
    Empty pages (blank or image-only) are skipped and logged.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        PdfLoadError: if the file exists but cannot be opened or parsed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    try:
        reader = PdfReader(path)
        pages: list[tuple[int, str]] = []
        skipped = 0

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                skipped += 1
                logger.warning("empty_page_skipped", page=page_number)
                continue
            pages.append((page_number, page_text))
            logger.debug("page_loaded", page=page_number, chars=len(page_text))

    except PyPdfError as e:
        # pypdf's own error hierarchy: malformed/corrupt/encrypted PDF.
        # Log with traceback, then translate to our domain error so callers
        # catch IngestionError, not a pypdf-internal type.
        logger.exception("pdf_parse_failed", path=str(path))
        raise PdfLoadError(f"Failed to parse PDF: {path}") from e

    logger.info(
        "pdf_loaded",
        path=str(path),
        total_pages=len(reader.pages),
        pages_loaded=len(pages),
        skipped=skipped,
    )
    return pages