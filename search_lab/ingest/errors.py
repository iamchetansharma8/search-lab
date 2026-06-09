"""Ingestion-specific exceptions, so callers can distinguish ingestion
failures from arbitrary runtime errors and handle them deliberately."""


class IngestionError(Exception):
    """Base class for all ingestion failures."""


class PdfLoadError(IngestionError):
    """Raised when a PDF cannot be opened or parsed."""

class ChunkingError(Exception):
    """Base class for all chunking failures."""

class InvalidParamError(ChunkingError):
    """Raised when chunking fails due to zero or negative value of overlap or chunk_size."""