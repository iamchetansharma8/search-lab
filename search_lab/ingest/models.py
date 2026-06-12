"""Core data types for ingestion. The Chunk is the unit that flows
through the entire pipeline: ingest -> embed -> search -> rerank -> eval."""


from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A single retrievable piece of the source document.

    Every chunk carries provenance (page) so retrieval results can be
    traced back to the source — needed for eval labeling and 'verifiable source span' rule.
    """
    id: str = Field(..., description="Stable unique id, e.g. 'fixed-0007'")
    text: str = Field(..., description="The chunk's text content")
    page: int = Field(..., description="1-indexed source page number")
    strategy: str = Field(..., description="Chunker that produced it, e.g. 'fixed' / 'recursive'")
    
    """char_start tells where in the page the given chunk began. Finer-grained provenance than page alone"""
    char_start: int = Field(..., description="Start offset within the page's text")
    char_end: int = Field(
        ..., description="End offset (exclusive) within the page's text"
    )
    def __len__(self) -> int:
        """Length in characters — convenience for inspection/logging."""
        return len(self.text)