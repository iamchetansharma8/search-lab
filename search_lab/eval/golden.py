"""Golden test set: schema + loader.

The golden set is the ground truth for the eval — 20 questions, each labeled
with the source-document span(s) that answer it. Labels are stored as
(page, char_start, char_end) so they are CHUNK-SET-AGNOSTIC: the same golden
set scores fixed, recursive, or any future chunker, because relevance is
decided by char-span overlap (see metrics.is_relevant), not by chunk id.

A query may have >1 relevant span; a result is relevant if it covers any one
of them past the coverage threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class RelevantSpan(BaseModel):
    """One answer location in the source document, per-page coordinates."""

    page: int = Field(..., description="1-indexed source page number")
    char_start: int = Field(
        ..., ge=0, description="Start offset within the page's text"
    )
    char_end: int = Field(
        ..., gt=0, description="End offset (exclusive) within the page's text"
    )

    @model_validator(mode="after")
    def _check_order(self) -> "RelevantSpan":
        if self.char_end <= self.char_start:
            raise ValueError(
                f"char_end ({self.char_end}) must be > char_start ({self.char_start})"
            )
        return self


class GoldenQuery(BaseModel):
    """A test question and its labeled relevant span(s)."""

    qid: str = Field(..., description="Stable id, e.g. 'q01'")
    query: str = Field(..., description="The question text")
    relevant: list[RelevantSpan] = Field(
        ..., min_length=1, description="≥1 answer location(s)"
    )
    note: str | None = Field(None, description="Optional: why this is the answer")


def load_golden_set(path: str | Path) -> list[GoldenQuery]:
    """Load + validate the golden set JSON. Raises on malformed data."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("golden set JSON must be a top-level list of queries")

    queries = [GoldenQuery.model_validate(item) for item in raw]

    # Guard against duplicate qids — a copy-paste slip would silently double-count.
    seen: set[str] = set()
    for q in queries:
        if q.qid in seen:
            raise ValueError(f"duplicate qid in golden set: {q.qid}")
        seen.add(q.qid)

    return queries
