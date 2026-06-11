from pydantic import BaseModel

from search_lab.search.modes import SearchMode


class SearchResult(BaseModel):
    id: str
    text: str
    score: float  # normalized: higher = better, regardless of backend
    rank: int  # 1-based position in this result list
    page: int
    char_start: int
    strategy: str
    mode: SearchMode  # which search produced it
