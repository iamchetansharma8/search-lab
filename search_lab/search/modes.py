# search/modes.py  (or wherever the enum lives)
from enum import Enum


class SearchMode(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"