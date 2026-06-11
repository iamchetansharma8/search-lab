from dataclasses import dataclass, field

from sentence_transformers import SentenceTransformer


@dataclass
class EmbedModelSpec:
    """Wraps a SentenceTransformer model with token-budget helpers for chunking."""

    hf_name: str
    model: SentenceTransformer = field(init=False)
    max_tokens: int = field(init=False)
    # Reserves 2 slots for the [CLS] and [SEP] special tokens the model prepends/appends.
    content_budget: int = field(init=False)

    def __post_init__(self):
        self.model = SentenceTransformer(self.hf_name)
        self.max_tokens = self.model.max_seq_length
        self.content_budget = self.max_tokens - 2

    def length_function(self, text: str) -> int:
        """Returns token count without special tokens, matching how chunkers measure text."""
        return len(self.model.tokenizer.encode(text, add_special_tokens=False))
