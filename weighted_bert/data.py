from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass(repr=False)
class InputExample:
    document: list[str]
    sentence_embeddings: np.ndarray | list[list[float]]

    def __post_init__(self):
        if isinstance(self.sentence_embeddings, list):
            self.sentence_embeddings = np.array(self.sentence_embeddings)

    def __str__(self) -> str:
        return f"<Input Example> document with {len(self.document)} sentences, sentence embeddings of shape {self.sentence_embeddings.shape}"

    def __repr__(self) -> str:
        return f"<Input Example> document with {len(self.document)} sentences, sentence embeddings of shape {self.sentence_embeddings.shape} shape"
