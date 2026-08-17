import logging
from typing import Optional

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


LOGGER = logging.getLogger(__name__)


class SemanticScorer:
    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.backend = "sentence-transformers"
        self.model = None
        self.tokenizer = None
        self._load()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as st_exc:
            LOGGER.warning(
                "SentenceTransformer load failed for %s; falling back to transformers mean pooling: %s",
                self.model_name,
                st_exc,
            )
            from transformers import AutoModel, AutoTokenizer

            self.backend = "transformers"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
            self.model.eval()

    def encode(self, text: str) -> np.ndarray:
        if self.backend == "sentence-transformers":
            return np.asarray(self.model.encode([text], normalize_embeddings=True)[0])

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=min(getattr(self.tokenizer, "model_max_length", 8192), 8192),
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            output = self.model(**encoded)
        token_embeddings = output.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * attention_mask, dim=1) / torch.clamp(
            attention_mask.sum(dim=1), min=1e-9
        )
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy()[0]

    def similarity(self, original: str, simplified: str) -> float:
        try:
            vectors = np.vstack([self.encode(original), self.encode(simplified)])
            return float(cosine_similarity(vectors[:1], vectors[1:2])[0][0])
        except Exception as exc:
            LOGGER.exception("Semantic similarity failed: %s", exc)
            return 0.0
