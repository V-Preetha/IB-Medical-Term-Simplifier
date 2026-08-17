"""ModernBERT difficult-term detection and contextual embeddings."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Protocol

from app.config.settings import Settings
from app.services.document_parsing import clean_extracted_text
from app.services.entity_recognition import MedicalEntity, MedicalEntityType

logger = logging.getLogger(__name__)


class ModernBERTProcessingError(RuntimeError):
    """Raised when ModernBERT processing cannot be completed."""


@dataclass(frozen=True)
class DifficultTerm:
    """A difficult medical term with ModernBERT context metadata."""

    term: str
    difficulty: float
    embedding: list[float]
    confidence: float
    entity_type: MedicalEntityType
    section_type: str
    context: str


@dataclass(frozen=True)
class ModernBERTResult:
    """ModernBERT output for difficult medical term processing."""

    terms: list[DifficultTerm]
    term_count: int
    model_name: str
    warnings: list[str] = field(default_factory=list)


class EmbeddingBackend(Protocol):
    """Protocol for contextual embedding backends."""

    model_name: str

    def embed(self, *, term: str, context: str) -> list[float]:
        """Return one contextual embedding vector for a term in context."""


class ModernBERTEmbeddingBackend:
    """HuggingFace ModernBERT backend for contextual term embeddings."""

    def __init__(self, settings: Settings) -> None:
        """Initialize a lazy-loading HuggingFace backend."""
        self.model_name = settings.modernbert_model_name
        self._max_length = settings.modernbert_max_length
        self._tokenizer = None
        self._model = None

    def embed(self, *, term: str, context: str) -> list[float]:
        """Create a normalized contextual embedding using ModernBERT."""
        tokenizer, model = self._load_model()
        try:
            import torch
        except ImportError as exc:
            raise ModernBERTProcessingError(
                "PyTorch is not installed. Install backend requirements before ModernBERT processing."
            ) from exc

        encoded = tokenizer(
            context,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
        )
        with torch.no_grad():
            output = model(**encoded)
        hidden_states = output.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        masked_hidden = hidden_states * attention_mask
        pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
        vector = pooled.squeeze(0).detach().cpu().tolist()
        return _normalize_vector([float(value) for value in vector])

    def _load_model(self) -> tuple[object, object]:
        """Load the configured ModernBERT tokenizer and model once."""
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ModernBERTProcessingError(
                "Transformers is not installed. Install backend requirements before ModernBERT processing."
            ) from exc

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
        except Exception as exc:
            logger.exception("ModernBERT model loading failed")
            raise ModernBERTProcessingError(
                f"ModernBERT model '{self.model_name}' could not be loaded."
            ) from exc
        return self._tokenizer, self._model


class ModernBERTProcessingService:
    """Detect difficult medical terms and embed them with ModernBERT."""

    _type_complexity: dict[MedicalEntityType, float] = {
        MedicalEntityType.DISEASE: 0.78,
        MedicalEntityType.SYMPTOM: 0.52,
        MedicalEntityType.DRUG: 0.72,
        MedicalEntityType.ANATOMY: 0.58,
        MedicalEntityType.PROCEDURE: 0.74,
        MedicalEntityType.LAB_TEST: 0.80,
    }

    _plain_language_terms = {
        "pain",
        "fever",
        "cough",
        "heart",
        "lung",
        "chest",
        "blood",
    }

    def __init__(
        self,
        settings: Settings,
        embedding_backend: EmbeddingBackend | None = None,
    ) -> None:
        """Initialize ModernBERT processing with injectable embeddings."""
        self._settings = settings
        self._embedding_backend = embedding_backend or ModernBERTEmbeddingBackend(settings)

    def process(self, entities: list[MedicalEntity]) -> ModernBERTResult:
        """Identify difficult terms and return contextual embeddings."""
        if not entities:
            raise ModernBERTProcessingError("At least one medical entity is required.")

        difficult_terms: list[DifficultTerm] = []
        seen: set[tuple[str, str, str]] = set()
        for entity in entities:
            cleaned_term = clean_extracted_text(entity.text)
            context = clean_extracted_text(entity.section_title + ": " + entity.text)
            if not cleaned_term:
                continue

            difficulty = self._score_difficulty(cleaned_term, entity.entity_type)
            if difficulty < self._settings.difficult_term_threshold:
                logger.debug("Skipping easy term candidate: %s", cleaned_term)
                continue

            dedupe_key = (
                cleaned_term.lower(),
                entity.entity_type.value,
                entity.section_type.value,
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            embedding = self._embedding_backend.embed(term=cleaned_term, context=context)
            difficult_terms.append(
                DifficultTerm(
                    term=cleaned_term,
                    difficulty=difficulty,
                    embedding=embedding,
                    confidence=self._score_confidence(entity.confidence, embedding),
                    entity_type=entity.entity_type,
                    section_type=entity.section_type.value,
                    context=context,
                )
            )

        logger.info("ModernBERT identified %d difficult term(s)", len(difficult_terms))
        return ModernBERTResult(
            terms=difficult_terms,
            term_count=len(difficult_terms),
            model_name=self._embedding_backend.model_name,
        )

    def _score_difficulty(self, term: str, entity_type: MedicalEntityType) -> float:
        """Estimate term difficulty from medical category and lexical complexity."""
        normalized = term.lower()
        if normalized in self._plain_language_terms:
            return 0.25

        lexical_score = 0.0
        lexical_score += min(len(term) / 60, 0.20)
        lexical_score += 0.08 if _count_syllable_groups(term) >= 4 else 0.0
        lexical_score += 0.10 if re.search(r"\d|/|-", term) else 0.0
        lexical_score += 0.10 if len(term.split()) > 1 else 0.0
        category_score = self._type_complexity[entity_type]
        return round(min((category_score * 0.7) + lexical_score, 1.0), 4)

    @staticmethod
    def _score_confidence(entity_confidence: float, embedding: list[float]) -> float:
        """Combine upstream entity confidence with embedding validity."""
        embedding_signal = 1.0 if embedding and any(value != 0.0 for value in embedding) else 0.2
        return round(min((entity_confidence * 0.8) + (embedding_signal * 0.2), 1.0), 4)


def _count_syllable_groups(term: str) -> int:
    """Approximate syllable-group count for lexical difficulty scoring."""
    groups = re.findall(r"[aeiouy]+", term.lower())
    return len(groups)


def _normalize_vector(vector: list[float]) -> list[float]:
    """L2-normalize an embedding vector for stable downstream fusion."""
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [round(value / magnitude, 8) for value in vector]
