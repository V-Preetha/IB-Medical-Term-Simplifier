"""BioClinicalBERT/OpenMed semantic understanding and ambiguity resolution."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Protocol

from app.config.settings import Settings
from app.services.document_parsing import clean_extracted_text
from app.services.entity_recognition import MedicalEntity, MedicalEntityType

logger = logging.getLogger(__name__)


class ClinicalContextError(RuntimeError):
    """Raised when clinical context processing cannot be completed."""


@dataclass(frozen=True)
class ClinicalConcept:
    """A candidate clinical concept used for semantic retrieval."""

    canonical_term: str
    entity_type: MedicalEntityType
    meaning: str
    clinical_context: str
    ambiguity_resolution: str


@dataclass(frozen=True)
class SemanticInterpretation:
    """Semantic interpretation of one medical term in clinical context."""

    term: str
    meaning: str
    context: str
    ambiguity_resolution: str
    confidence: float
    entity_type: MedicalEntityType
    section_type: str
    semantic_embedding: list[float]
    matched_concept: str | None = None


@dataclass(frozen=True)
class ClinicalContextResult:
    """BioClinicalBERT/OpenMed semantic interpretation result."""

    interpretations: list[SemanticInterpretation]
    interpretation_count: int
    model_name: str
    warnings: list[str] = field(default_factory=list)


class SemanticEmbeddingBackend(Protocol):
    """Protocol for clinical semantic embedding backends."""

    model_name: str

    def embed(self, text: str) -> list[float]:
        """Return one contextual semantic embedding."""


class BioClinicalBERTEmbeddingBackend:
    """HuggingFace BioClinicalBERT/OpenMed-compatible embedding backend."""

    def __init__(self, settings: Settings) -> None:
        """Initialize a lazy-loading clinical transformer backend."""
        self.model_name = settings.clinical_context_model_name
        self._max_length = settings.clinical_context_max_length
        self._tokenizer = None
        self._model = None

    def embed(self, text: str) -> list[float]:
        """Create a normalized clinical semantic embedding."""
        tokenizer, model = self._load_model()
        try:
            import torch
        except ImportError as exc:
            raise ClinicalContextError(
                "PyTorch is not installed. Install backend requirements before clinical context processing."
            ) from exc

        encoded = tokenizer(
            text,
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
        """Load the configured clinical tokenizer and model once."""
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ClinicalContextError(
                "Transformers is not installed. Install backend requirements before clinical context processing."
            ) from exc

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
        except Exception as exc:
            logger.exception("Clinical context model loading failed")
            raise ClinicalContextError(
                f"Clinical context model '{self.model_name}' could not be loaded."
            ) from exc
        return self._tokenizer, self._model


class ClinicalContextService:
    """Resolve semantic meaning and ambiguity for medical entities."""

    _concepts: tuple[ClinicalConcept, ...] = (
        ClinicalConcept(
            canonical_term="hypertension",
            entity_type=MedicalEntityType.DISEASE,
            meaning="High blood pressure.",
            clinical_context="A chronic condition where blood pressure remains higher than normal.",
            ambiguity_resolution=(
                "Resolved as the chronic disease because it appears as a diagnosis or disease entity."
            ),
        ),
        ClinicalConcept(
            canonical_term="diabetes",
            entity_type=MedicalEntityType.DISEASE,
            meaning="A condition where the body has trouble controlling blood sugar.",
            clinical_context="Usually a long-term metabolic condition monitored with glucose and HbA1c.",
            ambiguity_resolution="Resolved as a metabolic disease based on disease context.",
        ),
        ClinicalConcept(
            canonical_term="chest pain",
            entity_type=MedicalEntityType.SYMPTOM,
            meaning="Pain or discomfort felt in the chest area.",
            clinical_context="A symptom that may need urgent evaluation depending on severity and associated signs.",
            ambiguity_resolution="Resolved as a symptom because the entity appears in findings or symptoms context.",
        ),
        ClinicalConcept(
            canonical_term="lisinopril",
            entity_type=MedicalEntityType.DRUG,
            meaning="A medication commonly used to lower blood pressure.",
            clinical_context="An ACE inhibitor; medication names and doses should be preserved exactly.",
            ambiguity_resolution="Resolved as a medication because it appears as a drug entity.",
        ),
        ClinicalConcept(
            canonical_term="ldl",
            entity_type=MedicalEntityType.LAB_TEST,
            meaning="Low-density lipoprotein cholesterol, often called bad cholesterol.",
            clinical_context="A lab value used to assess cardiovascular risk.",
            ambiguity_resolution="Resolved as a laboratory test because it appears in lab results context.",
        ),
        ClinicalConcept(
            canonical_term="creatinine",
            entity_type=MedicalEntityType.LAB_TEST,
            meaning="A blood test marker used to assess kidney function.",
            clinical_context="Higher values can suggest reduced kidney filtering, depending on context.",
            ambiguity_resolution="Resolved as a laboratory test based on lab context.",
        ),
        ClinicalConcept(
            canonical_term="ct",
            entity_type=MedicalEntityType.PROCEDURE,
            meaning="Computed tomography, an imaging test that uses X-rays to create detailed pictures.",
            clinical_context="A diagnostic imaging procedure.",
            ambiguity_resolution="Resolved as a procedure when used in imaging or procedure context.",
        ),
    )

    def __init__(
        self,
        settings: Settings,
        embedding_backend: SemanticEmbeddingBackend | None = None,
    ) -> None:
        """Initialize semantic understanding with injectable embeddings."""
        self._settings = settings
        self._embedding_backend = embedding_backend or BioClinicalBERTEmbeddingBackend(settings)

    def interpret(self, entities: list[MedicalEntity]) -> ClinicalContextResult:
        """Produce semantic meaning, clinical context, and ambiguity resolution."""
        if not entities:
            raise ClinicalContextError("At least one medical entity is required.")

        interpretations: list[SemanticInterpretation] = []
        warnings: list[str] = []
        for entity in entities:
            term = clean_extracted_text(entity.text)
            if not term:
                continue
            query_text = self._build_query_text(entity)
            query_embedding = self._embedding_backend.embed(query_text)
            match, similarity = self._match_concept(entity, query_embedding)

            if match is None:
                warnings.append(f"No glossary concept met semantic threshold for '{term}'.")
                meaning = self._fallback_meaning(entity)
                context = self._fallback_context(entity)
                ambiguity_resolution = (
                    "No close concept match was found; interpretation is based on entity type and section context."
                )
                confidence = round(min(entity.confidence * 0.65, 0.65), 4)
                matched_concept = None
            else:
                meaning = match.meaning
                context = match.clinical_context
                ambiguity_resolution = match.ambiguity_resolution
                confidence = round(min((entity.confidence * 0.55) + ((similarity + 1) / 2 * 0.45), 1.0), 4)
                matched_concept = match.canonical_term

            interpretations.append(
                SemanticInterpretation(
                    term=term,
                    meaning=meaning,
                    context=context,
                    ambiguity_resolution=ambiguity_resolution,
                    confidence=confidence,
                    entity_type=entity.entity_type,
                    section_type=entity.section_type.value,
                    semantic_embedding=query_embedding,
                    matched_concept=matched_concept,
                )
            )

        logger.info("Clinical context produced %d interpretation(s)", len(interpretations))
        return ClinicalContextResult(
            interpretations=interpretations,
            interpretation_count=len(interpretations),
            model_name=self._embedding_backend.model_name,
            warnings=warnings,
        )

    def _match_concept(
        self,
        entity: MedicalEntity,
        query_embedding: list[float],
    ) -> tuple[ClinicalConcept | None, float]:
        """Select the best clinical concept by entity type and embedding similarity."""
        candidates = [
            concept
            for concept in self._concepts
            if concept.entity_type is entity.entity_type
        ]
        best_concept: ClinicalConcept | None = None
        best_similarity = -1.0
        for concept in candidates:
            concept_embedding = self._embedding_backend.embed(
                self._concept_to_text(concept)
            )
            similarity = _cosine_similarity(query_embedding, concept_embedding)
            lexical_bonus = 0.15 if concept.canonical_term == entity.text.lower() else 0.0
            similarity = min(similarity + lexical_bonus, 1.0)
            if similarity > best_similarity:
                best_concept = concept
                best_similarity = similarity

        if best_concept is None or best_similarity < self._settings.semantic_match_threshold:
            return None, best_similarity
        return best_concept, best_similarity

    @staticmethod
    def _build_query_text(entity: MedicalEntity) -> str:
        """Build contextual text for clinical semantic encoding."""
        return clean_extracted_text(
            " | ".join(
                [
                    f"term: {entity.text}",
                    f"type: {entity.entity_type.value}",
                    f"section: {entity.section_title}",
                ]
            )
        )

    @staticmethod
    def _concept_to_text(concept: ClinicalConcept) -> str:
        """Build retrieval text for a clinical concept candidate."""
        return clean_extracted_text(
            " | ".join(
                [
                    f"term: {concept.canonical_term}",
                    f"type: {concept.entity_type.value}",
                    f"meaning: {concept.meaning}",
                    f"context: {concept.clinical_context}",
                ]
            )
        )

    @staticmethod
    def _fallback_meaning(entity: MedicalEntity) -> str:
        """Provide a conservative type-level meaning when no concept matches."""
        meanings = {
            MedicalEntityType.DISEASE: "A medical condition or diagnosis.",
            MedicalEntityType.SYMPTOM: "A health problem or experience reported by the patient.",
            MedicalEntityType.DRUG: "A medication or drug name.",
            MedicalEntityType.ANATOMY: "A body part or anatomical location.",
            MedicalEntityType.PROCEDURE: "A medical test, treatment, or procedure.",
            MedicalEntityType.LAB_TEST: "A laboratory test or measured result.",
        }
        return meanings[entity.entity_type]

    @staticmethod
    def _fallback_context(entity: MedicalEntity) -> str:
        """Provide section-aware context when no concept matches."""
        return (
            f"Appears in the {entity.section_title} section as a "
            f"{entity.entity_type.value.replace('_', ' ')}."
        )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity for two vectors."""
    if not left or not right or len(left) != len(right):
        return -1.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return -1.0
    return numerator / (left_norm * right_norm)


def _normalize_vector(vector: list[float]) -> list[float]:
    """L2-normalize a clinical semantic embedding vector."""
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [round(value / magnitude, 8) for value in vector]
