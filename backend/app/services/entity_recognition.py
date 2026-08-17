"""Medical entity recognition using SciSpaCy."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from app.config.settings import Settings
from app.services.document_parsing import clean_extracted_text
from app.services.section_segmentation import ClinicalSection, ClinicalSectionType

logger = logging.getLogger(__name__)


class EntityRecognitionError(RuntimeError):
    """Raised when medical entity recognition cannot be completed."""


class MedicalEntityType(StrEnum):
    """Medical entity categories required by the pipeline."""

    DISEASE = "disease"
    SYMPTOM = "symptom"
    DRUG = "drug"
    ANATOMY = "anatomy"
    PROCEDURE = "procedure"
    LAB_TEST = "laboratory_test"


@dataclass(frozen=True)
class MedicalEntity:
    """A detected medical entity with structured metadata."""

    text: str
    entity_type: MedicalEntityType
    section_type: ClinicalSectionType
    section_title: str
    start_char: int
    end_char: int
    confidence: float
    source_label: str | None = None


@dataclass(frozen=True)
class EntityRecognitionResult:
    """Structured medical entities detected across report sections."""

    entities: list[MedicalEntity]
    entity_count: int
    warnings: list[str] = field(default_factory=list)


class _NlpModel(Protocol):
    """Protocol for spaCy-compatible NLP models."""

    def __call__(self, text: str) -> Any:
        """Return a spaCy-like document for text."""


class MedicalEntityRecognitionService:
    """Extract and classify medical entities from segmented report sections."""

    _label_map: dict[str, MedicalEntityType] = {
        "DISEASE": MedicalEntityType.DISEASE,
        "DISEASES": MedicalEntityType.DISEASE,
        "DISORDER": MedicalEntityType.DISEASE,
        "PROBLEM": MedicalEntityType.DISEASE,
        "CHEMICAL": MedicalEntityType.DRUG,
        "CHEMICALS": MedicalEntityType.DRUG,
        "DRUG": MedicalEntityType.DRUG,
        "MEDICATION": MedicalEntityType.DRUG,
        "ANATOMY": MedicalEntityType.ANATOMY,
        "BODY_PART": MedicalEntityType.ANATOMY,
        "PROCEDURE": MedicalEntityType.PROCEDURE,
        "TEST": MedicalEntityType.LAB_TEST,
        "LAB_TEST": MedicalEntityType.LAB_TEST,
    }

    _category_keywords: dict[MedicalEntityType, tuple[str, ...]] = {
        MedicalEntityType.DISEASE: (
            "diabetes",
            "hypertension",
            "asthma",
            "pneumonia",
            "infection",
            "disease",
            "failure",
            "cancer",
            "anemia",
        ),
        MedicalEntityType.SYMPTOM: (
            "pain",
            "fever",
            "cough",
            "nausea",
            "vomiting",
            "fatigue",
            "dizziness",
            "shortness of breath",
            "headache",
            "swelling",
        ),
        MedicalEntityType.DRUG: (
            "lisinopril",
            "metformin",
            "atorvastatin",
            "aspirin",
            "insulin",
            "tablet",
            "capsule",
            "mg",
            "mcg",
            "ml",
        ),
        MedicalEntityType.ANATOMY: (
            "heart",
            "lung",
            "liver",
            "kidney",
            "brain",
            "chest",
            "abdomen",
            "artery",
            "vein",
        ),
        MedicalEntityType.PROCEDURE: (
            "surgery",
            "biopsy",
            "ct",
            "mri",
            "x-ray",
            "ultrasound",
            "endoscopy",
            "procedure",
        ),
        MedicalEntityType.LAB_TEST: (
            "glucose",
            "hemoglobin",
            "hba1c",
            "ldl",
            "hdl",
            "creatinine",
            "cbc",
            "wbc",
            "platelet",
            "cholesterol",
        ),
    }

    _section_defaults: dict[ClinicalSectionType, MedicalEntityType] = {
        ClinicalSectionType.DIAGNOSIS: MedicalEntityType.DISEASE,
        ClinicalSectionType.MEDICATIONS: MedicalEntityType.DRUG,
        ClinicalSectionType.LAB_RESULTS: MedicalEntityType.LAB_TEST,
        ClinicalSectionType.FINDINGS: MedicalEntityType.SYMPTOM,
        ClinicalSectionType.PROCEDURES: MedicalEntityType.PROCEDURE,
    }

    def __init__(self, settings: Settings, nlp_model: _NlpModel | None = None) -> None:
        """Initialize entity recognizer with an injectable SciSpaCy model."""
        self._settings = settings
        self._nlp_model = nlp_model

    def extract(self, sections: list[ClinicalSection]) -> EntityRecognitionResult:
        """Extract medical entities from clinical sections."""
        if not sections:
            raise EntityRecognitionError("At least one clinical section is required.")

        nlp_model = self._get_nlp_model()
        entities: list[MedicalEntity] = []
        seen: set[tuple[str, MedicalEntityType, ClinicalSectionType, int, int]] = set()

        for section in sections:
            content = clean_extracted_text(section.content)
            if not content:
                continue
            document = nlp_model(content)
            for span in getattr(document, "ents", []):
                entity_text = clean_extracted_text(getattr(span, "text", ""))
                if not self._is_valid_entity_text(entity_text):
                    continue

                source_label = getattr(span, "label_", None) or None
                entity_type = self._classify_entity(
                    text=entity_text,
                    source_label=source_label,
                    section_type=section.section_type,
                )
                if entity_type is None:
                    logger.debug("Skipping unclassified entity candidate: %s", entity_text)
                    continue

                start_char = int(getattr(span, "start_char", 0))
                end_char = int(getattr(span, "end_char", start_char + len(entity_text)))
                dedupe_key = (
                    entity_text.lower(),
                    entity_type,
                    section.section_type,
                    start_char,
                    end_char,
                )
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                entities.append(
                    MedicalEntity(
                        text=entity_text,
                        entity_type=entity_type,
                        section_type=section.section_type,
                        section_title=section.title,
                        start_char=start_char,
                        end_char=end_char,
                        confidence=self._confidence_for(source_label, section.section_type),
                        source_label=source_label,
                    )
                )

        logger.info("Extracted %d medical entity/entities", len(entities))
        return EntityRecognitionResult(entities=entities, entity_count=len(entities))

    def _get_nlp_model(self) -> _NlpModel:
        """Load and cache the configured SciSpaCy model."""
        if self._nlp_model is not None:
            return self._nlp_model

        try:
            import spacy
        except ImportError as exc:
            raise EntityRecognitionError(
                "spaCy/SciSpaCy is not installed. Install backend requirements before entity recognition."
            ) from exc

        try:
            self._nlp_model = spacy.load(self._settings.scispacy_model_name)
        except OSError as exc:
            raise EntityRecognitionError(
                f"SciSpaCy model '{self._settings.scispacy_model_name}' is not installed."
            ) from exc
        return self._nlp_model

    def _classify_entity(
        self,
        *,
        text: str,
        source_label: str | None,
        section_type: ClinicalSectionType,
    ) -> MedicalEntityType | None:
        """Assign a detected span to one of the required medical categories."""
        if source_label:
            mapped_type = self._label_map.get(source_label.upper())
            if mapped_type is not None:
                return mapped_type

        normalized_text = _normalize_entity_text(text)
        for entity_type, keywords in self._category_keywords.items():
            if any(keyword in normalized_text for keyword in keywords):
                return entity_type

        return self._section_defaults.get(section_type)

    def _confidence_for(
        self,
        source_label: str | None,
        section_type: ClinicalSectionType,
    ) -> float:
        """Estimate confidence from model label specificity and section context."""
        if source_label and source_label.upper() in self._label_map:
            return 0.9
        if section_type in self._section_defaults:
            return max(self._settings.entity_default_confidence, 0.8)
        return self._settings.entity_default_confidence

    @staticmethod
    def _is_valid_entity_text(text: str) -> bool:
        """Filter out non-informative span candidates."""
        if len(text) < 2:
            return False
        if re.fullmatch(r"[\W_]+", text):
            return False
        return True


def _normalize_entity_text(text: str) -> str:
    """Normalize entity text for deterministic category matching."""
    normalized = text.lower()
    normalized = re.sub(r"[^a-z0-9+\-/ ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()
