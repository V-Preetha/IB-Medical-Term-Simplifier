"""Fusion layer for ModernBERT and BioClinicalBERT/OpenMed outputs.

Fusion algorithm:
1. Normalize each term to a stable key using lowercase text, entity type, and
   section type.
2. Match ModernBERT difficult-term outputs to clinical semantic interpretations
   first by full key, then by term-only fallback.
3. Preserve both model contributions: ModernBERT supplies difficulty,
   contextual embeddings, and confidence; BioClinicalBERT/OpenMed supplies
   meaning, clinical context, ambiguity resolution, semantic embeddings, and
   semantic confidence.
4. Compute fused confidence as a weighted score:
   45% ModernBERT confidence, 45% semantic confidence, 10% match quality.
5. Emit warnings for unmatched model outputs instead of dropping information.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.services.clinical_context import SemanticInterpretation
from app.services.document_parsing import clean_extracted_text
from app.services.modernbert import DifficultTerm

logger = logging.getLogger(__name__)


class FusionError(RuntimeError):
    """Raised when model outputs cannot be fused."""


@dataclass(frozen=True)
class FusedMedicalTerm:
    """Unified structured representation for one medical term."""

    term: str
    difficulty: float
    meaning: str
    context: str
    confidence: float
    entity_type: str
    section_type: str
    modernbert_confidence: float
    semantic_confidence: float
    ambiguity_resolution: str
    modernbert_embedding: list[float]
    semantic_embedding: list[float]
    matched_concept: str | None = None


@dataclass(frozen=True)
class FusionResult:
    """Fused structured medical representation."""

    terms: list[FusedMedicalTerm]
    term_count: int
    warnings: list[str] = field(default_factory=list)
    algorithm_version: str = "weighted-key-match-v1"


class FusionService:
    """Fuse ModernBERT and BioClinicalBERT/OpenMed outputs."""

    def fuse(
        self,
        *,
        difficult_terms: list[DifficultTerm],
        semantic_interpretations: list[SemanticInterpretation],
    ) -> FusionResult:
        """Merge difficult-term and semantic-understanding model outputs."""
        if not difficult_terms:
            raise FusionError("At least one ModernBERT difficult term is required.")
        if not semantic_interpretations:
            raise FusionError(
                "At least one BioClinicalBERT/OpenMed semantic interpretation is required."
            )

        semantic_by_key = {
            _fusion_key(
                term=interpretation.term,
                entity_type=interpretation.entity_type.value,
                section_type=interpretation.section_type,
            ): interpretation
            for interpretation in semantic_interpretations
        }
        semantic_by_term = {
            _normalize_term(interpretation.term): interpretation
            for interpretation in semantic_interpretations
        }

        fused_terms: list[FusedMedicalTerm] = []
        matched_semantic_keys: set[str] = set()
        warnings: list[str] = []

        for difficult_term in difficult_terms:
            full_key = _fusion_key(
                term=difficult_term.term,
                entity_type=difficult_term.entity_type.value,
                section_type=difficult_term.section_type,
            )
            semantic = semantic_by_key.get(full_key)
            match_quality = 1.0
            if semantic is None:
                semantic = semantic_by_term.get(_normalize_term(difficult_term.term))
                match_quality = 0.75

            if semantic is None:
                warnings.append(
                    f"No semantic interpretation found for '{difficult_term.term}'."
                )
                continue

            matched_semantic_keys.add(
                _fusion_key(
                    term=semantic.term,
                    entity_type=semantic.entity_type.value,
                    section_type=semantic.section_type,
                )
            )
            fused_terms.append(
                FusedMedicalTerm(
                    term=difficult_term.term,
                    difficulty=difficult_term.difficulty,
                    meaning=semantic.meaning,
                    context=semantic.context,
                    confidence=_fused_confidence(
                        modernbert_confidence=difficult_term.confidence,
                        semantic_confidence=semantic.confidence,
                        match_quality=match_quality,
                    ),
                    entity_type=difficult_term.entity_type.value,
                    section_type=difficult_term.section_type,
                    modernbert_confidence=difficult_term.confidence,
                    semantic_confidence=semantic.confidence,
                    ambiguity_resolution=semantic.ambiguity_resolution,
                    modernbert_embedding=difficult_term.embedding,
                    semantic_embedding=semantic.semantic_embedding,
                    matched_concept=semantic.matched_concept,
                )
            )

        for semantic in semantic_interpretations:
            semantic_key = _fusion_key(
                term=semantic.term,
                entity_type=semantic.entity_type.value,
                section_type=semantic.section_type,
            )
            if semantic_key not in matched_semantic_keys:
                warnings.append(
                    f"Semantic interpretation for '{semantic.term}' had no ModernBERT difficult-term match."
                )

        if not fused_terms:
            raise FusionError("No overlapping terms could be fused.")

        logger.info("Fused %d medical term(s)", len(fused_terms))
        return FusionResult(
            terms=fused_terms,
            term_count=len(fused_terms),
            warnings=warnings,
        )


def _fused_confidence(
    *,
    modernbert_confidence: float,
    semantic_confidence: float,
    match_quality: float,
) -> float:
    """Compute weighted fused confidence while preserving source confidences."""
    return round(
        min(
            (modernbert_confidence * 0.45)
            + (semantic_confidence * 0.45)
            + (match_quality * 0.10),
            1.0,
        ),
        4,
    )


def _fusion_key(*, term: str, entity_type: str, section_type: str) -> str:
    """Build the strict fusion key."""
    return "|".join([_normalize_term(term), entity_type, section_type])


def _normalize_term(term: str) -> str:
    """Normalize term text for matching."""
    return clean_extracted_text(term).lower()
