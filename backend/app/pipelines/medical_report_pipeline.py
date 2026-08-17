"""End-to-end medical report simplification pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.config.settings import Settings
from app.evaluation.report_evaluation import EvaluationResult, EvaluationService
from app.fusion.medical_fusion import FusedMedicalTerm, FusionResult, FusionService
from app.services.clinical_context import ClinicalContextResult, ClinicalContextService
from app.services.document_parsing import DocumentParsingService, ParsedDocument
from app.services.entity_recognition import EntityRecognitionResult, MedicalEntityRecognitionService
from app.services.granite_guardian import (
    GraniteGuardianValidationService,
    ValidationAction,
    ValidationResult,
)
from app.services.modernbert import ModernBERTProcessingService, ModernBERTResult
from app.services.qwen_simplification import QwenSimplificationService, SimplificationResult
from app.services.section_segmentation import SegmentedReport, SectionSegmentationService

logger = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """Raised when the final medical report pipeline cannot complete."""


@dataclass(frozen=True)
class HighlightedDifficultTerm:
    """Highlighted difficult term returned by the final API."""

    term: str
    difficulty: float
    confidence: float
    entity_type: str
    section_type: str
    meaning: str


@dataclass(frozen=True)
class PipelineResult:
    """Final end-to-end pipeline result."""

    simplified_report: str
    highlighted_difficult_terms: list[HighlightedDifficultTerm]
    explanations: list[dict[str, float | str]]
    confidence: float
    evaluation: EvaluationResult
    validation: ValidationResult
    fused_terms: list[FusedMedicalTerm]
    warnings: list[str] = field(default_factory=list)


class MedicalReportPipeline:
    """Run the complete medical report simplification architecture."""

    def __init__(
        self,
        *,
        settings: Settings,
        document_parser: DocumentParsingService,
        section_segmenter: SectionSegmentationService,
        entity_recognizer: MedicalEntityRecognitionService,
        modernbert_processor: ModernBERTProcessingService,
        clinical_context_service: ClinicalContextService,
        fusion_service: FusionService,
        qwen_service: QwenSimplificationService,
        validator: GraniteGuardianValidationService,
        evaluator: EvaluationService,
    ) -> None:
        """Initialize the full pipeline with explicit stage dependencies."""
        self._settings = settings
        self._document_parser = document_parser
        self._section_segmenter = section_segmenter
        self._entity_recognizer = entity_recognizer
        self._modernbert_processor = modernbert_processor
        self._clinical_context_service = clinical_context_service
        self._fusion_service = fusion_service
        self._qwen_service = qwen_service
        self._validator = validator
        self._evaluator = evaluator

    def simplify_text(self, text: str) -> PipelineResult:
        """Run the full pipeline for directly submitted text."""
        try:
            parsed_document = self._document_parser.parse_text(text)
            return self._run(parsed_document)
        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError(str(exc)) from exc

    def simplify_file(
        self,
        *,
        filename: str,
        content: bytes,
        content_type: str | None,
    ) -> PipelineResult:
        """Run the full pipeline for an uploaded file."""
        try:
            parsed_document = self._document_parser.parse_file(
                filename=filename,
                content=content,
                content_type=content_type,
            )
            return self._run(parsed_document)
        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError(str(exc)) from exc

    def _run(self, parsed_document: ParsedDocument) -> PipelineResult:
        """Run every stage in the required architecture order."""
        segmented_report = self._section_segmenter.segment(parsed_document.text)
        entity_result = self._entity_recognizer.extract(segmented_report.sections)
        modernbert_result = self._modernbert_processor.process(entity_result.entities)
        clinical_context_result = self._clinical_context_service.interpret(
            entity_result.entities
        )
        fusion_result = self._fusion_service.fuse(
            difficult_terms=modernbert_result.terms,
            semantic_interpretations=clinical_context_result.interpretations,
        )
        simplification_result, validation_result = self._generate_validated_report(
            fused_terms=fusion_result.terms
        )
        evaluation_result = self._evaluator.evaluate(
            reference_text=parsed_document.text,
            simplified_report=simplification_result.simplified_report,
            fused_terms=fusion_result.terms,
        )
        warnings = _collect_warnings(
            parsed_document=parsed_document,
            segmented_report=segmented_report,
            entity_result=entity_result,
            modernbert_result=modernbert_result,
            clinical_context_result=clinical_context_result,
            fusion_result=fusion_result,
            simplification_result=simplification_result,
            validation_result=validation_result,
            evaluation_result=evaluation_result,
        )
        logger.info("Completed final medical report simplification pipeline")
        return PipelineResult(
            simplified_report=simplification_result.simplified_report,
            highlighted_difficult_terms=[
                HighlightedDifficultTerm(
                    term=term.term,
                    difficulty=term.difficulty,
                    confidence=term.confidence,
                    entity_type=term.entity_type,
                    section_type=term.section_type,
                    meaning=term.meaning,
                )
                for term in fusion_result.terms
            ],
            explanations=[
                {
                    "term": explanation.term,
                    "explanation": explanation.explanation,
                    "difficulty": explanation.difficulty,
                    "confidence": explanation.confidence,
                }
                for explanation in simplification_result.term_explanations
            ],
            confidence=_aggregate_confidence(fusion_result.terms, validation_result),
            evaluation=evaluation_result,
            validation=validation_result,
            fused_terms=fusion_result.terms,
            warnings=warnings,
        )

    def _generate_validated_report(
        self,
        *,
        fused_terms: list[FusedMedicalTerm],
    ) -> tuple[SimplificationResult, ValidationResult]:
        """Generate and validate, regenerating once when validation requests it."""
        attempts = self._settings.max_simplification_regeneration_attempts + 1
        last_simplification: SimplificationResult | None = None
        last_validation: ValidationResult | None = None

        for attempt in range(attempts):
            last_simplification = self._qwen_service.simplify(fused_terms)
            last_validation = self._validator.validate(
                fused_terms=fused_terms,
                simplified_report=last_simplification.simplified_report,
            )
            if last_validation.action is ValidationAction.APPROVE:
                return last_simplification, last_validation
            if last_validation.action is ValidationAction.REJECT:
                raise PipelineError(
                    "Generated simplification was rejected by validation."
                )
            logger.info("Validation requested regeneration on attempt %d", attempt + 1)

        raise PipelineError(
            "Generated simplification did not pass validation after regeneration."
        )


def _aggregate_confidence(
    fused_terms: list[FusedMedicalTerm],
    validation_result: ValidationResult,
) -> float:
    """Aggregate fused term confidence and validation status."""
    if not fused_terms:
        return 0.0
    fused_confidence = sum(term.confidence for term in fused_terms) / len(fused_terms)
    validation_factor = 1.0 if validation_result.validation_passed else 0.5
    return round(fused_confidence * validation_factor, 4)


def _collect_warnings(
    *,
    parsed_document: ParsedDocument,
    segmented_report: SegmentedReport,
    entity_result: EntityRecognitionResult,
    modernbert_result: ModernBERTResult,
    clinical_context_result: ClinicalContextResult,
    fusion_result: FusionResult,
    simplification_result: SimplificationResult,
    validation_result: ValidationResult,
    evaluation_result: EvaluationResult,
) -> list[str]:
    """Collect warnings emitted by each pipeline stage."""
    warnings: list[str] = []
    warnings.extend(parsed_document.warnings)
    warnings.extend(segmented_report.warnings)
    warnings.extend(entity_result.warnings)
    warnings.extend(modernbert_result.warnings)
    warnings.extend(clinical_context_result.warnings)
    warnings.extend(fusion_result.warnings)
    warnings.extend(simplification_result.warnings)
    warnings.extend(validation_result.warnings)
    warnings.extend(evaluation_result.warnings)
    return warnings
