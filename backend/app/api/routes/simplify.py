"""Report parsing and simplification API routes.

Stage 11 exposes both individual stage endpoints and the final end-to-end API.
"""

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.config.settings import Settings, get_settings
from app.schemas.reports import (
    ClinicalSectionResponse,
    ClinicalContextRequest,
    ClinicalContextResponse,
    EntityRecognitionRequest,
    EntityRecognitionResponse,
    BERTScoreResponse,
    DifficultTermResponse,
    EvaluationRequest,
    EvaluationResponse,
    FinalSimplificationResponse,
    FusedMedicalTermResponse,
    FusionRequest,
    FusionResponse,
    HighlightedDifficultTermResponse,
    MedicalEntityResponse,
    ModernBERTRequest,
    ModernBERTResponse,
    ParsedDocumentResponse,
    SectionSegmentationRequest,
    SectionSegmentationResponse,
    SemanticInterpretationResponse,
    SimplificationFromFusionRequest,
    SimplificationRequest,
    SimplificationResponse,
    TermExplanationResponse,
    ReadabilityResponse,
    MedicalConsistencyResponse,
    GuardianAssessmentResponse,
    ValidationCheckResponse,
    ValidationRequest,
    ValidationResponse,
)
from app.evaluation.report_evaluation import EvaluationError, EvaluationService
from app.fusion.medical_fusion import FusionError, FusionService
from app.fusion.medical_fusion import FusedMedicalTerm
from app.services.clinical_context import ClinicalContextError, ClinicalContextService
from app.services.document_parsing import DocumentParsingError, DocumentParsingService
from app.services.entity_recognition import (
    EntityRecognitionError,
    MedicalEntity,
    MedicalEntityRecognitionService,
    MedicalEntityType,
)
from app.services.modernbert import ModernBERTProcessingError, ModernBERTProcessingService
from app.services.modernbert import DifficultTerm
from app.services.qwen_simplification import (
    QwenSimplificationService,
    SimplificationError,
)
from app.services.granite_guardian import (
    GraniteGuardianValidationService,
    ValidationError,
)
from app.services.clinical_context import SemanticInterpretation
from app.services.section_segmentation import (
    ClinicalSection,
    ClinicalSectionType,
    SectionSegmentationError,
    SectionSegmentationService,
)
from app.pipelines.medical_report_pipeline import MedicalReportPipeline, PipelineError

logger = logging.getLogger(__name__)

router = APIRouter()


def get_document_parser(
    settings: Settings = Depends(get_settings),
) -> DocumentParsingService:
    """Provide a document parser instance for request-scoped dependency injection."""
    return DocumentParsingService(settings)


def get_section_segmenter() -> SectionSegmentationService:
    """Provide a clinical section segmenter for dependency injection."""
    return SectionSegmentationService()


def get_entity_recognizer(
    settings: Settings = Depends(get_settings),
) -> MedicalEntityRecognitionService:
    """Provide a SciSpaCy entity recognizer for dependency injection."""
    return MedicalEntityRecognitionService(settings)


def get_modernbert_processor(
    settings: Settings = Depends(get_settings),
) -> ModernBERTProcessingService:
    """Provide a ModernBERT processor for dependency injection."""
    return ModernBERTProcessingService(settings)


def get_clinical_context_service(
    settings: Settings = Depends(get_settings),
) -> ClinicalContextService:
    """Provide a BioClinicalBERT/OpenMed semantic context service."""
    return ClinicalContextService(settings)


def get_fusion_service() -> FusionService:
    """Provide the Stage 7 fusion layer."""
    return FusionService()


def get_qwen_simplification_service(
    settings: Settings = Depends(get_settings),
) -> QwenSimplificationService:
    """Provide the Stage 8 Qwen3 simplification engine."""
    return QwenSimplificationService(settings)


def get_granite_guardian_validator(
    settings: Settings = Depends(get_settings),
) -> GraniteGuardianValidationService:
    """Provide the Stage 9 IBM Granite Guardian validator."""
    return GraniteGuardianValidationService(settings)


def get_evaluation_service(
    settings: Settings = Depends(get_settings),
) -> EvaluationService:
    """Provide the Stage 10 evaluation service."""
    return EvaluationService(settings)


def get_medical_report_pipeline(
    settings: Settings = Depends(get_settings),
    document_parser: DocumentParsingService = Depends(get_document_parser),
    section_segmenter: SectionSegmentationService = Depends(get_section_segmenter),
    entity_recognizer: MedicalEntityRecognitionService = Depends(get_entity_recognizer),
    modernbert_processor: ModernBERTProcessingService = Depends(get_modernbert_processor),
    clinical_context_service: ClinicalContextService = Depends(get_clinical_context_service),
    fusion_service: FusionService = Depends(get_fusion_service),
    qwen_service: QwenSimplificationService = Depends(get_qwen_simplification_service),
    validator: GraniteGuardianValidationService = Depends(get_granite_guardian_validator),
    evaluator: EvaluationService = Depends(get_evaluation_service),
) -> MedicalReportPipeline:
    """Provide the complete Stage 11 medical report pipeline."""
    return MedicalReportPipeline(
        settings=settings,
        document_parser=document_parser,
        section_segmenter=section_segmenter,
        entity_recognizer=entity_recognizer,
        modernbert_processor=modernbert_processor,
        clinical_context_service=clinical_context_service,
        fusion_service=fusion_service,
        qwen_service=qwen_service,
        validator=validator,
        evaluator=evaluator,
    )


@router.post("/extract", response_model=ParsedDocumentResponse)
async def extract_report_text(
    text: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
    settings: Settings = Depends(get_settings),
    parser: DocumentParsingService = Depends(get_document_parser),
) -> ParsedDocumentResponse:
    """Extract clean text from plain text, PDFs, scanned PDFs, or images."""
    if bool(text) == bool(file):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide exactly one input: either text or file.",
        )

    try:
        if text is not None:
            parsed_document = parser.parse_text(text)
        else:
            assert file is not None
            content = await file.read()
            max_upload_bytes = settings.max_upload_size_mb * 1024 * 1024
            if len(content) > max_upload_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=(
                        "Uploaded file exceeds the configured "
                        f"{settings.max_upload_size_mb} MB limit."
                    ),
                )
            parsed_document = parser.parse_file(
                filename=file.filename or "uploaded-report",
                content=content,
                content_type=file.content_type,
            )
    except DocumentParsingError as exc:
        logger.info("Document parsing failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return ParsedDocumentResponse(
        text=parsed_document.text,
        source_type=parsed_document.source_type.value,
        extraction_method=parsed_document.extraction_method.value,
        ocr_applied=parsed_document.ocr_applied,
        page_count=parsed_document.page_count,
        warnings=parsed_document.warnings,
    )


@router.post("/segment", response_model=SectionSegmentationResponse)
def segment_report_sections(
    request: SectionSegmentationRequest,
    segmenter: SectionSegmentationService = Depends(get_section_segmenter),
) -> SectionSegmentationResponse:
    """Split extracted report text into structured clinical sections."""
    try:
        segmented_report = segmenter.segment(request.text)
    except SectionSegmentationError as exc:
        logger.info("Section segmentation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return SectionSegmentationResponse(
        sections=[
            ClinicalSectionResponse(
                section_type=section.section_type.value,
                title=section.title,
                content=section.content,
                order=section.order,
                confidence=section.confidence,
            )
            for section in segmented_report.sections
        ],
        section_count=segmented_report.section_count,
        warnings=segmented_report.warnings,
    )


@router.post("/entities", response_model=EntityRecognitionResponse)
def extract_medical_entities(
    request: EntityRecognitionRequest,
    recognizer: MedicalEntityRecognitionService = Depends(get_entity_recognizer),
) -> EntityRecognitionResponse:
    """Extract diseases, symptoms, drugs, anatomy, procedures, and lab tests."""
    try:
        sections = [
            ClinicalSection(
                section_type=ClinicalSectionType(section.section_type),
                title=section.title,
                content=section.content,
                order=section.order,
                confidence=section.confidence,
            )
            for section in request.sections
        ]
        result = recognizer.extract(sections)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Request contains an unsupported section_type value.",
        ) from exc
    except EntityRecognitionError as exc:
        logger.info("Entity recognition failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return EntityRecognitionResponse(
        entities=[
            MedicalEntityResponse(
                text=entity.text,
                entity_type=entity.entity_type.value,
                section_type=entity.section_type.value,
                section_title=entity.section_title,
                start_char=entity.start_char,
                end_char=entity.end_char,
                confidence=entity.confidence,
                source_label=entity.source_label,
            )
            for entity in result.entities
        ],
        entity_count=result.entity_count,
        warnings=result.warnings,
    )


@router.post("/modernbert/difficult-terms", response_model=ModernBERTResponse)
def detect_difficult_terms(
    request: ModernBERTRequest,
    processor: ModernBERTProcessingService = Depends(get_modernbert_processor),
) -> ModernBERTResponse:
    """Identify difficult medical terms and contextual ModernBERT embeddings."""
    try:
        entities = [
            MedicalEntity(
                text=entity.text,
                entity_type=MedicalEntityType(entity.entity_type),
                section_type=ClinicalSectionType(entity.section_type),
                section_title=entity.section_title,
                start_char=entity.start_char,
                end_char=entity.end_char,
                confidence=entity.confidence,
                source_label=entity.source_label,
            )
            for entity in request.entities
        ]
        result = processor.process(entities)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Request contains an unsupported entity_type or section_type value.",
        ) from exc
    except ModernBERTProcessingError as exc:
        logger.info("ModernBERT processing failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return ModernBERTResponse(
        terms=[
            DifficultTermResponse(
                term=term.term,
                difficulty=term.difficulty,
                embedding=term.embedding,
                confidence=term.confidence,
                entity_type=term.entity_type.value,
                section_type=term.section_type,
                context=term.context,
            )
            for term in result.terms
        ],
        term_count=result.term_count,
        model_name=result.model_name,
        warnings=result.warnings,
    )


@router.post("/clinical-context", response_model=ClinicalContextResponse)
def understand_clinical_context(
    request: ClinicalContextRequest,
    service: ClinicalContextService = Depends(get_clinical_context_service),
) -> ClinicalContextResponse:
    """Resolve semantic meaning, clinical context, and ambiguity for entities."""
    try:
        entities = [
            MedicalEntity(
                text=entity.text,
                entity_type=MedicalEntityType(entity.entity_type),
                section_type=ClinicalSectionType(entity.section_type),
                section_title=entity.section_title,
                start_char=entity.start_char,
                end_char=entity.end_char,
                confidence=entity.confidence,
                source_label=entity.source_label,
            )
            for entity in request.entities
        ]
        result = service.interpret(entities)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Request contains an unsupported entity_type or section_type value.",
        ) from exc
    except ClinicalContextError as exc:
        logger.info("Clinical context processing failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return ClinicalContextResponse(
        interpretations=[
            SemanticInterpretationResponse(
                term=interpretation.term,
                meaning=interpretation.meaning,
                context=interpretation.context,
                ambiguity_resolution=interpretation.ambiguity_resolution,
                confidence=interpretation.confidence,
                entity_type=interpretation.entity_type.value,
                section_type=interpretation.section_type,
                semantic_embedding=interpretation.semantic_embedding,
                matched_concept=interpretation.matched_concept,
            )
            for interpretation in result.interpretations
        ],
        interpretation_count=result.interpretation_count,
        model_name=result.model_name,
        warnings=result.warnings,
    )


@router.post("/fusion", response_model=FusionResponse)
def fuse_model_outputs(
    request: FusionRequest,
    service: FusionService = Depends(get_fusion_service),
) -> FusionResponse:
    """Fuse ModernBERT and BioClinicalBERT/OpenMed outputs."""
    try:
        difficult_terms = [
            DifficultTerm(
                term=term.term,
                difficulty=term.difficulty,
                embedding=term.embedding,
                confidence=term.confidence,
                entity_type=MedicalEntityType(term.entity_type),
                section_type=term.section_type,
                context=term.context,
            )
            for term in request.difficult_terms
        ]
        semantic_interpretations = [
            SemanticInterpretation(
                term=interpretation.term,
                meaning=interpretation.meaning,
                context=interpretation.context,
                ambiguity_resolution=interpretation.ambiguity_resolution,
                confidence=interpretation.confidence,
                entity_type=MedicalEntityType(interpretation.entity_type),
                section_type=interpretation.section_type,
                semantic_embedding=interpretation.semantic_embedding,
                matched_concept=interpretation.matched_concept,
            )
            for interpretation in request.semantic_interpretations
        ]
        result = service.fuse(
            difficult_terms=difficult_terms,
            semantic_interpretations=semantic_interpretations,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Request contains an unsupported entity_type value.",
        ) from exc
    except FusionError as exc:
        logger.info("Fusion failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return FusionResponse(
        terms=[
            FusedMedicalTermResponse(
                term=term.term,
                difficulty=term.difficulty,
                meaning=term.meaning,
                context=term.context,
                confidence=term.confidence,
                entity_type=term.entity_type,
                section_type=term.section_type,
                modernbert_confidence=term.modernbert_confidence,
                semantic_confidence=term.semantic_confidence,
                ambiguity_resolution=term.ambiguity_resolution,
                modernbert_embedding=term.modernbert_embedding,
                semantic_embedding=term.semantic_embedding,
                matched_concept=term.matched_concept,
            )
            for term in result.terms
        ],
        term_count=result.term_count,
        warnings=result.warnings,
        algorithm_version=result.algorithm_version,
    )


@router.post("/simplify/from-fusion", response_model=SimplificationResponse)
def simplify_from_fusion(
    request: SimplificationFromFusionRequest,
    service: QwenSimplificationService = Depends(get_qwen_simplification_service),
) -> SimplificationResponse:
    """Generate a patient-friendly report from fused structured data only."""
    fused_terms = [
        FusedMedicalTerm(
            term=term.term,
            difficulty=term.difficulty,
            meaning=term.meaning,
            context=term.context,
            confidence=term.confidence,
            entity_type=term.entity_type,
            section_type=term.section_type,
            modernbert_confidence=term.modernbert_confidence,
            semantic_confidence=term.semantic_confidence,
            ambiguity_resolution=term.ambiguity_resolution,
            modernbert_embedding=term.modernbert_embedding,
            semantic_embedding=term.semantic_embedding,
            matched_concept=term.matched_concept,
        )
        for term in request.fused_terms
    ]

    try:
        result = service.simplify(fused_terms)
    except SimplificationError as exc:
        logger.info("Qwen3 simplification failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return SimplificationResponse(
        simplified_report=result.simplified_report,
        term_explanations=[
            TermExplanationResponse(
                term=explanation.term,
                explanation=explanation.explanation,
                difficulty=explanation.difficulty,
                confidence=explanation.confidence,
            )
            for explanation in result.term_explanations
        ],
        model_name=result.model_name,
        warnings=result.warnings,
    )


@router.post("/validate", response_model=ValidationResponse)
def validate_simplification(
    request: ValidationRequest,
    validator: GraniteGuardianValidationService = Depends(get_granite_guardian_validator),
) -> ValidationResponse:
    """Validate generated simplification with IBM Granite Guardian."""
    fused_terms = [
        FusedMedicalTerm(
            term=term.term,
            difficulty=term.difficulty,
            meaning=term.meaning,
            context=term.context,
            confidence=term.confidence,
            entity_type=term.entity_type,
            section_type=term.section_type,
            modernbert_confidence=term.modernbert_confidence,
            semantic_confidence=term.semantic_confidence,
            ambiguity_resolution=term.ambiguity_resolution,
            modernbert_embedding=term.modernbert_embedding,
            semantic_embedding=term.semantic_embedding,
            matched_concept=term.matched_concept,
        )
        for term in request.fused_terms
    ]

    try:
        result = validator.validate(
            fused_terms=fused_terms,
            simplified_report=request.simplified_report,
        )
    except ValidationError as exc:
        logger.info("Granite Guardian validation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return ValidationResponse(
        validation_passed=result.validation_passed,
        action=result.action.value,
        checks=[
            ValidationCheckResponse(
                name=check.name,
                passed=check.passed,
                score=check.score,
                details=check.details,
            )
            for check in result.checks
        ],
        guardian_assessment=GuardianAssessmentResponse(
            hallucination_risk=result.guardian_assessment.hallucination_risk,
            factual_consistency_risk=result.guardian_assessment.factual_consistency_risk,
            unsafe_content_risk=result.guardian_assessment.unsafe_content_risk,
            terminology_risk=result.guardian_assessment.terminology_risk,
            raw_response=result.guardian_assessment.raw_response,
        ),
        warnings=result.warnings,
    )


@router.post("/evaluate", response_model=EvaluationResponse)
def evaluate_simplification(
    request: EvaluationRequest,
    service: EvaluationService = Depends(get_evaluation_service),
) -> EvaluationResponse:
    """Evaluate simplified report quality."""
    fused_terms = [
        FusedMedicalTerm(
            term=term.term,
            difficulty=term.difficulty,
            meaning=term.meaning,
            context=term.context,
            confidence=term.confidence,
            entity_type=term.entity_type,
            section_type=term.section_type,
            modernbert_confidence=term.modernbert_confidence,
            semantic_confidence=term.semantic_confidence,
            ambiguity_resolution=term.ambiguity_resolution,
            modernbert_embedding=term.modernbert_embedding,
            semantic_embedding=term.semantic_embedding,
            matched_concept=term.matched_concept,
        )
        for term in request.fused_terms
    ]

    try:
        result = service.evaluate(
            reference_text=request.reference_text,
            simplified_report=request.simplified_report,
            fused_terms=fused_terms,
        )
    except EvaluationError as exc:
        logger.info("Evaluation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return EvaluationResponse(
        bertscore=BERTScoreResponse(
            precision=result.bertscore.precision,
            recall=result.bertscore.recall,
            f1=result.bertscore.f1,
        ),
        semantic_similarity=result.semantic_similarity,
        readability=ReadabilityResponse(
            flesch_kincaid_grade_level=result.readability.flesch_kincaid_grade_level,
            flesch_reading_ease=result.readability.flesch_reading_ease,
        ),
        medical_consistency=MedicalConsistencyResponse(
            score=result.medical_consistency.score,
            terms_preserved=result.medical_consistency.terms_preserved,
            meanings_preserved=result.medical_consistency.meanings_preserved,
            unsupported_numbers=result.medical_consistency.unsupported_numbers,
            missing_terms=result.medical_consistency.missing_terms,
            missing_meanings=result.medical_consistency.missing_meanings,
        ),
        warnings=result.warnings,
    )


@router.post("/simplify", response_model=FinalSimplificationResponse)
def simplify_report(
    request: SimplificationRequest,
    pipeline: MedicalReportPipeline = Depends(get_medical_report_pipeline),
) -> FinalSimplificationResponse:
    """Run the complete simplification pipeline for JSON text input."""
    try:
        result = pipeline.simplify_text(request.text)
    except PipelineError as exc:
        logger.info("Final simplification pipeline failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return _final_response_from_pipeline_result(result)


@router.post("/simplify/file", response_model=FinalSimplificationResponse)
async def simplify_report_file(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    pipeline: MedicalReportPipeline = Depends(get_medical_report_pipeline),
) -> FinalSimplificationResponse:
    """Run the complete simplification pipeline for PDF, image, or text files."""
    content = await file.read()
    max_upload_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                "Uploaded file exceeds the configured "
                f"{settings.max_upload_size_mb} MB limit."
            ),
        )
    try:
        result = pipeline.simplify_file(
            filename=file.filename or "uploaded-report",
            content=content,
            content_type=file.content_type,
        )
    except PipelineError as exc:
        logger.info("Final file simplification pipeline failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return _final_response_from_pipeline_result(result)


def _final_response_from_pipeline_result(result: object) -> FinalSimplificationResponse:
    """Convert a pipeline result into the final API response schema."""
    return FinalSimplificationResponse(
        simplified_report=result.simplified_report,
        highlighted_difficult_terms=[
            HighlightedDifficultTermResponse(
                term=term.term,
                difficulty=term.difficulty,
                confidence=term.confidence,
                entity_type=term.entity_type,
                section_type=term.section_type,
                meaning=term.meaning,
            )
            for term in result.highlighted_difficult_terms
        ],
        explanations=[
            TermExplanationResponse(
                term=str(explanation["term"]),
                explanation=str(explanation["explanation"]),
                difficulty=float(explanation["difficulty"]),
                confidence=float(explanation["confidence"]),
            )
            for explanation in result.explanations
        ],
        confidence=result.confidence,
        evaluation_scores=EvaluationResponse(
            bertscore=BERTScoreResponse(
                precision=result.evaluation.bertscore.precision,
                recall=result.evaluation.bertscore.recall,
                f1=result.evaluation.bertscore.f1,
            ),
            semantic_similarity=result.evaluation.semantic_similarity,
            readability=ReadabilityResponse(
                flesch_kincaid_grade_level=(
                    result.evaluation.readability.flesch_kincaid_grade_level
                ),
                flesch_reading_ease=result.evaluation.readability.flesch_reading_ease,
            ),
            medical_consistency=MedicalConsistencyResponse(
                score=result.evaluation.medical_consistency.score,
                terms_preserved=result.evaluation.medical_consistency.terms_preserved,
                meanings_preserved=result.evaluation.medical_consistency.meanings_preserved,
                unsupported_numbers=(
                    result.evaluation.medical_consistency.unsupported_numbers
                ),
                missing_terms=result.evaluation.medical_consistency.missing_terms,
                missing_meanings=result.evaluation.medical_consistency.missing_meanings,
            ),
            warnings=result.evaluation.warnings,
        ),
        validation=ValidationResponse(
            validation_passed=result.validation.validation_passed,
            action=result.validation.action.value,
            checks=[
                ValidationCheckResponse(
                    name=check.name,
                    passed=check.passed,
                    score=check.score,
                    details=check.details,
                )
                for check in result.validation.checks
            ],
            guardian_assessment=GuardianAssessmentResponse(
                hallucination_risk=result.validation.guardian_assessment.hallucination_risk,
                factual_consistency_risk=(
                    result.validation.guardian_assessment.factual_consistency_risk
                ),
                unsafe_content_risk=result.validation.guardian_assessment.unsafe_content_risk,
                terminology_risk=result.validation.guardian_assessment.terminology_risk,
                raw_response=result.validation.guardian_assessment.raw_response,
            ),
            warnings=result.validation.warnings,
        ),
        warnings=result.warnings,
    )
