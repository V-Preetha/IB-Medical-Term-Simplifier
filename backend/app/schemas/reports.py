"""Schemas for report parsing and simplification API boundaries."""

from pydantic import BaseModel, ConfigDict, Field


class ParsedDocumentResponse(BaseModel):
    """Response returned after Stage 2 document parsing."""

    model_config = ConfigDict(extra="forbid")

    text: str
    source_type: str
    extraction_method: str
    ocr_applied: bool
    page_count: int | None = None
    warnings: list[str] = Field(default_factory=list)


class SectionSegmentationRequest(BaseModel):
    """Request body for Stage 3 clinical section segmentation."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, description="Clean extracted report text.")


class ClinicalSectionResponse(BaseModel):
    """A normalized clinical report section."""

    model_config = ConfigDict(extra="forbid")

    section_type: str
    title: str
    content: str
    order: int
    confidence: float


class SectionSegmentationResponse(BaseModel):
    """Structured clinical sections returned by Stage 3."""

    model_config = ConfigDict(extra="forbid")

    sections: list[ClinicalSectionResponse]
    section_count: int
    warnings: list[str] = Field(default_factory=list)


class EntityRecognitionRequest(BaseModel):
    """Request body for Stage 4 medical entity recognition."""

    model_config = ConfigDict(extra="forbid")

    sections: list[ClinicalSectionResponse] = Field(..., min_length=1)


class MedicalEntityResponse(BaseModel):
    """A structured medical entity detected by SciSpaCy."""

    model_config = ConfigDict(extra="forbid")

    text: str
    entity_type: str
    section_type: str
    section_title: str
    start_char: int
    end_char: int
    confidence: float
    source_label: str | None = None


class EntityRecognitionResponse(BaseModel):
    """Structured medical entity recognition result."""

    model_config = ConfigDict(extra="forbid")

    entities: list[MedicalEntityResponse]
    entity_count: int
    warnings: list[str] = Field(default_factory=list)


class ModernBERTRequest(BaseModel):
    """Request body for Stage 5 ModernBERT processing."""

    model_config = ConfigDict(extra="forbid")

    entities: list[MedicalEntityResponse] = Field(..., min_length=1)


class DifficultTermResponse(BaseModel):
    """A difficult medical term detected by ModernBERT processing."""

    model_config = ConfigDict(extra="forbid")

    term: str
    difficulty: float
    embedding: list[float]
    confidence: float
    entity_type: str
    section_type: str
    context: str


class ModernBERTResponse(BaseModel):
    """ModernBERT difficult-term detection response."""

    model_config = ConfigDict(extra="forbid")

    terms: list[DifficultTermResponse]
    term_count: int
    model_name: str
    warnings: list[str] = Field(default_factory=list)


class ClinicalContextRequest(BaseModel):
    """Request body for Stage 6 BioClinicalBERT/OpenMed processing."""

    model_config = ConfigDict(extra="forbid")

    entities: list[MedicalEntityResponse] = Field(..., min_length=1)


class SemanticInterpretationResponse(BaseModel):
    """Semantic meaning and context for one medical term."""

    model_config = ConfigDict(extra="forbid")

    term: str
    meaning: str
    context: str
    ambiguity_resolution: str
    confidence: float
    entity_type: str
    section_type: str
    semantic_embedding: list[float]
    matched_concept: str | None = None


class ClinicalContextResponse(BaseModel):
    """BioClinicalBERT/OpenMed semantic understanding response."""

    model_config = ConfigDict(extra="forbid")

    interpretations: list[SemanticInterpretationResponse]
    interpretation_count: int
    model_name: str
    warnings: list[str] = Field(default_factory=list)


class FusionRequest(BaseModel):
    """Request body for Stage 7 model-output fusion."""

    model_config = ConfigDict(extra="forbid")

    difficult_terms: list[DifficultTermResponse] = Field(..., min_length=1)
    semantic_interpretations: list[SemanticInterpretationResponse] = Field(
        ...,
        min_length=1,
    )


class FusedMedicalTermResponse(BaseModel):
    """Unified structured medical representation for one term."""

    model_config = ConfigDict(extra="forbid")

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


class FusionResponse(BaseModel):
    """Fused structured medical representation returned by Stage 7."""

    model_config = ConfigDict(extra="forbid")

    terms: list[FusedMedicalTermResponse]
    term_count: int
    warnings: list[str] = Field(default_factory=list)
    algorithm_version: str


class SimplificationFromFusionRequest(BaseModel):
    """Request body for Stage 8 Qwen3 simplification."""

    model_config = ConfigDict(extra="forbid")

    fused_terms: list[FusedMedicalTermResponse] = Field(..., min_length=1)


class TermExplanationResponse(BaseModel):
    """Patient-friendly explanation for one difficult term."""

    model_config = ConfigDict(extra="forbid")

    term: str
    explanation: str
    difficulty: float
    confidence: float


class SimplificationResponse(BaseModel):
    """Qwen3 patient-friendly simplification response."""

    model_config = ConfigDict(extra="forbid")

    simplified_report: str
    term_explanations: list[TermExplanationResponse]
    model_name: str
    warnings: list[str] = Field(default_factory=list)


class ValidationRequest(BaseModel):
    """Request body for Stage 9 Granite Guardian validation."""

    model_config = ConfigDict(extra="forbid")

    fused_terms: list[FusedMedicalTermResponse] = Field(..., min_length=1)
    simplified_report: str = Field(..., min_length=1)


class ValidationCheckResponse(BaseModel):
    """One validation check returned by Granite Guardian validation."""

    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    score: float
    details: str


class GuardianAssessmentResponse(BaseModel):
    """Granite Guardian model risk scores."""

    model_config = ConfigDict(extra="forbid")

    hallucination_risk: float
    factual_consistency_risk: float
    unsafe_content_risk: float
    terminology_risk: float
    raw_response: str


class ValidationResponse(BaseModel):
    """Stage 9 validation response."""

    model_config = ConfigDict(extra="forbid")

    validation_passed: bool
    action: str
    checks: list[ValidationCheckResponse]
    guardian_assessment: GuardianAssessmentResponse
    warnings: list[str] = Field(default_factory=list)


class EvaluationRequest(BaseModel):
    """Request body for Stage 10 evaluation metrics."""

    model_config = ConfigDict(extra="forbid")

    reference_text: str = Field(..., min_length=1)
    simplified_report: str = Field(..., min_length=1)
    fused_terms: list[FusedMedicalTermResponse] = Field(..., min_length=1)


class BERTScoreResponse(BaseModel):
    """BERTScore precision, recall, and F1 response."""

    model_config = ConfigDict(extra="forbid")

    precision: float
    recall: float
    f1: float


class ReadabilityResponse(BaseModel):
    """Readability metrics response."""

    model_config = ConfigDict(extra="forbid")

    flesch_kincaid_grade_level: float
    flesch_reading_ease: float


class MedicalConsistencyResponse(BaseModel):
    """Medical consistency metrics response."""

    model_config = ConfigDict(extra="forbid")

    score: float
    terms_preserved: bool
    meanings_preserved: bool
    unsupported_numbers: list[str]
    missing_terms: list[str]
    missing_meanings: list[str]


class EvaluationResponse(BaseModel):
    """Stage 10 evaluation response."""

    model_config = ConfigDict(extra="forbid")

    bertscore: BERTScoreResponse
    semantic_similarity: float
    readability: ReadabilityResponse
    medical_consistency: MedicalConsistencyResponse
    warnings: list[str] = Field(default_factory=list)


class HighlightedDifficultTermResponse(BaseModel):
    """Highlighted difficult term returned by the final API."""

    model_config = ConfigDict(extra="forbid")

    term: str
    difficulty: float
    confidence: float
    entity_type: str
    section_type: str
    meaning: str


class FinalSimplificationResponse(BaseModel):
    """Final API response for the full simplification pipeline."""

    model_config = ConfigDict(extra="forbid")

    simplified_report: str
    highlighted_difficult_terms: list[HighlightedDifficultTermResponse]
    explanations: list[TermExplanationResponse]
    confidence: float
    evaluation_scores: EvaluationResponse
    validation: ValidationResponse
    warnings: list[str] = Field(default_factory=list)


class SimplificationRequest(BaseModel):
    """Initial API contract for plain-text report submission.

    Full simplification is intentionally deferred to later stages.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, description="Raw report text.")
