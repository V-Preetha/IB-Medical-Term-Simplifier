"""SQLAlchemy 2.x declarative models for the IB Health persistence layer.

Tables `users` through `user_preferences` transcribe the authoritative
logical schema in `ARCHITECTURE.md` section 7 (sourced from `DB.pdf`);
column names/types/relationships are unchanged from that document except for
the additive physical mechanics `ARCHITECTURE.md` explicitly allows
(`created_at`/`updated_at`/`version` via `TimestampVersionMixin`, and
`deleted_at` via `SoftDeleteMixin` on a subset of tables).

Six new tables are additive per `AGENTS.md` ("add a table only when
necessary"): `entity_links` (entity-linking results, mirrors
`app/entity_linking/schemas.py`), `embedding_records` (embedding metadata
only -- never the raw vector -- mirrors `app/embeddings/schemas.py`),
`translations`, `processing_jobs` (Celery orchestration tracking),
`audit_logs` (append-only), and `model_registry` (reference data).
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.db.base import GUID, Base, SoftDeleteMixin, TimestampVersionMixin

# Portable JSON column: real JSONB on PostgreSQL, plain JSON on any other
# dialect (only used by the sqlite structural test fallback).
_JSONB = JSONB().with_variant(JSON(), "sqlite")


def _uuid_pk(name: str) -> Mapped[uuid.UUID]:
    return mapped_column(name, GUID(), primary_key=True, default=uuid.uuid4)


# --------------------------------------------------------------------------
# Core report processing (ARCHITECTURE.md section 7)
# --------------------------------------------------------------------------


class User(TimestampVersionMixin, Base):
    """A product user. No soft delete: account deletion is a compliance
    workflow handled outside this table (retention/erasure requests are
    tracked via `audit_logs`, not by hiding rows here)."""

    __tablename__ = "users"

    user_id: Mapped[uuid.UUID] = _uuid_pk("user_id")
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True)

    reports: Mapped[list["Report"]] = relationship(back_populates="user")
    voice_profiles: Mapped[list["VoiceProfile"]] = relationship(back_populates="user")
    preferences: Mapped[list["UserPreference"]] = relationship(back_populates="user")


class Report(TimestampVersionMixin, SoftDeleteMixin, Base):
    """An uploaded clinical document. Stores only a reference to encrypted
    object storage -- never the binary payload (`storage_url`)."""

    __tablename__ = "reports"

    report_id: Mapped[uuid.UUID] = _uuid_pk("report_id")
    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("users.user_id"), nullable=False, index=True
    )
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    storage_url: Mapped[str] = mapped_column(Text, nullable=False)
    file_type: Mapped[str] = mapped_column(String(64), nullable=False)
    language: Mapped[str | None] = mapped_column(String(32), nullable=True)
    upload_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)

    user: Mapped[User] = relationship(back_populates="reports")
    processing_attempts: Mapped[list["ReportProcessing"]] = relationship(back_populates="report")
    jobs: Mapped[list["ProcessingJob"]] = relationship(back_populates="report")

    __table_args__ = (Index("ix_reports_user_status", "user_id", "status"),)


class ReportProcessing(TimestampVersionMixin, Base):
    """A single processing attempt for a report. No soft delete: processing
    history is retained for auditability of the pipeline run."""

    __tablename__ = "report_processing"

    process_id: Mapped[uuid.UUID] = _uuid_pk("process_id")
    report_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("reports.report_id"), nullable=False, index=True
    )
    pipeline_version: Mapped[str] = mapped_column(String(64), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    processing_status: Mapped[str] = mapped_column(String(64), nullable=False)
    processing_time_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    ocr_required: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    ocr_success: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    raw_extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    report: Mapped[Report] = relationship(back_populates="processing_attempts")
    medical_entities: Mapped[list["MedicalEntity"]] = relationship(back_populates="process")
    model_outputs: Mapped[list["ModelOutput"]] = relationship(back_populates="process")
    simplifications: Mapped[list["Simplification"]] = relationship(back_populates="process")
    embedding_records: Mapped[list["EmbeddingRecord"]] = relationship(back_populates="process")
    translations: Mapped[list["Translation"]] = relationship(back_populates="process")


class MedicalEntity(TimestampVersionMixin, Base):
    """A recognized medical entity span produced by NER."""

    __tablename__ = "medical_entities"

    entity_id: Mapped[uuid.UUID] = _uuid_pk("entity_id")
    process_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("report_processing.process_id"), nullable=False, index=True
    )
    entity_text: Mapped[str] = mapped_column(Text, nullable=False)
    entity_type: Mapped[str] = mapped_column(String(128), nullable=False)
    start_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    end_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    process: Mapped[ReportProcessing] = relationship(back_populates="medical_entities")
    entity_links: Mapped[list["EntityLink"]] = relationship(back_populates="entity")

    __table_args__ = (
        CheckConstraint("end_offset > start_offset", name="ck_medical_entities_span"),
    )


class Simplification(TimestampVersionMixin, SoftDeleteMixin, Base):
    """A patient-facing simplified rendering of one processing attempt."""

    __tablename__ = "simplifications"

    simplification_id: Mapped[uuid.UUID] = _uuid_pk("simplification_id")
    process_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("report_processing.process_id"), nullable=False, index=True
    )
    predicted_level: Mapped[str | None] = mapped_column(String(64), nullable=True)
    prediction_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    simplified_text: Mapped[str] = mapped_column(Text, nullable=False)
    executive_summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    process: Mapped[ReportProcessing] = relationship(back_populates="simplifications")
    feedback_entries: Mapped[list["Feedback"]] = relationship(back_populates="simplification")
    voice_generations: Mapped[list["VoiceGeneration"]] = relationship(
        back_populates="simplification"
    )


class ModelOutput(TimestampVersionMixin, Base):
    """Per-stage model execution metadata for a processing attempt."""

    __tablename__ = "model_outputs"

    output_id: Mapped[uuid.UUID] = _uuid_pk("output_id")
    process_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("report_processing.process_id"), nullable=False, index=True
    )
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version: Mapped[str] = mapped_column(String(255), nullable=False)
    stage: Mapped[str] = mapped_column(String(64), nullable=False)
    execution_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    token_usage: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False)

    process: Mapped[ReportProcessing] = relationship(back_populates="model_outputs")


class Feedback(TimestampVersionMixin, Base):
    """Patient/clinician feedback on a simplification."""

    __tablename__ = "feedback"

    feedback_id: Mapped[uuid.UUID] = _uuid_pk("feedback_id")
    simplification_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("simplifications.simplification_id"), nullable=False, index=True
    )
    rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    comments: Mapped[str | None] = mapped_column(Text, nullable=True)

    simplification: Mapped[Simplification] = relationship(back_populates="feedback_entries")

    __table_args__ = (
        CheckConstraint("rating IS NULL OR (rating BETWEEN 1 AND 5)", name="ck_feedback_rating"),
    )


# --------------------------------------------------------------------------
# Preferences and speech (ARCHITECTURE.md section 7)
# --------------------------------------------------------------------------


class VoiceProfile(TimestampVersionMixin, Base):
    """A user's preferred voice/accent configuration for speech output."""

    __tablename__ = "voice_profiles"

    voice_profile_id: Mapped[uuid.UUID] = _uuid_pk("voice_profile_id")
    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("users.user_id"), nullable=False, index=True
    )
    preferred_language: Mapped[str | None] = mapped_column(String(32), nullable=True)
    preferred_accent: Mapped[str | None] = mapped_column(String(64), nullable=True)
    preferred_dialect: Mapped[str | None] = mapped_column(String(64), nullable=True)
    voice_sample_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    accent_detected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    user: Mapped[User] = relationship(back_populates="voice_profiles")
    voice_generations: Mapped[list["VoiceGeneration"]] = relationship(
        back_populates="voice_profile"
    )


class VoiceGeneration(TimestampVersionMixin, Base):
    """A synthesized audio artifact for a simplification. No binary audio
    is stored here -- only a reference to encrypted object storage."""

    __tablename__ = "voice_generations"

    generation_id: Mapped[uuid.UUID] = _uuid_pk("generation_id")
    simplification_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("simplifications.simplification_id"), nullable=False, index=True
    )
    voice_profile_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("voice_profiles.voice_profile_id"), nullable=False, index=True
    )
    tts_model: Mapped[str] = mapped_column(String(255), nullable=False)
    audio_url: Mapped[str] = mapped_column(Text, nullable=False)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    generation_status: Mapped[str] = mapped_column(String(64), nullable=False)

    simplification: Mapped[Simplification] = relationship(back_populates="voice_generations")
    voice_profile: Mapped[VoiceProfile] = relationship(back_populates="voice_generations")


class SupportedDialect(TimestampVersionMixin, Base):
    """Reference data: dialects/accents supported by speech output. No
    explicit FK relationship in the source diagram; activation is toggled
    via `is_active` rather than soft delete."""

    __tablename__ = "supported_dialects"

    dialect_id: Mapped[uuid.UUID] = _uuid_pk("dialect_id")
    language: Mapped[str] = mapped_column(String(32), nullable=False)
    accent: Mapped[str | None] = mapped_column(String(64), nullable=True)
    dialect_name: Mapped[str] = mapped_column(String(128), nullable=False)
    region: Mapped[str | None] = mapped_column(String(128), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    __table_args__ = (
        UniqueConstraint("language", "dialect_name", name="uq_supported_dialects_language_name"),
    )


class UserPreference(TimestampVersionMixin, Base):
    """A user's simplification/accessibility preferences."""

    __tablename__ = "user_preferences"

    preference_id: Mapped[uuid.UUID] = _uuid_pk("preference_id")
    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("users.user_id"), nullable=False, index=True
    )
    health_literacy_level: Mapped[str | None] = mapped_column(String(64), nullable=True)
    preferred_language: Mapped[str | None] = mapped_column(String(32), nullable=True)
    accessibility_mode: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    user: Mapped[User] = relationship(back_populates="preferences")


# --------------------------------------------------------------------------
# New tables (additive, Phase 3)
# --------------------------------------------------------------------------


class EntityLink(TimestampVersionMixin, SoftDeleteMixin, Base):
    """Entity-linking result for one `medical_entities` row. Column names
    mirror `app/entity_linking/schemas.py` (`ConceptCandidateSchema`,
    `EntityLinkSchema`): `cui` <- `concept_id`, `semantic_type_ids` <-
    `semantic_types`, `confidence_score` <- `confidence`, `review_state` <-
    `requires_review`/`status`, `ranked_candidates` <- `candidates`."""

    __tablename__ = "entity_links"

    entity_link_id: Mapped[uuid.UUID] = _uuid_pk("entity_link_id")
    entity_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("medical_entities.entity_id"), nullable=False, index=True
    )
    cui: Mapped[str | None] = mapped_column(String(64), nullable=True)
    preferred_name: Mapped[str | None] = mapped_column(String(512), nullable=True)
    semantic_type_ids: Mapped[list | None] = mapped_column(_JSONB, nullable=True)
    source_ontology: Mapped[str | None] = mapped_column(String(128), nullable=True)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_ambiguous: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    review_state: Mapped[str] = mapped_column(String(32), nullable=False, default="not_required")
    ranked_candidates: Mapped[list | None] = mapped_column(_JSONB, nullable=True)

    entity: Mapped[MedicalEntity] = relationship(back_populates="entity_links")

    __table_args__ = (
        CheckConstraint(
            "confidence_score IS NULL OR (confidence_score BETWEEN 0 AND 1)",
            name="ck_entity_links_confidence_range",
        ),
    )


class EmbeddingRecord(TimestampVersionMixin, SoftDeleteMixin, Base):
    """Embedding metadata for one processing attempt (and, optionally, one
    medical entity). The raw vector is never stored here -- only metadata
    plus `vector_store_ref`, an optional pointer into the separately scoped future
    Qdrant collection. Column names mirror
    `app/embeddings/schemas.py` (`EmbeddingVectorSchema`,
    `EmbeddingReproducibilitySchema`)."""

    __tablename__ = "embedding_records"

    embedding_record_id: Mapped[uuid.UUID] = _uuid_pk("embedding_record_id")
    process_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("report_processing.process_id"), nullable=False, index=True
    )
    entity_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("medical_entities.entity_id"), nullable=True, index=True
    )
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_revision: Mapped[str] = mapped_column(String(255), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    pooling_strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    normalized: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    vector_norm: Mapped[float | None] = mapped_column(Float, nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    vector_store_ref: Mapped[str | None] = mapped_column(Text, nullable=True)

    process: Mapped[ReportProcessing] = relationship(back_populates="embedding_records")

    __table_args__ = (
        CheckConstraint("dimensions > 0", name="ck_embedding_records_dimensions_positive"),
    )


class Translation(TimestampVersionMixin, SoftDeleteMixin, Base):
    """Translation output and reproducibility metadata for one processing attempt."""

    __tablename__ = "translations"

    translation_id: Mapped[uuid.UUID] = _uuid_pk("translation_id")
    process_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("report_processing.process_id"), nullable=False, index=True
    )
    source_language: Mapped[str] = mapped_column(String(32), nullable=False)
    target_language: Mapped[str] = mapped_column(String(32), nullable=False)
    translated_text: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_revision: Mapped[str] = mapped_column(String(255), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    processing_time_ms: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False)

    process: Mapped[ReportProcessing] = relationship(back_populates="translations")

    __table_args__ = (
        CheckConstraint(
            "confidence IS NULL OR (confidence BETWEEN 0 AND 1)",
            name="ck_translations_confidence_range",
        ),
        CheckConstraint(
            "processing_time_ms >= 0", name="ck_translations_processing_time_nonnegative"
        ),
    )


class ProcessingStage(str, enum.Enum):
    """Pipeline stages orchestrated by Celery."""

    OCR = "ocr"
    NER = "ner"
    ENTITY_LINKING = "entity_linking"
    EMBEDDINGS = "embeddings"
    SIMPLIFICATION = "simplification"
    TRANSLATION = "translation"


class ProcessingJobStatus(str, enum.Enum):
    """Lifecycle states for a `processing_jobs` row."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class ProcessingJob(TimestampVersionMixin, SoftDeleteMixin, Base):
    """Tracks one Celery task's orchestration state for one pipeline stage
    of one report. The Celery/Redis wiring itself is a Phase 4 deliverable;
    this table only records the durable job state that survives worker
    restarts."""

    __tablename__ = "processing_jobs"

    job_id: Mapped[uuid.UUID] = _uuid_pk("job_id")
    request_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)
    report_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("reports.report_id"), nullable=False, index=True
    )
    stage: Mapped[ProcessingStage] = mapped_column(
        Enum(
            ProcessingStage,
            name="processing_stage",
            values_callable=lambda choices: [choice.value for choice in choices],
        ),
        nullable=False,
    )
    celery_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    pipeline_version: Mapped[str] = mapped_column(String(64), nullable=False)
    configuration_version: Mapped[str] = mapped_column(String(128), nullable=False)
    model_revision: Mapped[str] = mapped_column(String(255), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    requested_stages: Mapped[list] = mapped_column(_JSONB, nullable=False)
    stage_statuses: Mapped[dict] = mapped_column(_JSONB, nullable=False)
    status: Mapped[ProcessingJobStatus] = mapped_column(
        Enum(
            ProcessingJobStatus,
            name="processing_job_status",
            values_callable=lambda choices: [choice.value for choice in choices],
        ),
        nullable=False,
        default=ProcessingJobStatus.PENDING,
    )
    progress_percent: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cancellation_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    error_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    queued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    report: Mapped[Report] = relationship(back_populates="jobs")

    __table_args__ = (
        CheckConstraint(
            "progress_percent BETWEEN 0 AND 100", name="ck_processing_jobs_progress_range"
        ),
        Index("ix_processing_jobs_report_stage", "report_id", "stage"),
    )


class AuditLog(TimestampVersionMixin, Base):
    """Append-only audit trail. NO soft delete and NO update path is exposed
    by `AuditLogRepository` (create/get/list only): an audit record that
    could be edited or hidden after the fact would defeat its purpose. The
    `version`/`updated_at` columns exist for schema consistency across all
    16 tables but are never touched after insert.

    Redaction policy: `metadata` MUST NOT contain raw clinical text (report
    bodies, extracted OCR text, entity text, simplified text, or audio).
    Only structural/operational facts belong here -- e.g. counts, status
    codes, durations, model/version identifiers, and resource IDs. Callers
    writing audit rows are responsible for enforcing this at the call site;
    this model only documents the constraint.
    """

    __tablename__ = "audit_logs"

    log_id: Mapped[uuid.UUID] = _uuid_pk("log_id")
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("users.user_id"), nullable=True, index=True
    )
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True)
    request_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    log_metadata: Mapped[dict | None] = mapped_column("metadata", _JSONB, nullable=True)

    __table_args__ = (Index("ix_audit_logs_resource", "resource_type", "resource_id"),)


class ModelRegistry(TimestampVersionMixin, Base):
    """Reference data mirroring `MODEL_MANIFEST.md` at the DB level. This
    table is a read model synced from deployment configuration. Clinical model choices
    are NOT hard-coded here -- rows are populated from the existing manifest
    files, never authored directly in application code."""

    __tablename__ = "model_registry"

    registry_id: Mapped[uuid.UUID] = _uuid_pk("registry_id")
    stage: Mapped[str] = mapped_column(String(64), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    revision: Mapped[str] = mapped_column(String(255), nullable=False)
    license: Mapped[str | None] = mapped_column(String(128), nullable=True)
    approval_status: Mapped[str] = mapped_column(String(32), nullable=False)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    source: Mapped[str] = mapped_column(String(16), nullable=False)

    __table_args__ = (
        UniqueConstraint("stage", "model_name", "revision", name="uq_model_registry_identity"),
        CheckConstraint("source IN ('config', 'manifest')", name="ck_model_registry_source"),
    )
