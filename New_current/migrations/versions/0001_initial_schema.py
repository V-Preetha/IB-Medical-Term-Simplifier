"""Initial schema: all 17 tables in app.db.models

Hand-transcribed column-by-column from `app/db/models.py` (no Postgres was
reachable in the authoring sandbox to run `alembic revision --autogenerate`;
see `New_current/INFRASTRUCTURE_IMPLEMENTATION_REPORT.md` for the exact
verification command a human should run against a live database). Native
Postgres types (`UUID`, `JSONB`, native `ENUM`) are used throughout, matching
`app/db/base.py::GUID` (which only degrades to `CHAR(32)` on non-Postgres
dialects) and the `_JSONB = JSONB().with_variant(JSON(), "sqlite")` column
helper in `app/db/models.py`.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-08-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _uuid_pk(name: str) -> sa.Column:
    return sa.Column(name, postgresql.UUID(as_uuid=True), primary_key=True)


def _timestamp_version_columns() -> list[sa.Column]:
    return [
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
    ]


def upgrade() -> None:
    # -- users ---------------------------------------------------------
    op.create_table(
        "users",
        _uuid_pk("user_id"),
        sa.Column("full_name", sa.String(255), nullable=False),
        sa.Column("email", sa.String(320), nullable=False, unique=True),
        *_timestamp_version_columns(),
    )

    # -- reports ---------------------------------------------------------
    op.create_table(
        "reports",
        _uuid_pk("report_id"),
        sa.Column(
            "user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.user_id"), nullable=False
        ),
        sa.Column("original_filename", sa.String(512), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("storage_url", sa.Text(), nullable=False),
        sa.Column("file_type", sa.String(64), nullable=False),
        sa.Column("language", sa.String(32), nullable=True),
        sa.Column("upload_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(64), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("file_size", sa.BigInteger(), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamp_version_columns(),
    )
    op.create_index("ix_reports_user_id", "reports", ["user_id"])
    op.create_index("ix_reports_user_status", "reports", ["user_id", "status"])

    # -- report_processing -----------------------------------------------
    op.create_table(
        "report_processing",
        _uuid_pk("process_id"),
        sa.Column(
            "report_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("reports.report_id"),
            nullable=False,
        ),
        sa.Column("pipeline_version", sa.String(64), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("processing_status", sa.String(64), nullable=False),
        sa.Column("processing_time_seconds", sa.Float(), nullable=True),
        sa.Column("ocr_required", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("ocr_success", sa.Boolean(), nullable=True),
        sa.Column("raw_extracted_text", sa.Text(), nullable=True),
        *_timestamp_version_columns(),
    )
    op.create_index("ix_report_processing_report_id", "report_processing", ["report_id"])

    # -- medical_entities --------------------------------------------------
    op.create_table(
        "medical_entities",
        _uuid_pk("entity_id"),
        sa.Column(
            "process_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("report_processing.process_id"),
            nullable=False,
        ),
        sa.Column("entity_text", sa.Text(), nullable=False),
        sa.Column("entity_type", sa.String(128), nullable=False),
        sa.Column("start_offset", sa.Integer(), nullable=False),
        sa.Column("end_offset", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.CheckConstraint("end_offset > start_offset", name="ck_medical_entities_span"),
        *_timestamp_version_columns(),
    )
    op.create_index("ix_medical_entities_process_id", "medical_entities", ["process_id"])

    # -- simplifications -----------------------------------------------------
    op.create_table(
        "simplifications",
        _uuid_pk("simplification_id"),
        sa.Column(
            "process_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("report_processing.process_id"),
            nullable=False,
        ),
        sa.Column("predicted_level", sa.String(64), nullable=True),
        sa.Column("prediction_confidence", sa.Float(), nullable=True),
        sa.Column("simplified_text", sa.Text(), nullable=False),
        sa.Column("executive_summary", sa.Text(), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamp_version_columns(),
    )
    op.create_index("ix_simplifications_process_id", "simplifications", ["process_id"])

    # -- model_outputs ---------------------------------------------------
    op.create_table(
        "model_outputs",
        _uuid_pk("output_id"),
        sa.Column(
            "process_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("report_processing.process_id"),
            nullable=False,
        ),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("model_version", sa.String(255), nullable=False),
        sa.Column("stage", sa.String(64), nullable=False),
        sa.Column("execution_time", sa.Float(), nullable=True),
        sa.Column("token_usage", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(64), nullable=False),
        *_timestamp_version_columns(),
    )
    op.create_index("ix_model_outputs_process_id", "model_outputs", ["process_id"])

    # -- feedback ---------------------------------------------------------
    op.create_table(
        "feedback",
        _uuid_pk("feedback_id"),
        sa.Column(
            "simplification_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("simplifications.simplification_id"),
            nullable=False,
        ),
        sa.Column("rating", sa.Integer(), nullable=True),
        sa.Column("comments", sa.Text(), nullable=True),
        sa.CheckConstraint("rating IS NULL OR (rating BETWEEN 1 AND 5)", name="ck_feedback_rating"),
        *_timestamp_version_columns(),
    )
    op.create_index("ix_feedback_simplification_id", "feedback", ["simplification_id"])

    # -- voice_profiles ------------------------------------------------
    op.create_table(
        "voice_profiles",
        _uuid_pk("voice_profile_id"),
        sa.Column(
            "user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.user_id"), nullable=False
        ),
        sa.Column("preferred_language", sa.String(32), nullable=True),
        sa.Column("preferred_accent", sa.String(64), nullable=True),
        sa.Column("preferred_dialect", sa.String(64), nullable=True),
        sa.Column("voice_sample_url", sa.Text(), nullable=True),
        sa.Column("accent_detected", sa.Boolean(), nullable=False, server_default=sa.false()),
        *_timestamp_version_columns(),
    )
    op.create_index("ix_voice_profiles_user_id", "voice_profiles", ["user_id"])

    # -- voice_generations -----------------------------------------------
    op.create_table(
        "voice_generations",
        _uuid_pk("generation_id"),
        sa.Column(
            "simplification_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("simplifications.simplification_id"),
            nullable=False,
        ),
        sa.Column(
            "voice_profile_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("voice_profiles.voice_profile_id"),
            nullable=False,
        ),
        sa.Column("tts_model", sa.String(255), nullable=False),
        sa.Column("audio_url", sa.Text(), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("generation_status", sa.String(64), nullable=False),
        *_timestamp_version_columns(),
    )
    op.create_index(
        "ix_voice_generations_simplification_id", "voice_generations", ["simplification_id"]
    )
    op.create_index(
        "ix_voice_generations_voice_profile_id", "voice_generations", ["voice_profile_id"]
    )

    # -- supported_dialects ------------------------------------------------
    op.create_table(
        "supported_dialects",
        _uuid_pk("dialect_id"),
        sa.Column("language", sa.String(32), nullable=False),
        sa.Column("accent", sa.String(64), nullable=True),
        sa.Column("dialect_name", sa.String(128), nullable=False),
        sa.Column("region", sa.String(128), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.UniqueConstraint("language", "dialect_name", name="uq_supported_dialects_language_name"),
        *_timestamp_version_columns(),
    )

    # -- user_preferences --------------------------------------------------
    op.create_table(
        "user_preferences",
        _uuid_pk("preference_id"),
        sa.Column(
            "user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.user_id"), nullable=False
        ),
        sa.Column("health_literacy_level", sa.String(64), nullable=True),
        sa.Column("preferred_language", sa.String(32), nullable=True),
        sa.Column("accessibility_mode", sa.Boolean(), nullable=False, server_default=sa.false()),
        *_timestamp_version_columns(),
    )
    op.create_index("ix_user_preferences_user_id", "user_preferences", ["user_id"])

    # -- entity_links (additive) --------------------------------------------
    op.create_table(
        "entity_links",
        _uuid_pk("entity_link_id"),
        sa.Column(
            "entity_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("medical_entities.entity_id"),
            nullable=False,
        ),
        sa.Column("cui", sa.String(64), nullable=True),
        sa.Column("preferred_name", sa.String(512), nullable=True),
        sa.Column("semantic_type_ids", postgresql.JSONB(), nullable=True),
        sa.Column("source_ontology", sa.String(128), nullable=True),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("is_ambiguous", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("review_state", sa.String(32), nullable=False, server_default="not_required"),
        sa.Column("ranked_candidates", postgresql.JSONB(), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "confidence_score IS NULL OR (confidence_score BETWEEN 0 AND 1)",
            name="ck_entity_links_confidence_range",
        ),
        *_timestamp_version_columns(),
    )
    op.create_index("ix_entity_links_entity_id", "entity_links", ["entity_id"])

    # -- embedding_records (additive) ---------------------------------------
    op.create_table(
        "embedding_records",
        _uuid_pk("embedding_record_id"),
        sa.Column(
            "process_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("report_processing.process_id"),
            nullable=False,
        ),
        sa.Column(
            "entity_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("medical_entities.entity_id"),
            nullable=True,
        ),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("model_revision", sa.String(255), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("pooling_strategy", sa.String(64), nullable=False),
        sa.Column("normalized", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("vector_norm", sa.Float(), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("vector_store_ref", sa.Text(), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("dimensions > 0", name="ck_embedding_records_dimensions_positive"),
        *_timestamp_version_columns(),
    )
    op.create_index("ix_embedding_records_process_id", "embedding_records", ["process_id"])
    op.create_index("ix_embedding_records_entity_id", "embedding_records", ["entity_id"])

    # -- translations (additive) --------------------------------------------
    op.create_table(
        "translations",
        _uuid_pk("translation_id"),
        sa.Column(
            "process_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("report_processing.process_id"),
            nullable=False,
        ),
        sa.Column("source_language", sa.String(32), nullable=False),
        sa.Column("target_language", sa.String(32), nullable=False),
        sa.Column("translated_text", sa.Text(), nullable=False),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("model_revision", sa.String(255), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("processing_time_ms", sa.Float(), nullable=False),
        sa.Column("status", sa.String(64), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "confidence IS NULL OR (confidence BETWEEN 0 AND 1)",
            name="ck_translations_confidence_range",
        ),
        sa.CheckConstraint(
            "processing_time_ms >= 0", name="ck_translations_processing_time_nonnegative"
        ),
        *_timestamp_version_columns(),
    )
    op.create_index("ix_translations_process_id", "translations", ["process_id"])

    # -- processing_jobs (additive) ------------------------------------------
    processing_stage = postgresql.ENUM(
        "ocr",
        "ner",
        "entity_linking",
        "embeddings",
        "simplification",
        "translation",
        name="processing_stage",
    )
    processing_job_status = postgresql.ENUM(
        "pending",
        "running",
        "succeeded",
        "failed",
        "retrying",
        "cancelled",
        name="processing_job_status",
    )
    # Enum types are created implicitly by `op.create_table` below (the
    # columns reference these `postgresql.ENUM` objects directly); no
    # separate explicit `.create()` call is made here to avoid emitting a
    # duplicate `CREATE TYPE` statement.
    op.create_table(
        "processing_jobs",
        _uuid_pk("job_id"),
        sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "report_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("reports.report_id"),
            nullable=False,
        ),
        sa.Column("stage", processing_stage, nullable=False),
        sa.Column("celery_task_id", sa.String(255), nullable=True),
        sa.Column("pipeline_version", sa.String(64), nullable=False),
        sa.Column("configuration_version", sa.String(128), nullable=False),
        sa.Column("model_revision", sa.String(255), nullable=False),
        sa.Column("idempotency_key", sa.String(64), nullable=False, unique=True),
        sa.Column("requested_stages", postgresql.JSONB(), nullable=False),
        sa.Column("stage_statuses", postgresql.JSONB(), nullable=False),
        sa.Column(
            "status",
            processing_job_status,
            nullable=False,
            server_default="pending",
        ),
        sa.Column("progress_percent", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "cancellation_requested", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column("error_code", sa.String(128), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("queued_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "progress_percent BETWEEN 0 AND 100", name="ck_processing_jobs_progress_range"
        ),
        *_timestamp_version_columns(),
    )
    op.create_index("ix_processing_jobs_report_id", "processing_jobs", ["report_id"])
    op.create_index("ix_processing_jobs_request_id", "processing_jobs", ["request_id"])
    op.create_index("ix_processing_jobs_report_stage", "processing_jobs", ["report_id", "stage"])

    # -- audit_logs (additive, append-only) ----------------------------------
    op.create_table(
        "audit_logs",
        _uuid_pk("log_id"),
        sa.Column(
            "actor_user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.user_id"),
            nullable=True,
        ),
        sa.Column("action", sa.String(128), nullable=False),
        sa.Column("resource_type", sa.String(64), nullable=False),
        sa.Column("resource_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        *_timestamp_version_columns(),
    )
    op.create_index("ix_audit_logs_actor_user_id", "audit_logs", ["actor_user_id"])
    op.create_index("ix_audit_logs_request_id", "audit_logs", ["request_id"])
    op.create_index("ix_audit_logs_resource", "audit_logs", ["resource_type", "resource_id"])

    # -- model_registry (additive, reference data) ---------------------------
    op.create_table(
        "model_registry",
        _uuid_pk("registry_id"),
        sa.Column("stage", sa.String(64), nullable=False),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("revision", sa.String(255), nullable=False),
        sa.Column("license", sa.String(128), nullable=True),
        sa.Column("approval_status", sa.String(32), nullable=False),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source", sa.String(16), nullable=False),
        sa.UniqueConstraint("stage", "model_name", "revision", name="uq_model_registry_identity"),
        sa.CheckConstraint("source IN ('config', 'manifest')", name="ck_model_registry_source"),
        *_timestamp_version_columns(),
    )


def downgrade() -> None:
    op.drop_table("model_registry")
    op.drop_table("audit_logs")
    op.drop_table("processing_jobs")
    bind = op.get_bind()
    postgresql.ENUM(name="processing_job_status").drop(bind, checkfirst=True)
    postgresql.ENUM(name="processing_stage").drop(bind, checkfirst=True)
    op.drop_table("translations")
    op.drop_table("embedding_records")
    op.drop_table("entity_links")
    op.drop_table("user_preferences")
    op.drop_table("supported_dialects")
    op.drop_table("voice_generations")
    op.drop_table("voice_profiles")
    op.drop_table("feedback")
    op.drop_table("model_outputs")
    op.drop_table("simplifications")
    op.drop_table("medical_entities")
    op.drop_table("report_processing")
    op.drop_table("reports")
    op.drop_table("users")
