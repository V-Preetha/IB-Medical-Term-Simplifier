"""Concrete async SQLAlchemy implementations of `app.db.repositories.contracts`.

Every method uses SQLAlchemy Core/ORM expressions only -- no raw SQL string
concatenation. Ownership is enforced in the query itself (a `WHERE`/`JOIN`
clause on the owning `users.user_id`), not by filtering results in Python,
so an unauthorized `get_by_id`/`update`/`soft_delete` behaves identically to
"not found" and never leaks another tenant's row.

`version` is incremented here (not via a database trigger) on every
`update`/`soft_delete`; see `app/db/base.py` for the rationale.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.errors import ConflictError, DatabaseUnavailableError, NotFoundError
from app.db.models import (
    AuditLog,
    EmbeddingRecord,
    EntityLink,
    MedicalEntity,
    ModelRegistry,
    ProcessingJob,
    Report,
    ReportProcessing,
    Simplification,
    Translation,
)


class _SessionBoundRepository:
    """Shared flush/error-translation helpers for the concrete repositories."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def _flush(self, instance: Any) -> Any:
        try:
            self._session.add(instance)
            await self._session.flush()
        except IntegrityError as exc:
            raise ConflictError("The record conflicts with existing data.") from exc
        except SQLAlchemyError as exc:
            raise DatabaseUnavailableError(
                "The database operation could not be completed."
            ) from exc
        return instance

    async def _apply_update(self, instance: Any, changes: dict[str, Any]) -> Any:
        for key, value in changes.items():
            setattr(instance, key, value)
        instance.version = instance.version + 1
        try:
            await self._session.flush()
        except IntegrityError as exc:
            raise ConflictError("The record conflicts with existing data.") from exc
        except SQLAlchemyError as exc:
            raise DatabaseUnavailableError(
                "The database operation could not be completed."
            ) from exc
        return instance


class ReportRepositoryImpl(_SessionBoundRepository):
    async def create(self, **fields: Any) -> Report:
        return await self._flush(Report(**fields))

    async def get_by_id(self, report_id: UUID, *, owner_id: UUID) -> Report | None:
        stmt = select(Report).where(
            Report.report_id == report_id,
            Report.user_id == owner_id,
            Report.deleted_at.is_(None),
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def list(self, *, owner_id: UUID, limit: int, offset: int = 0) -> Sequence[Report]:
        stmt = (
            select(Report)
            .where(Report.user_id == owner_id, Report.deleted_at.is_(None))
            .order_by(Report.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return (await self._session.execute(stmt)).scalars().all()

    async def update(self, report_id: UUID, *, owner_id: UUID, **changes: Any) -> Report:
        instance = await self.get_by_id(report_id, owner_id=owner_id)
        if instance is None:
            raise NotFoundError(f"report {report_id} not found")
        return await self._apply_update(instance, changes)

    async def soft_delete(self, report_id: UUID, *, owner_id: UUID) -> bool:
        instance = await self.get_by_id(report_id, owner_id=owner_id)
        if instance is None:
            return False
        await self._apply_update(instance, {"deleted_at": datetime.now(UTC)})
        return True


class ReportProcessingRepositoryImpl(_SessionBoundRepository):
    async def create(self, **fields: Any) -> ReportProcessing:
        return await self._flush(ReportProcessing(**fields))

    async def get_by_id(self, process_id: UUID, *, owner_id: UUID) -> ReportProcessing | None:
        stmt = (
            select(ReportProcessing)
            .join(Report, ReportProcessing.report_id == Report.report_id)
            .where(ReportProcessing.process_id == process_id, Report.user_id == owner_id)
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def list(
        self, *, report_id: UUID, owner_id: UUID, limit: int, offset: int = 0
    ) -> Sequence[ReportProcessing]:
        stmt = (
            select(ReportProcessing)
            .join(Report, ReportProcessing.report_id == Report.report_id)
            .where(ReportProcessing.report_id == report_id, Report.user_id == owner_id)
            .order_by(ReportProcessing.started_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return (await self._session.execute(stmt)).scalars().all()

    async def update(self, process_id: UUID, *, owner_id: UUID, **changes: Any) -> ReportProcessing:
        instance = await self.get_by_id(process_id, owner_id=owner_id)
        if instance is None:
            raise NotFoundError(f"report_processing {process_id} not found")
        return await self._apply_update(instance, changes)


class MedicalEntityRepositoryImpl(_SessionBoundRepository):
    async def create(self, **fields: Any) -> MedicalEntity:
        return await self._flush(MedicalEntity(**fields))

    async def get_by_id(self, entity_id: UUID, *, owner_id: UUID) -> MedicalEntity | None:
        stmt = (
            select(MedicalEntity)
            .join(
                ReportProcessing,
                MedicalEntity.process_id == ReportProcessing.process_id,
            )
            .join(Report, ReportProcessing.report_id == Report.report_id)
            .where(MedicalEntity.entity_id == entity_id, Report.user_id == owner_id)
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def list(
        self, *, process_id: UUID, owner_id: UUID, limit: int, offset: int = 0
    ) -> Sequence[MedicalEntity]:
        stmt = (
            select(MedicalEntity)
            .join(
                ReportProcessing,
                MedicalEntity.process_id == ReportProcessing.process_id,
            )
            .join(Report, ReportProcessing.report_id == Report.report_id)
            .where(MedicalEntity.process_id == process_id, Report.user_id == owner_id)
            .order_by(MedicalEntity.start_offset.asc())
            .limit(limit)
            .offset(offset)
        )
        return (await self._session.execute(stmt)).scalars().all()

    async def update(self, entity_id: UUID, *, owner_id: UUID, **changes: Any) -> MedicalEntity:
        instance = await self.get_by_id(entity_id, owner_id=owner_id)
        if instance is None:
            raise NotFoundError(f"medical_entity {entity_id} not found")
        return await self._apply_update(instance, changes)


class SimplificationRepositoryImpl(_SessionBoundRepository):
    async def create(self, **fields: Any) -> Simplification:
        return await self._flush(Simplification(**fields))

    async def get_by_id(self, simplification_id: UUID, *, owner_id: UUID) -> Simplification | None:
        stmt = (
            select(Simplification)
            .join(
                ReportProcessing,
                Simplification.process_id == ReportProcessing.process_id,
            )
            .join(Report, ReportProcessing.report_id == Report.report_id)
            .where(
                Simplification.simplification_id == simplification_id,
                Report.user_id == owner_id,
                Simplification.deleted_at.is_(None),
            )
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def list(
        self, *, process_id: UUID, owner_id: UUID, limit: int, offset: int = 0
    ) -> Sequence[Simplification]:
        stmt = (
            select(Simplification)
            .join(
                ReportProcessing,
                Simplification.process_id == ReportProcessing.process_id,
            )
            .join(Report, ReportProcessing.report_id == Report.report_id)
            .where(
                Simplification.process_id == process_id,
                Report.user_id == owner_id,
                Simplification.deleted_at.is_(None),
            )
            .order_by(Simplification.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return (await self._session.execute(stmt)).scalars().all()

    async def update(
        self, simplification_id: UUID, *, owner_id: UUID, **changes: Any
    ) -> Simplification:
        instance = await self.get_by_id(simplification_id, owner_id=owner_id)
        if instance is None:
            raise NotFoundError(f"simplification {simplification_id} not found")
        return await self._apply_update(instance, changes)

    async def soft_delete(self, simplification_id: UUID, *, owner_id: UUID) -> bool:
        instance = await self.get_by_id(simplification_id, owner_id=owner_id)
        if instance is None:
            return False
        await self._apply_update(instance, {"deleted_at": datetime.now(UTC)})
        return True


class EntityLinkRepositoryImpl(_SessionBoundRepository):
    async def create(self, **fields: Any) -> EntityLink:
        return await self._flush(EntityLink(**fields))

    async def get_by_id(self, entity_link_id: UUID, *, owner_id: UUID) -> EntityLink | None:
        stmt = (
            select(EntityLink)
            .join(MedicalEntity, EntityLink.entity_id == MedicalEntity.entity_id)
            .join(
                ReportProcessing,
                MedicalEntity.process_id == ReportProcessing.process_id,
            )
            .join(Report, ReportProcessing.report_id == Report.report_id)
            .where(
                EntityLink.entity_link_id == entity_link_id,
                Report.user_id == owner_id,
                EntityLink.deleted_at.is_(None),
            )
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def list(
        self, *, entity_id: UUID, owner_id: UUID, limit: int, offset: int = 0
    ) -> Sequence[EntityLink]:
        stmt = (
            select(EntityLink)
            .join(MedicalEntity, EntityLink.entity_id == MedicalEntity.entity_id)
            .join(
                ReportProcessing,
                MedicalEntity.process_id == ReportProcessing.process_id,
            )
            .join(Report, ReportProcessing.report_id == Report.report_id)
            .where(
                EntityLink.entity_id == entity_id,
                Report.user_id == owner_id,
                EntityLink.deleted_at.is_(None),
            )
            .order_by(EntityLink.confidence_score.desc().nullslast())
            .limit(limit)
            .offset(offset)
        )
        return (await self._session.execute(stmt)).scalars().all()

    async def update(self, entity_link_id: UUID, *, owner_id: UUID, **changes: Any) -> EntityLink:
        instance = await self.get_by_id(entity_link_id, owner_id=owner_id)
        if instance is None:
            raise NotFoundError(f"entity_link {entity_link_id} not found")
        return await self._apply_update(instance, changes)

    async def soft_delete(self, entity_link_id: UUID, *, owner_id: UUID) -> bool:
        instance = await self.get_by_id(entity_link_id, owner_id=owner_id)
        if instance is None:
            return False
        await self._apply_update(instance, {"deleted_at": datetime.now(UTC)})
        return True


class EmbeddingRecordRepositoryImpl(_SessionBoundRepository):
    async def create(self, **fields: Any) -> EmbeddingRecord:
        return await self._flush(EmbeddingRecord(**fields))

    async def get_by_id(
        self, embedding_record_id: UUID, *, owner_id: UUID
    ) -> EmbeddingRecord | None:
        stmt = (
            select(EmbeddingRecord)
            .join(
                ReportProcessing,
                EmbeddingRecord.process_id == ReportProcessing.process_id,
            )
            .join(Report, ReportProcessing.report_id == Report.report_id)
            .where(
                EmbeddingRecord.embedding_record_id == embedding_record_id,
                Report.user_id == owner_id,
                EmbeddingRecord.deleted_at.is_(None),
            )
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def list(
        self, *, process_id: UUID, owner_id: UUID, limit: int, offset: int = 0
    ) -> Sequence[EmbeddingRecord]:
        stmt = (
            select(EmbeddingRecord)
            .join(
                ReportProcessing,
                EmbeddingRecord.process_id == ReportProcessing.process_id,
            )
            .join(Report, ReportProcessing.report_id == Report.report_id)
            .where(
                EmbeddingRecord.process_id == process_id,
                Report.user_id == owner_id,
                EmbeddingRecord.deleted_at.is_(None),
            )
            .order_by(EmbeddingRecord.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return (await self._session.execute(stmt)).scalars().all()

    async def update(
        self, embedding_record_id: UUID, *, owner_id: UUID, **changes: Any
    ) -> EmbeddingRecord:
        instance = await self.get_by_id(embedding_record_id, owner_id=owner_id)
        if instance is None:
            raise NotFoundError(f"embedding_record {embedding_record_id} not found")
        return await self._apply_update(instance, changes)

    async def soft_delete(self, embedding_record_id: UUID, *, owner_id: UUID) -> bool:
        instance = await self.get_by_id(embedding_record_id, owner_id=owner_id)
        if instance is None:
            return False
        await self._apply_update(instance, {"deleted_at": datetime.now(UTC)})
        return True


class TranslationRepositoryImpl(_SessionBoundRepository):
    async def create(self, **fields: Any) -> Translation:
        return await self._flush(Translation(**fields))

    async def get_by_id(self, translation_id: UUID, *, owner_id: UUID) -> Translation | None:
        stmt = (
            select(Translation)
            .join(ReportProcessing, Translation.process_id == ReportProcessing.process_id)
            .join(Report, ReportProcessing.report_id == Report.report_id)
            .where(
                Translation.translation_id == translation_id,
                Report.user_id == owner_id,
                Translation.deleted_at.is_(None),
            )
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def list(
        self, *, process_id: UUID, owner_id: UUID, limit: int, offset: int = 0
    ) -> Sequence[Translation]:
        stmt = (
            select(Translation)
            .join(ReportProcessing, Translation.process_id == ReportProcessing.process_id)
            .join(Report, ReportProcessing.report_id == Report.report_id)
            .where(
                Translation.process_id == process_id,
                Report.user_id == owner_id,
                Translation.deleted_at.is_(None),
            )
            .order_by(Translation.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return (await self._session.execute(stmt)).scalars().all()

    async def update(self, translation_id: UUID, *, owner_id: UUID, **changes: Any) -> Translation:
        instance = await self.get_by_id(translation_id, owner_id=owner_id)
        if instance is None:
            raise NotFoundError(f"translation {translation_id} not found")
        return await self._apply_update(instance, changes)

    async def soft_delete(self, translation_id: UUID, *, owner_id: UUID) -> bool:
        instance = await self.get_by_id(translation_id, owner_id=owner_id)
        if instance is None:
            return False
        await self._apply_update(instance, {"deleted_at": datetime.now(UTC)})
        return True


class ProcessingJobRepositoryImpl(_SessionBoundRepository):
    async def create(self, **fields: Any) -> ProcessingJob:
        return await self._flush(ProcessingJob(**fields))

    async def get_by_id(self, job_id: UUID, *, owner_id: UUID) -> ProcessingJob | None:
        stmt = (
            select(ProcessingJob)
            .join(Report, ProcessingJob.report_id == Report.report_id)
            .where(
                ProcessingJob.job_id == job_id,
                Report.user_id == owner_id,
                ProcessingJob.deleted_at.is_(None),
            )
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def list(
        self, *, report_id: UUID, owner_id: UUID, limit: int, offset: int = 0
    ) -> Sequence[ProcessingJob]:
        stmt = (
            select(ProcessingJob)
            .join(Report, ProcessingJob.report_id == Report.report_id)
            .where(
                ProcessingJob.report_id == report_id,
                Report.user_id == owner_id,
                ProcessingJob.deleted_at.is_(None),
            )
            .order_by(ProcessingJob.queued_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return (await self._session.execute(stmt)).scalars().all()

    async def update(self, job_id: UUID, *, owner_id: UUID, **changes: Any) -> ProcessingJob:
        instance = await self.get_by_id(job_id, owner_id=owner_id)
        if instance is None:
            raise NotFoundError(f"processing_job {job_id} not found")
        return await self._apply_update(instance, changes)

    async def soft_delete(self, job_id: UUID, *, owner_id: UUID) -> bool:
        instance = await self.get_by_id(job_id, owner_id=owner_id)
        if instance is None:
            return False
        await self._apply_update(instance, {"deleted_at": datetime.now(UTC)})
        return True


class AuditLogRepositoryImpl(_SessionBoundRepository):
    """No `update`/`soft_delete`: audit rows are append-only (see
    `app.db.models.AuditLog`)."""

    async def create(self, **fields: Any) -> AuditLog:
        return await self._flush(AuditLog(**fields))

    async def get_by_id(self, log_id: UUID) -> AuditLog | None:
        stmt = select(AuditLog).where(AuditLog.log_id == log_id)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def list(
        self,
        *,
        resource_type: str | None = None,
        resource_id: UUID | None = None,
        limit: int,
        offset: int = 0,
    ) -> Sequence[AuditLog]:
        stmt = select(AuditLog).order_by(AuditLog.occurred_at.desc())
        if resource_type is not None:
            stmt = stmt.where(AuditLog.resource_type == resource_type)
        if resource_id is not None:
            stmt = stmt.where(AuditLog.resource_id == resource_id)
        stmt = stmt.limit(limit).offset(offset)
        return (await self._session.execute(stmt)).scalars().all()


class ModelRegistryRepositoryImpl(_SessionBoundRepository):
    async def create(self, **fields: Any) -> ModelRegistry:
        return await self._flush(ModelRegistry(**fields))

    async def get_by_id(self, registry_id: UUID) -> ModelRegistry | None:
        stmt = select(ModelRegistry).where(ModelRegistry.registry_id == registry_id)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def list(
        self, *, stage: str | None = None, limit: int, offset: int = 0
    ) -> Sequence[ModelRegistry]:
        stmt = select(ModelRegistry).order_by(ModelRegistry.model_name.asc())
        if stage is not None:
            stmt = stmt.where(ModelRegistry.stage == stage)
        stmt = stmt.limit(limit).offset(offset)
        return (await self._session.execute(stmt)).scalars().all()

    async def update(self, registry_id: UUID, **changes: Any) -> ModelRegistry:
        instance = await self.get_by_id(registry_id)
        if instance is None:
            raise NotFoundError(f"model_registry {registry_id} not found")
        return await self._apply_update(instance, changes)
