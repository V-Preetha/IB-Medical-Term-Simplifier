"""Persistence-port contracts for the database layer.

Mirrors the shape of `app/ocr/application/repositories.py`
(`OCRRequestRepository`/`OCRResultRepository`/`OCRUnitOfWork`): every
repository is a `typing.Protocol`, so any conforming implementation
satisfies it structurally without inheritance. `app/db/repositories/
sqlalchemy_repositories.py` provides the concrete async SQLAlchemy
adapters; future in-memory fakes for other modules' tests can implement
the same Protocols without importing SQLAlchemy at all.

Repositories return ORM model instances directly rather than separate
domain dataclasses. This package IS the SQLAlchemy adapter boundary for
persistence (there is no further inward layer to shield from SQLAlchemy
types), so returning `app.db.models` instances keeps the surface small
without leaking SQLAlchemy *session* mechanics (queries, `Select`, etc.)
into callers.

Each repository exposes `create`, `get_by_id`, `list` (paginated, and
owner-scoped for every aggregate reachable from a `users` row), `update`,
and -- where the aggregate has `SoftDeleteMixin` -- `soft_delete`.
"""

from collections.abc import Sequence
from typing import Any, Protocol
from uuid import UUID

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


class ReportRepository(Protocol):
    """Owner-scoped persistence for `reports`."""

    async def create(self, **fields: Any) -> Report: ...

    async def get_by_id(self, report_id: UUID, *, owner_id: UUID) -> Report | None: ...

    async def list(self, *, owner_id: UUID, limit: int, offset: int = 0) -> Sequence[Report]: ...

    async def update(self, report_id: UUID, *, owner_id: UUID, **changes: Any) -> Report: ...

    async def soft_delete(self, report_id: UUID, *, owner_id: UUID) -> bool: ...


class ReportProcessingRepository(Protocol):
    """Owner-scoped (via parent report) persistence for `report_processing`."""

    async def create(self, **fields: Any) -> ReportProcessing: ...

    async def get_by_id(self, process_id: UUID, *, owner_id: UUID) -> ReportProcessing | None: ...

    async def list(
        self, *, report_id: UUID, owner_id: UUID, limit: int, offset: int = 0
    ) -> Sequence[ReportProcessing]: ...

    async def update(
        self, process_id: UUID, *, owner_id: UUID, **changes: Any
    ) -> ReportProcessing: ...


class MedicalEntityRepository(Protocol):
    """Owner-scoped (via process -> report) persistence for `medical_entities`."""

    async def create(self, **fields: Any) -> MedicalEntity: ...

    async def get_by_id(self, entity_id: UUID, *, owner_id: UUID) -> MedicalEntity | None: ...

    async def list(
        self, *, process_id: UUID, owner_id: UUID, limit: int, offset: int = 0
    ) -> Sequence[MedicalEntity]: ...

    async def update(self, entity_id: UUID, *, owner_id: UUID, **changes: Any) -> MedicalEntity: ...


class SimplificationRepository(Protocol):
    """Owner-scoped (via process -> report) persistence for `simplifications`."""

    async def create(self, **fields: Any) -> Simplification: ...

    async def get_by_id(
        self, simplification_id: UUID, *, owner_id: UUID
    ) -> Simplification | None: ...

    async def list(
        self, *, process_id: UUID, owner_id: UUID, limit: int, offset: int = 0
    ) -> Sequence[Simplification]: ...

    async def update(
        self, simplification_id: UUID, *, owner_id: UUID, **changes: Any
    ) -> Simplification: ...

    async def soft_delete(self, simplification_id: UUID, *, owner_id: UUID) -> bool: ...


class EntityLinkRepository(Protocol):
    """Owner-scoped (via entity -> process -> report) persistence for `entity_links`."""

    async def create(self, **fields: Any) -> EntityLink: ...

    async def get_by_id(self, entity_link_id: UUID, *, owner_id: UUID) -> EntityLink | None: ...

    async def list(
        self, *, entity_id: UUID, owner_id: UUID, limit: int, offset: int = 0
    ) -> Sequence[EntityLink]: ...

    async def update(
        self, entity_link_id: UUID, *, owner_id: UUID, **changes: Any
    ) -> EntityLink: ...

    async def soft_delete(self, entity_link_id: UUID, *, owner_id: UUID) -> bool: ...


class EmbeddingRecordRepository(Protocol):
    """Owner-scoped (via process -> report) persistence for `embedding_records`."""

    async def create(self, **fields: Any) -> EmbeddingRecord: ...

    async def get_by_id(
        self, embedding_record_id: UUID, *, owner_id: UUID
    ) -> EmbeddingRecord | None: ...

    async def list(
        self, *, process_id: UUID, owner_id: UUID, limit: int, offset: int = 0
    ) -> Sequence[EmbeddingRecord]: ...

    async def update(
        self, embedding_record_id: UUID, *, owner_id: UUID, **changes: Any
    ) -> EmbeddingRecord: ...

    async def soft_delete(self, embedding_record_id: UUID, *, owner_id: UUID) -> bool: ...


class TranslationRepository(Protocol):
    """Owner-scoped persistence for translated processing outputs."""

    async def create(self, **fields: Any) -> Translation: ...

    async def get_by_id(self, translation_id: UUID, *, owner_id: UUID) -> Translation | None: ...

    async def list(
        self, *, process_id: UUID, owner_id: UUID, limit: int, offset: int = 0
    ) -> Sequence[Translation]: ...

    async def update(
        self, translation_id: UUID, *, owner_id: UUID, **changes: Any
    ) -> Translation: ...

    async def soft_delete(self, translation_id: UUID, *, owner_id: UUID) -> bool: ...


class ProcessingJobRepository(Protocol):
    """Owner-scoped (via report) persistence for `processing_jobs`."""

    async def create(self, **fields: Any) -> ProcessingJob: ...

    async def get_by_id(self, job_id: UUID, *, owner_id: UUID) -> ProcessingJob | None: ...

    async def list(
        self, *, report_id: UUID, owner_id: UUID, limit: int, offset: int = 0
    ) -> Sequence[ProcessingJob]: ...

    async def update(self, job_id: UUID, *, owner_id: UUID, **changes: Any) -> ProcessingJob: ...

    async def soft_delete(self, job_id: UUID, *, owner_id: UUID) -> bool: ...


class AuditLogRepository(Protocol):
    """Append-only persistence for `audit_logs`. No `update`/`soft_delete`:
    audit rows must never be mutated or hidden after insertion."""

    async def create(self, **fields: Any) -> AuditLog: ...

    async def get_by_id(self, log_id: UUID) -> AuditLog | None: ...

    async def list(
        self,
        *,
        resource_type: str | None = None,
        resource_id: UUID | None = None,
        limit: int,
        offset: int = 0,
    ) -> Sequence[AuditLog]: ...


class ModelRegistryRepository(Protocol):
    """Reference-data persistence for `model_registry` (not owner-scoped)."""

    async def create(self, **fields: Any) -> ModelRegistry: ...

    async def get_by_id(self, registry_id: UUID) -> ModelRegistry | None: ...

    async def list(
        self, *, stage: str | None = None, limit: int, offset: int = 0
    ) -> Sequence[ModelRegistry]: ...

    async def update(self, registry_id: UUID, **changes: Any) -> ModelRegistry: ...
