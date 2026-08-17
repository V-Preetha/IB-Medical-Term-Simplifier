"""Persistence ports owned by the OCR application layer."""

from collections.abc import Sequence
from types import TracebackType
from typing import Protocol
from uuid import UUID

from app.ocr.domain.records import OCRRequestRecord, OCRResultRecord


class OCRRequestRepository(Protocol):
    """Tenant-scoped persistence contract for OCR request lifecycle state."""

    async def add(self, record: OCRRequestRecord) -> None: ...

    async def get(self, request_id: UUID, owner_id: UUID) -> OCRRequestRecord | None: ...

    async def save(self, record: OCRRequestRecord) -> None: ...

    async def delete(self, request_id: UUID, owner_id: UUID) -> bool: ...

    async def list_recent(
        self,
        owner_id: UUID,
        *,
        limit: int,
    ) -> Sequence[OCRRequestRecord]: ...


class OCRResultRepository(Protocol):
    """Tenant-scoped persistence contract for completed OCR output."""

    async def save(self, owner_id: UUID, result: OCRResultRecord) -> None: ...

    async def get(self, request_id: UUID, owner_id: UUID) -> OCRResultRecord | None: ...

    async def delete(self, request_id: UUID, owner_id: UUID) -> bool: ...


class OCRUnitOfWork(Protocol):
    """Atomic boundary for report, processing, and model-output persistence."""

    requests: OCRRequestRepository
    results: OCRResultRepository

    async def __aenter__(self) -> "OCRUnitOfWork": ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...
