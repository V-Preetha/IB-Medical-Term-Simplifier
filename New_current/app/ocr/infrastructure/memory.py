"""Bounded process-local adapters for Phase 2 execution and tests."""

import asyncio
from collections import OrderedDict
from collections.abc import Sequence
from types import TracebackType
from uuid import UUID

from app.ocr.domain.records import OCRRequestRecord, OCRResultRecord


class _MemoryState:
    def __init__(self, max_entries: int) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self.requests: OrderedDict[UUID, OCRRequestRecord] = OrderedDict()
        self.results: dict[UUID, tuple[UUID, OCRResultRecord]] = {}
        self.lock = asyncio.Lock()

    def trim(self) -> None:
        while len(self.requests) > self.max_entries:
            request_id, _ = self.requests.popitem(last=False)
            self.results.pop(request_id, None)


class MemoryOCRRequestRepository:
    def __init__(self, state: _MemoryState) -> None:
        self._state = state

    async def add(self, record: OCRRequestRecord) -> None:
        async with self._state.lock:
            if record.request_id in self._state.requests:
                raise ValueError("OCR request already exists")
            self._state.requests[record.request_id] = record
            self._state.trim()

    async def get(self, request_id: UUID, owner_id: UUID) -> OCRRequestRecord | None:
        async with self._state.lock:
            record = self._state.requests.get(request_id)
            return record if record is not None and record.owner_id == owner_id else None

    async def save(self, record: OCRRequestRecord) -> None:
        async with self._state.lock:
            current = self._state.requests.get(record.request_id)
            if current is not None and current.owner_id != record.owner_id:
                raise PermissionError("OCR request ownership cannot change")
            self._state.requests[record.request_id] = record
            self._state.requests.move_to_end(record.request_id)
            self._state.trim()

    async def delete(self, request_id: UUID, owner_id: UUID) -> bool:
        async with self._state.lock:
            record = self._state.requests.get(request_id)
            if record is None or record.owner_id != owner_id:
                return False
            del self._state.requests[request_id]
            self._state.results.pop(request_id, None)
            return True

    async def list_recent(self, owner_id: UUID, *, limit: int) -> Sequence[OCRRequestRecord]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        async with self._state.lock:
            records = [
                record
                for record in reversed(self._state.requests.values())
                if record.owner_id == owner_id
            ]
            return tuple(records[:limit])


class MemoryOCRResultRepository:
    def __init__(self, state: _MemoryState) -> None:
        self._state = state

    async def save(self, owner_id: UUID, result: OCRResultRecord) -> None:
        async with self._state.lock:
            self._state.results[result.request_id] = (owner_id, result)

    async def get(self, request_id: UUID, owner_id: UUID) -> OCRResultRecord | None:
        async with self._state.lock:
            stored = self._state.results.get(request_id)
            return stored[1] if stored is not None and stored[0] == owner_id else None

    async def delete(self, request_id: UUID, owner_id: UUID) -> bool:
        async with self._state.lock:
            stored = self._state.results.get(request_id)
            if stored is None or stored[0] != owner_id:
                return False
            del self._state.results[request_id]
            return True


class MemoryOCRUnitOfWork:
    """Shared process-local unit of work; PostgreSQL replaces it in Phase 3."""

    def __init__(self, max_entries: int = 512) -> None:
        state = _MemoryState(max_entries)
        self.requests = MemoryOCRRequestRepository(state)
        self.results = MemoryOCRResultRepository(state)

    async def __aenter__(self) -> "MemoryOCRUnitOfWork":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None


class MemoryOCRResultCache:
    """Bounded local adapter preserving the Redis replacement boundary."""

    def __init__(self, max_entries: int = 128) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self._entries: OrderedDict[str, OCRResultRecord] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> OCRResultRecord | None:
        async with self._lock:
            result = self._entries.get(key)
            if result is not None:
                self._entries.move_to_end(key)
            return result

    async def set(self, key: str, result: OCRResultRecord) -> None:
        async with self._lock:
            self._entries[key] = result
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._entries.pop(key, None)
