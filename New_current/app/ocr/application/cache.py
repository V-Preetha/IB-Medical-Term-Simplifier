"""Cache port for deterministic OCR results."""

from typing import Protocol

from app.ocr.domain.records import OCRResultRecord


class OCRResultCache(Protocol):
    """Version-aware cache boundary implemented locally now and by Redis in Phase 4."""

    async def get(self, key: str) -> OCRResultRecord | None: ...

    async def set(self, key: str, result: OCRResultRecord) -> None: ...

    async def delete(self, key: str) -> None: ...
