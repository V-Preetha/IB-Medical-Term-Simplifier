"""Tests for the single OCR application orchestration path."""

import asyncio
from uuid import uuid4

from app.ocr.application.result_builder import OCRResultBuilder
from app.ocr.application.service import OCRApplicationService
from app.ocr.domain.records import OCRRequestStatus
from app.ocr.infrastructure.memory import MemoryOCRResultCache, MemoryOCRUnitOfWork
from tests.ocr.fakes import FakeOCRProvider, FakePostProcessor


def _service(ocr: FakeOCRProvider | None = None) -> OCRApplicationService:
    return OCRApplicationService(
        ocr_provider=ocr or FakeOCRProvider(),
        postprocessor=FakePostProcessor(),
        unit_of_work=MemoryOCRUnitOfWork(),
        cache=MemoryOCRResultCache(),
        result_builder=OCRResultBuilder(
            pipeline_version="test-pipeline-v1",
            schema_version="test-schema-v1",
        ),
    )


def test_application_service_executes_only_configured_provider_pipeline() -> None:
    service = _service()
    request_id = uuid4()
    owner_id = uuid4()
    result = asyncio.run(
        service.process(
            request_id=request_id,
            owner_id=owner_id,
            content=b"synthetic-image",
            filename="synthetic.png",
            file_type="png",
            media_type="image/png",
        )
    )

    assert result.request_id == request_id
    assert result.document_type == "scanned_pdf"
    assert result.provider_name == "qwen3-vl"
    assert result.normalized_text.endswith("hemoglobin 6.5%")
    assert result.confidence == 0.93
    assert result.page_count == 1
    assert result.cache_hit is False
    status = asyncio.run(service.get_status(request_id, owner_id))
    assert status is not None and status.status is OCRRequestStatus.COMPLETED


def test_duplicate_content_uses_versioned_cache() -> None:
    ocr = FakeOCRProvider()
    service = _service(ocr)
    owner_id = uuid4()
    first = asyncio.run(
        service.process(
            request_id=uuid4(),
            owner_id=owner_id,
            content=b"same-content",
            filename="one.png",
            file_type="png",
            media_type="image/png",
        )
    )
    second = asyncio.run(
        service.process(
            request_id=uuid4(),
            owner_id=owner_id,
            content=b"same-content",
            filename="two.png",
            file_type="png",
            media_type="image/png",
        )
    )

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.normalized_text == second.normalized_text
    assert first.request_id != second.request_id
    assert ocr.calls == 1


def test_delete_is_tenant_scoped() -> None:
    service = _service()
    owner_id = uuid4()
    request_id = uuid4()
    asyncio.run(
        service.process(
            request_id=request_id,
            owner_id=owner_id,
            content=b"tenant-content",
            filename="tenant.png",
            file_type="png",
            media_type="image/png",
        )
    )

    assert asyncio.run(service.delete(request_id, uuid4())) is False
    assert asyncio.run(service.delete(request_id, owner_id)) is True
    assert asyncio.run(service.get_result(request_id, owner_id)) is None
