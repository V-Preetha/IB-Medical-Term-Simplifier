"""Single application service orchestrating the approved OCR pipeline."""

import asyncio
import logging
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from time import perf_counter
from uuid import UUID, uuid4

from app.ocr.application.cache import OCRResultCache
from app.ocr.application.repositories import OCRUnitOfWork
from app.ocr.application.result_builder import OCRResultBuilder
from app.ocr.domain.records import OCRRequestRecord, OCRRequestStatus, OCRResultRecord
from app.ocr.providers.contracts import (
    BaseOCRProvider,
    BasePostProcessor,
    ProviderDocument,
)
from app.ocr.providers.errors import ProviderError

logger = logging.getLogger(__name__)


class OCRApplicationService:
    """Coordinate providers without exposing model details to HTTP routes."""

    def __init__(
        self,
        *,
        ocr_provider: BaseOCRProvider,
        postprocessor: BasePostProcessor,
        unit_of_work: OCRUnitOfWork,
        cache: OCRResultCache,
        result_builder: OCRResultBuilder,
    ) -> None:
        self._ocr = ocr_provider
        self._postprocessor = postprocessor
        self._uow = unit_of_work
        self._cache = cache
        self._result_builder = result_builder

    async def process(
        self,
        *,
        request_id: UUID,
        owner_id: UUID,
        content: bytes,
        filename: str,
        file_type: str,
        media_type: str,
        upload_time_ms: float | None = None,
    ) -> OCRResultRecord:
        content_digest = sha256(content).hexdigest()
        report_id = uuid4()
        ocr_id = uuid4()
        queued = OCRRequestRecord.queued(
            request_id=request_id,
            report_id=report_id,
            process_id=ocr_id,
            owner_id=owner_id,
            content_sha256=content_digest,
            original_filename=filename,
            media_type=media_type,
        )
        async with self._uow as uow:
            await uow.requests.add(queued)
            await uow.commit()
        await self._save_request(
            replace(
                queued,
                status=OCRRequestStatus.RUNNING,
                stage="cache_lookup",
                progress_percent=10,
                updated_at=datetime.now(UTC),
            )
        )
        cache_key = self._cache_key(owner_id, content_digest)
        cached = await self._cache.get(cache_key)
        if cached is not None:
            result = replace(
                cached,
                request_id=request_id,
                report_id=report_id,
                process_id=ocr_id,
                cache_hit=True,
                created_at=datetime.now(UTC),
            )
            await self._complete_request(queued, result)
            self._log_stage(
                request_id=request_id,
                stage="cache_hit",
                provider=result.provider_name,
                provider_version=result.provider_version,
                processing_time_ms=0.0,
                confidence=result.confidence,
                cache_hit=True,
            )
            return result

        document = ProviderDocument(
            content=content,
            file_type=file_type,
            filename=filename,
        )
        total_started = perf_counter()
        try:
            await self._update_stage(queued, "ocr", 50)
            ocr_result = await self._run_stage(
                request_id,
                "ocr",
                self._ocr.metadata().provider_name,
                self._ocr.metadata().provider_version,
                self._ocr.process,
                document,
            )
            await self._update_stage(queued, "postprocessing", 80)
            postprocessed = await self._run_stage(
                request_id,
                "postprocessing",
                self._postprocessor.metadata().provider_name,
                self._postprocessor.metadata().provider_version,
                self._postprocessor.normalize,
                ocr_result.text,
                document_type=ocr_result.document_type,
            )
            page_count = _page_count(ocr_result)
            elapsed_ms = round((perf_counter() - total_started) * 1000, 3)
            result = self._result_builder.build(
                request_id=request_id,
                report_id=report_id,
                ocr_id=ocr_id,
                ocr_result=ocr_result,
                postprocessed=postprocessed,
                ocr_metadata=self._ocr.metadata(),
                postprocessor_metadata=self._postprocessor.metadata(),
                processing_time_ms=elapsed_ms,
                cache_hit=False,
                metadata={"page_count": page_count},
                upload_time_ms=upload_time_ms,
            )
            await self._cache.set(cache_key, result)
            await self._complete_request(queued, result)
            return result
        except ProviderError as exc:
            await self._fail_request(queued, exc.code, exc.message)
            self._log_failure(request_id, "ocr_pipeline", exc.code)
            raise
        except Exception:
            await self._fail_request(
                queued,
                "ocr_processing_failed",
                "The OCR request could not be processed.",
            )
            self._log_failure(request_id, "ocr_pipeline", "ocr_processing_failed")
            raise

    async def get_result(self, request_id: UUID, owner_id: UUID) -> OCRResultRecord | None:
        async with self._uow as uow:
            return await uow.results.get(request_id, owner_id)

    async def get_status(self, request_id: UUID, owner_id: UUID) -> OCRRequestRecord | None:
        async with self._uow as uow:
            return await uow.requests.get(request_id, owner_id)

    async def delete(self, request_id: UUID, owner_id: UUID) -> bool:
        async with self._uow as uow:
            request = await uow.requests.get(request_id, owner_id)
            if request is None:
                return False
            await uow.results.delete(request_id, owner_id)
            deleted = await uow.requests.delete(request_id, owner_id)
            await uow.commit()
        await self._cache.delete(self._cache_key(owner_id, request.content_sha256))
        return deleted

    async def list_recent(self, owner_id: UUID, *, limit: int) -> tuple[OCRRequestRecord, ...]:
        async with self._uow as uow:
            return tuple(await uow.requests.list_recent(owner_id, limit=limit))

    def provider_metadata(self):
        """Return configuration-safe provider metadata for inspection APIs."""

        return tuple(provider.metadata() for provider in (self._ocr, self._postprocessor))

    def provider_health(self):
        """Return provider health without executing inference."""

        return tuple(provider.health() for provider in (self._ocr, self._postprocessor))

    def _cache_key(self, owner_id: UUID, content_digest: str) -> str:
        identities = []
        for provider in (self._ocr, self._postprocessor):
            metadata = provider.metadata()
            identities.append(
                ":".join(
                    (
                        metadata.provider_name,
                        metadata.provider_version,
                        str(metadata.configuration.get("model_revision", "none")),
                        str(metadata.configuration.get("prompt_version", "none")),
                        str(metadata.configuration.get("rule_version", "none")),
                    )
                )
            )
        identity = "|".join(
            (
                str(owner_id),
                content_digest,
                self._result_builder.pipeline_version,
                self._result_builder.schema_version,
                *identities,
            )
        )
        return f"ocr:{sha256(identity.encode('utf-8')).hexdigest()}"

    async def _run_stage(
        self,
        request_id: UUID,
        stage: str,
        provider: str,
        version: str,
        func,
        *args,
        **kwargs,
    ):
        started = perf_counter()
        result = await asyncio.to_thread(func, *args, **kwargs)
        elapsed_ms = round((perf_counter() - started) * 1000, 3)
        self._log_stage(
            request_id=request_id,
            stage=stage,
            provider=provider,
            provider_version=version,
            processing_time_ms=elapsed_ms,
            confidence=getattr(result, "confidence", None),
            cache_hit=False,
        )
        return result

    async def _save_request(self, record: OCRRequestRecord) -> None:
        async with self._uow as uow:
            await uow.requests.save(record)
            await uow.commit()

    async def _update_stage(self, original: OCRRequestRecord, stage: str, progress: int) -> None:
        await self._save_request(
            replace(
                original,
                status=OCRRequestStatus.RUNNING,
                stage=stage,
                progress_percent=progress,
                updated_at=datetime.now(UTC),
            )
        )

    async def _complete_request(self, original: OCRRequestRecord, result: OCRResultRecord) -> None:
        now = datetime.now(UTC)
        completed = replace(
            original,
            status=OCRRequestStatus.COMPLETED,
            stage="completed",
            progress_percent=100,
            updated_at=now,
            completed_at=now,
        )
        async with self._uow as uow:
            await uow.results.save(original.owner_id, result)
            await uow.requests.save(completed)
            await uow.commit()

    async def _fail_request(self, original: OCRRequestRecord, code: str, message: str) -> None:
        await self._save_request(
            replace(
                original,
                status=OCRRequestStatus.FAILED,
                stage="failed",
                progress_percent=100,
                updated_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                error_code=code,
                error_message=message,
            )
        )

    @staticmethod
    def _log_stage(
        *,
        request_id: UUID,
        stage: str,
        provider: str,
        provider_version: str,
        processing_time_ms: float,
        confidence: float | None,
        cache_hit: bool,
    ) -> None:
        logger.info(
            "OCR pipeline stage completed",
            extra={
                "event": "ocr_stage_completed",
                "request_id": str(request_id),
                "pipeline_stage": stage,
                "provider": provider,
                "provider_version": provider_version,
                "processing_time_ms": processing_time_ms,
                "latency_ms": processing_time_ms,
                "confidence": confidence,
                "cache_hit": cache_hit,
            },
        )

    @staticmethod
    def _log_failure(request_id: UUID, stage: str, error_code: str) -> None:
        logger.exception(
            "OCR pipeline failed",
            extra={
                "event": "ocr_pipeline_failed",
                "request_id": str(request_id),
                "pipeline_stage": stage,
                "error_code": error_code,
            },
        )


def _page_count(result) -> int:
    pages = getattr(result, "pages", ())
    if pages:
        return len(pages)
    statistics = getattr(result, "statistics", None)
    return int(getattr(statistics, "page_count", 1))
