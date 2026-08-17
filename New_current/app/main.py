"""FastAPI entry point for OCR and production medical NER."""

import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.embeddings.errors import EmbeddingError
from app.embeddings.routes import router as embeddings_router
from app.embeddings.runtime import create_embedding_service
from app.embeddings.service import MedicalEmbeddingService
from app.entity_linking.errors import EntityLinkingError
from app.entity_linking.routes import router as entity_linking_router
from app.entity_linking.runtime import create_entity_linking_service
from app.entity_linking.service import EntityLinkingService
from app.infrastructure.errors import InfrastructureError
from app.infrastructure.routes import router as infrastructure_router
from app.infrastructure.runtime import InfrastructureRuntime
from app.ner.errors import NERError
from app.ner.routes import router as ner_router
from app.ner.runtime import create_ner_service
from app.ner.service import MedicalNERService
from app.ocr.api.routes import router as ocr_router
from app.ocr.api.routes import service_health_router
from app.ocr.application.errors import OCRApplicationError
from app.ocr.application.result_builder import OCRResultBuilder
from app.ocr.application.service import OCRApplicationService
from app.ocr.infrastructure.memory import MemoryOCRResultCache, MemoryOCRUnitOfWork
from app.ocr.observability.logging import configure_structured_logging
from app.ocr.providers.config import ProviderSettings
from app.ocr.providers.errors import (
    InferenceError,
    ProviderConfigurationError,
    ProviderError,
    ProviderInitializationError,
    ProviderUnavailableError,
    UnsupportedDocumentError,
)
from app.ocr.providers.runtime import ProviderContainer, create_provider_container
from app.relation_extraction.errors import RelationExtractionError
from app.relation_extraction.routes import router as relation_extraction_router
from app.relation_extraction.runtime import create_relation_extraction_service
from app.relation_extraction.service import RelationExtractionService
from app.simplification.errors import SimplificationError
from app.simplification.routes import router as simplification_router
from app.simplification.runtime import create_simplification_service
from app.simplification.service import SimplificationService
from app.translation.errors import TranslationError
from app.translation.routes import router as translation_router
from app.translation.runtime import create_translation_service
from app.translation.service import TranslationService
from app.verification.errors import VerificationError
from app.verification.routes import router as verification_router
from app.verification.runtime import create_verification_service
from app.verification.service import MedicalVerificationService

logger = logging.getLogger(__name__)
_APPLICATION_DIRECTORY = Path(__file__).parent
_STATIC_DIRECTORY = _APPLICATION_DIRECTORY / "static"
_TEMPLATE_DIRECTORY = _APPLICATION_DIRECTORY / "templates"
_SAFE_REQUEST_ID = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)


def create_app(
    *,
    provider_container: ProviderContainer | None = None,
    ner_service: MedicalNERService | None = None,
    entity_linking_service: EntityLinkingService | None = None,
    relation_extraction_service: RelationExtractionService | None = None,
    embedding_service: MedicalEmbeddingService | None = None,
    simplification_service: SimplificationService | None = None,
    translation_service: TranslationService | None = None,
    verification_service: MedicalVerificationService | None = None,
    infrastructure_runtime: InfrastructureRuntime | None = None,
) -> FastAPI:
    """Create the application with optional test-owned stage dependencies."""

    log_store = configure_structured_logging(settings.log_buffer_max_entries)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        container = provider_container
        if container is None:
            provider_settings = ProviderSettings.from_environment()
            container = create_provider_container(provider_settings)
        await container.initialize()
        unit_of_work = MemoryOCRUnitOfWork(settings.request_store_max_entries)
        cache = MemoryOCRResultCache(settings.cache_max_entries)
        application.state.ocr_provider_container = container
        application.state.ocr_application_service = OCRApplicationService(
            ocr_provider=container.ocr_provider,
            postprocessor=container.postprocessor,
            unit_of_work=unit_of_work,
            cache=cache,
            result_builder=OCRResultBuilder(
                pipeline_version=settings.pipeline_version,
                schema_version=settings.schema_version,
            ),
        )
        application.state.ocr_log_store = log_store
        production_infrastructure = infrastructure_runtime or InfrastructureRuntime()
        await production_infrastructure.initialize()
        application.state.infrastructure_runtime = production_infrastructure
        production_ner_service = ner_service or create_ner_service()
        await production_ner_service.initialize()
        application.state.ner_service = production_ner_service
        production_linking_service = entity_linking_service or create_entity_linking_service()
        await production_linking_service.initialize(strict=False)
        application.state.entity_linking_service = production_linking_service
        production_relation_service = (
            relation_extraction_service or create_relation_extraction_service()
        )
        await production_relation_service.initialize(strict=False)
        application.state.relation_extraction_service = production_relation_service
        production_embedding_service = embedding_service or create_embedding_service()
        await production_embedding_service.initialize(strict=False)
        application.state.embedding_service = production_embedding_service
        production_simplification_service = (
            simplification_service or create_simplification_service()
        )
        await production_simplification_service.initialize(strict=False)
        application.state.simplification_service = production_simplification_service
        production_translation_service = translation_service or create_translation_service()
        await production_translation_service.initialize(strict=False)
        application.state.translation_service = production_translation_service
        production_verification_service = verification_service or create_verification_service()
        await production_verification_service.initialize(strict=False)
        application.state.verification_service = production_verification_service
        logger.info(
            "OCR service ready",
            extra={
                "event": "ocr_service_ready",
                "pipeline_stage": "startup",
                "pipeline_version": settings.pipeline_version,
            },
        )
        try:
            yield
        finally:
            await production_infrastructure.shutdown()
            await production_verification_service.shutdown()
            await production_translation_service.shutdown()
            await production_simplification_service.shutdown()
            await production_embedding_service.shutdown()
            await production_relation_service.shutdown()
            await production_linking_service.shutdown()
            await production_ner_service.shutdown()
            await container.shutdown()

    application = FastAPI(
        title="IB Health Clinical Document Processing Service",
        description=(
            "Phase 2 service for Qwen3-VL OCR with prompted document-type inference and "
            "medical post-processing plus production medical entity recognition using the "
            "approved immutable biomedical-ner-all checkpoint, and local SciSpaCy UMLS "
            "entity linking, plus a BioLinkBERT relation-extraction boundary. The service "
            "also exposes a BioClinical ModernBERT medical-embedding boundary and the "
            "local Qwen3 medical report simplification service. It does not "
            "provide diagnosis or treatment advice."
        ),
        version="2.0.0",
        lifespan=lifespan,
    )
    application.include_router(ocr_router)
    application.include_router(service_health_router)
    application.include_router(ner_router)
    application.include_router(entity_linking_router)
    application.include_router(relation_extraction_router)
    application.include_router(embeddings_router)
    application.include_router(simplification_router)
    application.include_router(translation_router)
    application.include_router(verification_router)
    application.include_router(infrastructure_router)
    application.mount("/static", StaticFiles(directory=_STATIC_DIRECTORY), name="static")
    templates = Jinja2Templates(directory=_TEMPLATE_DIRECTORY)

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        supplied_request_id = request.headers.get("X-Request-ID", "")
        request_id = (
            UUID(supplied_request_id)
            if _SAFE_REQUEST_ID.fullmatch(supplied_request_id)
            else uuid4()
        )
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = str(request_id)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    @application.get(
        "/",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    async def engineering_console(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="ocr_dashboard.html",
            context={
                "pipeline_version": settings.pipeline_version,
                "owner_id": str(settings.default_owner_id),
            },
        )

    @application.get(
        "/ner",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    async def ner_console(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="ner_dashboard.html",
            context={"schema_version": "ner-response-v1"},
        )

    @application.get(
        "/entity-linking",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    async def entity_linking_console(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="entity_linking_dashboard.html",
            context={"schema_version": "entity-linking-response-v1"},
        )

    @application.get(
        "/embeddings",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    async def embeddings_console(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="embeddings_dashboard.html",
            context={"schema_version": "medical-embedding-response-v1"},
        )

    @application.get(
        "/simplify",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    async def simplification_console(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="simplification_dashboard.html",
            context={"schema_version": "simplification-response-v2"},
        )

    @application.get("/verification", response_class=HTMLResponse, include_in_schema=False)
    async def verification_console(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="verification_dashboard.html",
            context={"schema_version": "medical-verification-v1"},
        )

    @application.get(
        "/engineering-demo",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    async def engineering_demo(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="engineering_demo.html",
            context={"dashboard_version": "engineering-demo-v1"},
        )

    @application.get(
        "/infrastructure",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    async def infrastructure_console(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="infrastructure_dashboard.html",
            context={"dashboard_version": "infrastructure-dashboard-v1"},
        )

    @application.exception_handler(OCRApplicationError)
    async def application_error_handler(request: Request, exc: OCRApplicationError) -> JSONResponse:
        logger.warning(
            "OCR application request rejected",
            extra={
                "event": "ocr_application_error",
                "request_id": str(request.state.request_id),
                "pipeline_stage": "api",
                "error_code": exc.code,
            },
        )
        return _error_response(request, exc.status_code, exc.code, exc.message)

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        del exc
        return _error_response(
            request,
            422,
            "input_validation_error",
            "The request payload is invalid.",
        )

    @application.exception_handler(ProviderError)
    async def provider_error_handler(request: Request, exc: ProviderError) -> JSONResponse:
        if isinstance(exc, UnsupportedDocumentError):
            status_code = 415
        elif isinstance(exc, InferenceError):
            status_code = 500
        elif isinstance(
            exc,
            (
                ProviderConfigurationError,
                ProviderInitializationError,
                ProviderUnavailableError,
            ),
        ):
            status_code = 503
        else:
            status_code = 500
        logger.warning(
            "OCR provider request failed",
            extra={
                "event": "ocr_provider_error",
                "request_id": str(request.state.request_id),
                "pipeline_stage": "provider",
                "error_code": exc.code,
            },
        )
        return _error_response(request, status_code, exc.code, exc.message)

    @application.exception_handler(NERError)
    async def ner_error_handler(request: Request, exc: NERError) -> JSONResponse:
        logger.warning(
            "Medical NER request rejected",
            extra={
                "event": "ner_error",
                "request_id": str(request.state.request_id),
                "pipeline_stage": "medical_ner",
                "error_code": exc.code,
            },
        )
        return _error_response(request, exc.status_code, exc.code, exc.message)

    @application.exception_handler(EntityLinkingError)
    async def entity_linking_error_handler(
        request: Request, exc: EntityLinkingError
    ) -> JSONResponse:
        logger.warning(
            "Entity-linking request rejected",
            extra={
                "event": "entity_linking_error",
                "request_id": str(request.state.request_id),
                "pipeline_stage": "entity_linking",
                "error_code": exc.code,
            },
        )
        return _error_response(request, exc.status_code, exc.code, exc.message)

    @application.exception_handler(RelationExtractionError)
    async def relation_extraction_error_handler(
        request: Request, exc: RelationExtractionError
    ) -> JSONResponse:
        logger.warning(
            "Relation-extraction request rejected",
            extra={
                "event": "relation_extraction_error",
                "request_id": str(request.state.request_id),
                "pipeline_stage": "relation_extraction",
                "error_code": exc.code,
            },
        )
        return _error_response(request, exc.status_code, exc.code, exc.message)

    @application.exception_handler(EmbeddingError)
    async def embedding_error_handler(request: Request, exc: EmbeddingError) -> JSONResponse:
        logger.warning(
            "Medical embedding request rejected",
            extra={
                "event": "embedding_error",
                "request_id": str(request.state.request_id),
                "pipeline_stage": "medical_embeddings",
                "error_code": exc.code,
            },
        )
        return _error_response(request, exc.status_code, exc.code, exc.message)

    @application.exception_handler(SimplificationError)
    async def simplification_error_handler(
        request: Request, exc: SimplificationError
    ) -> JSONResponse:
        logger.warning(
            "Simplification request rejected",
            extra={
                "event": "simplification_error",
                "request_id": str(request.state.request_id),
                "pipeline_stage": "simplification",
                "error_code": exc.code,
            },
        )
        return _error_response(request, exc.status_code, exc.code, exc.message)

    @application.exception_handler(TranslationError)
    async def translation_error_handler(request: Request, exc: TranslationError) -> JSONResponse:
        logger.warning(
            "Translation request rejected",
            extra={
                "event": "translation_error",
                "request_id": str(request.state.request_id),
                "pipeline_stage": "translation",
                "error_code": exc.code,
            },
        )
        return _error_response(request, exc.status_code, exc.code, exc.message)

    @application.exception_handler(VerificationError)
    async def verification_error_handler(request: Request, exc: VerificationError) -> JSONResponse:
        return _error_response(request, exc.status_code, exc.code, str(exc))

    @application.exception_handler(InfrastructureError)
    async def infrastructure_error_handler(
        request: Request, exc: InfrastructureError
    ) -> JSONResponse:
        logger.warning(
            "Infrastructure request rejected",
            extra={
                "event": "infrastructure_error",
                "request_id": str(request.state.request_id),
                "pipeline_stage": "infrastructure",
                "error_code": exc.code,
            },
        )
        return _error_response(request, exc.status_code, exc.code, exc.message)

    @application.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unexpected OCR processing failure",
            extra={
                "event": "ocr_unexpected_error",
                "request_id": str(request.state.request_id),
                "pipeline_stage": "api",
                "error_code": "internal_error",
            },
        )
        del exc
        return _error_response(
            request,
            500,
            "internal_error",
            "The OCR request could not be processed due to an internal error.",
        )

    return application


def _error_response(request: Request, status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {"code": code, "message": message},
            "request_id": str(request.state.request_id),
        },
    )


app = create_app()
