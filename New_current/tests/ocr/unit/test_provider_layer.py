"""Unit coverage for provider contracts, registration, factories, and lifecycle."""

import asyncio
import json
import logging
import os
from pathlib import Path

import pytest

from app.ocr.api.dependencies import (
    get_ocr_provider,
    get_postprocessor,
)
from app.ocr.providers.config import ProviderSettings
from app.ocr.providers.contracts import (
    BaseOCRProvider,
    BasePostProcessor,
    ProviderHealthStatus,
    ProviderKind,
)
from app.ocr.providers.errors import ProviderConfigurationError
from app.ocr.providers.implementations import (
    Qwen3VLOCRProvider,
    SymSpellPostProcessor,
    register_builtin_providers,
)
from app.ocr.providers.manifest import load_model_manifest
from app.ocr.providers.registry import ProviderRegistry
from app.ocr.providers.runtime import (
    OCRProviderFactory,
    PostProcessorFactory,
    create_provider_container,
)


def _provider_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "OCR_PROVIDER",
            "POSTPROCESSOR_PROVIDER",
        }
        or key.startswith(
            (
                "OCR_CONFIG__",
                "POSTPROCESSOR_CONFIG__",
            )
        )
    }
    environment["OCR_CONFIG__ACCESS_TOKEN"] = "must-not-be-exposed"
    return environment


def _registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    register_builtin_providers(registry)
    return registry


def test_builtin_provider_registration_and_duplicate_protection() -> None:
    registry = _registry()

    assert registry.registered_names(ProviderKind.OCR) == ("qwen3-vl",)
    assert registry.registered_names(ProviderKind.POSTPROCESSOR) == ("symspell",)

    with pytest.raises(ProviderConfigurationError, match="already registered"):
        registry.register(
            ProviderKind.OCR,
            "qwen3-vl",
            Qwen3VLOCRProvider,
        )


def test_registration_emits_structured_log_fields(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    registry = ProviderRegistry()

    registry.register(ProviderKind.OCR, "qwen3-vl", Qwen3VLOCRProvider)

    record = next(record for record in caplog.records if record.msg == "OCR provider registered")
    assert record.event == "provider_registered"
    assert record.provider_kind == "ocr"
    assert record.provider_name == "qwen3-vl"


def test_factories_select_registered_provider_contracts() -> None:
    settings = ProviderSettings.from_environment(_provider_environment())
    registry = _registry()

    ocr_provider = OCRProviderFactory(registry).create(
        settings.selected(ProviderKind.OCR),
        settings.configuration_for(ProviderKind.OCR),
    )
    postprocessor = PostProcessorFactory(registry).create(
        settings.selected(ProviderKind.POSTPROCESSOR),
        settings.configuration_for(ProviderKind.POSTPROCESSOR),
    )

    assert isinstance(ocr_provider, BaseOCRProvider)
    assert isinstance(postprocessor, BasePostProcessor)
    assert isinstance(ocr_provider, Qwen3VLOCRProvider)
    assert isinstance(postprocessor, SymSpellPostProcessor)


def test_factory_rejects_unsupported_provider() -> None:
    registry = _registry()

    with pytest.raises(ProviderConfigurationError, match="Unsupported ocr provider"):
        OCRProviderFactory(registry).create("not-installed", {})


def test_environment_configuration_is_required() -> None:
    with pytest.raises(
        ProviderConfigurationError,
        match="OCR_PROVIDER",
    ):
        ProviderSettings.from_environment({})


def test_repository_model_manifest_contains_approved_qwen_revision() -> None:
    manifest = load_model_manifest()

    assert manifest.schema_version == "1"
    assert set(manifest.models) == {"qwen3-vl"}
    ocr = manifest.models["qwen3-vl"]
    assert not ocr.approval_pending
    assert ocr.repository_id == "Qwen/Qwen3-VL-4B-Instruct"
    assert ocr.pinned_revision == "ebb281ec70b05090aa6165b016eac8ec08e71b17"


def test_manifest_supplies_approved_model_identity_when_environment_omits_it(
    tmp_path: Path,
) -> None:
    payload = _approved_manifest_payload()
    manifest_path = tmp_path / "MODEL_MANIFEST.md"
    manifest_path.write_text(_manifest_document(payload), encoding="utf-8")
    environment = _provider_environment()
    for key in (
        "OCR_CONFIG__MODEL_NAME",
        "OCR_CONFIG__MODEL_REVISION",
        "OCR_CONFIG__HF_CACHE_DIR",
        "OCR_CONFIG__DEVICE",
    ):
        environment.pop(key, None)

    settings = ProviderSettings.from_environment(
        environment,
        manifest_path=manifest_path,
    )

    ocr = settings.configuration_for(ProviderKind.OCR)
    assert ocr["model_name"] == "test-only-qwen-checkpoint"
    assert ocr["model_revision"] == "c" * 40


def test_manifest_rejects_moving_revision(tmp_path: Path) -> None:
    payload = _approved_manifest_payload()
    payload["models"]["qwen3-vl"]["pinned_revision"] = "latest"
    manifest_path = tmp_path / "MODEL_MANIFEST.md"
    manifest_path.write_text(_manifest_document(payload), encoding="utf-8")

    with pytest.raises(ProviderConfigurationError, match="moving revision"):
        load_model_manifest(manifest_path)


def _approved_manifest_payload() -> dict:
    return {
        "schema_version": "1",
        "models": {
            "qwen3-vl": {
                "model_name": "Qwen3-VL",
                "purpose": "Test-only OCR",
                "provider": "qwen3-vl",
                "repository_id": "test-only-qwen-checkpoint",
                "pinned_revision": "c" * 40,
                "license": "test-only",
                "local_cache_path": "cache/qwen3-vl",
                "expected_sha256": "d" * 64,
                "device": "cpu",
                "configuration_variables": [
                    "OCR_CONFIG__MODEL_NAME",
                    "OCR_CONFIG__MODEL_REVISION",
                    "OCR_CONFIG__HF_CACHE_DIR",
                    "OCR_CONFIG__DEVICE",
                ],
            },
        },
    }


def _manifest_document(payload: dict) -> str:
    return (
        "<!-- MODEL_MANIFEST_DATA_BEGIN -->\n```json\n"
        + json.dumps(payload)
        + "\n```\n<!-- MODEL_MANIFEST_DATA_END -->\n"
    )


def test_configuration_validation_fails_closed() -> None:
    environment = _provider_environment()
    environment["POSTPROCESSOR_CONFIG__MAX_EDIT_DISTANCE"] = "9"
    settings = ProviderSettings.from_environment(environment)
    provider = PostProcessorFactory(_registry()).create(
        settings.selected(ProviderKind.POSTPROCESSOR),
        settings.configuration_for(ProviderKind.POSTPROCESSOR),
    )

    with pytest.raises(ProviderConfigurationError, match="at most 3"):
        asyncio.run(provider.initialize())

    assert provider.health().status is ProviderHealthStatus.UNAVAILABLE


def test_provider_lifecycle_metadata_and_dependency_resolution() -> None:
    settings = ProviderSettings.from_environment(_provider_environment())
    container = create_provider_container(settings, discover=False)

    asyncio.run(container.initialize())
    try:
        assert get_ocr_provider(container) is container.ocr_provider
        assert get_postprocessor(container) is container.postprocessor

        for provider in container.providers:
            metadata = provider.metadata()
            health = provider.health()
            assert metadata.provider_name
            assert metadata.provider_version == "contract-1.0.0"
            assert metadata.supported_file_types
            assert metadata.supported_document_types
            assert metadata.startup_timestamp is not None
            assert health.status is ProviderHealthStatus.READY
            assert health.startup_timestamp == metadata.startup_timestamp

        ocr_configuration = container.ocr_provider.metadata().configuration
        assert ocr_configuration["access_token"] == "[REDACTED]"
    finally:
        asyncio.run(container.shutdown())

    assert all(
        provider.health().status is ProviderHealthStatus.STOPPED for provider in container.providers
    )
