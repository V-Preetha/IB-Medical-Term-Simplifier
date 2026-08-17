"""Focused provider, service, API, and fail-closed IndicTrans2 tests."""

import asyncio
import json
from contextlib import nullcontext
from pathlib import Path
from uuid import uuid4

import pytest

from app.translation.errors import TranslationPreservationError, TranslationUnavailableError
from app.translation.provider import IndicTrans2Provider
from app.translation.service import TranslationService


def _manifest(
    path: Path,
    *,
    revision: str = "a" * 40,
    cache_path: str = "PENDING_APPROVAL",
    license_name: str = "MIT",
    expected_sha256: str = "PENDING_APPROVAL",
) -> Path:
    payload = {
        "schema_version": "1",
        "production_model": {
            "model_name": "IndicTrans2 English-to-Indic Distilled 200M",
            "provider": "indictrans2",
            "repository_id": "ai4bharat/indictrans2-en-indic-dist-200M",
            "pinned_revision": revision,
            "license": license_name,
            "local_cache_path": cache_path,
            "expected_sha256": expected_sha256,
            "configuration_variables": [
                "TRANSLATION_CONFIG__MODEL_ID",
                "TRANSLATION_CONFIG__MODEL_REVISION",
                "TRANSLATION_CONFIG__MODEL_PATH",
                "TRANSLATION_CONFIG__DEVICE",
                "TRANSLATION_CONFIG__MODULE_CACHE_PATH",
            ],
        },
    }
    path.write_text(
        "<!-- TRANSLATION_MODEL_MANIFEST_DATA_BEGIN -->\n```json\n"
        + json.dumps(payload)
        + "\n```\n<!-- TRANSLATION_MODEL_MANIFEST_DATA_END -->\n",
        encoding="utf-8",
    )
    return path


class _Inputs(dict):
    def to(self, device: str):
        self["transferred_to"] = device
        return self


class _Processor:
    def __init__(self, *, preserve_markers: bool = True) -> None:
        self.preserve_markers = preserve_markers
        self.calls: list[tuple[list[str], str, str]] = []

    def preprocess_batch(self, texts, *, src_lang, tgt_lang):
        self.calls.append((list(texts), src_lang, tgt_lang))
        return list(texts)

    def postprocess_batch(self, texts, *, lang):
        if self.preserve_markers:
            return [f"translated {text}" for text in texts]
        return ["translated content" for _ in texts]


class _Tokenizer:
    def __call__(self, batch, **kwargs):
        del kwargs
        return _Inputs(batch=list(batch))

    @staticmethod
    def batch_decode(generated, **kwargs):
        del kwargs
        return list(generated)


class _Model:
    class _Parameter:
        dtype = "float32"

    @staticmethod
    def generate(*, batch, **kwargs):
        del kwargs
        return batch

    @staticmethod
    def parameters():
        return iter((_Model._Parameter(),))


class _Torch:
    @staticmethod
    def inference_mode():
        return nullcontext()

    @staticmethod
    def device(value):
        return value

    class cuda:
        @staticmethod
        def is_available():
            return True


def _ready_provider(tmp_path: Path, monkeypatch, *, preserve_markers: bool = True):
    revision = "a" * 40
    snapshot = tmp_path / revision
    snapshot.mkdir(exist_ok=True)
    manifest = _manifest(tmp_path / "MODEL_MANIFEST.md", cache_path=str(snapshot))
    monkeypatch.setenv("TRANSLATION_CONFIG__MODEL_PATH", str(snapshot))
    monkeypatch.setenv("TRANSLATION_CONFIG__MODEL_REVISION", revision)
    monkeypatch.setenv(
        "TRANSLATION_CONFIG__MODEL_ID", "ai4bharat/indictrans2-en-indic-dist-200M"
    )
    monkeypatch.setenv("TRANSLATION_CONFIG__DEVICE", "cpu")
    provider = IndicTrans2Provider(manifest_path=manifest)
    provider._model = _Model()
    provider._tokenizer = _Tokenizer()
    provider._processor = _Processor(preserve_markers=preserve_markers)
    import torch

    provider._torch = torch
    provider._device = "cpu"
    return provider


def test_unprovisioned_manifest_remains_not_ready(tmp_path, monkeypatch) -> None:
    for name in (
        "TRANSLATION_CONFIG__MODEL_ID",
        "TRANSLATION_CONFIG__MODEL_REVISION",
        "TRANSLATION_CONFIG__MODEL_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    manifest = _manifest(
        tmp_path / "MODEL_MANIFEST.md",
        revision="PENDING_APPROVAL",
        cache_path="PENDING_APPROVAL",
        license_name="PENDING_APPROVAL",
    )
    provider = IndicTrans2Provider(manifest_path=manifest)

    asyncio.run(provider.initialize(strict=False))

    assert provider.metadata().ready is False
    assert "pinned_revision" in provider.metadata().detail
    with pytest.raises(TranslationUnavailableError):
        asyncio.run(provider.initialize(strict=True))


def test_initialization_validates_approved_local_configuration(tmp_path, monkeypatch) -> None:
    revision = "a" * 40
    snapshot = tmp_path / revision
    snapshot.mkdir()
    manifest = _manifest(tmp_path / "MODEL_MANIFEST.md", cache_path=str(snapshot))
    monkeypatch.setenv("TRANSLATION_CONFIG__MODEL_PATH", str(snapshot))
    monkeypatch.setenv("TRANSLATION_CONFIG__MODEL_REVISION", revision)
    monkeypatch.setenv(
        "TRANSLATION_CONFIG__MODEL_ID", "ai4bharat/indictrans2-en-indic-dist-200M"
    )
    provider = IndicTrans2Provider(manifest_path=manifest)

    def fake_load():
        provider._model = object()
        provider._device = "cpu"
        provider._model_loading_time_ms = 1.5

    monkeypatch.setattr(provider, "_load", fake_load)
    asyncio.run(provider.initialize())

    assert provider.metadata().ready is True
    assert provider.metadata().configuration["manifest_approved"] is True


def test_initialization_verifies_approved_safetensors_checksum(tmp_path, monkeypatch) -> None:
    revision = "a" * 40
    snapshot = tmp_path / revision
    snapshot.mkdir()
    weight = snapshot / "model.safetensors"
    weight.write_bytes(b"approved-synthetic-weight")
    manifest = _manifest(
        tmp_path / "MODEL_MANIFEST.md",
        cache_path=str(snapshot),
        expected_sha256=IndicTrans2Provider._file_sha256(weight),
    )
    monkeypatch.setenv("TRANSLATION_CONFIG__MODEL_PATH", str(snapshot))
    monkeypatch.setenv("TRANSLATION_CONFIG__MODEL_REVISION", revision)
    provider = IndicTrans2Provider(manifest_path=manifest)
    monkeypatch.setattr(provider, "_load", lambda: setattr(provider, "_model", object()))

    asyncio.run(provider.initialize())

    assert provider.metadata().configuration["checksum_verified"] is True
    weight.write_bytes(b"tampered")
    with pytest.raises(TranslationUnavailableError, match="checksum"):
        asyncio.run(IndicTrans2Provider(manifest_path=manifest).initialize())


def test_device_selection_prefers_cuda_for_auto() -> None:
    assert IndicTrans2Provider._resolve_device(_Torch, "auto") == "cuda"
    assert IndicTrans2Provider._resolve_device(_Torch, "cpu") == "cpu"


def test_generation_limit_configuration_fails_closed(tmp_path, monkeypatch) -> None:
    revision = "a" * 40
    snapshot = tmp_path / revision
    snapshot.mkdir()
    manifest = _manifest(tmp_path / "MODEL_MANIFEST.md", cache_path=str(snapshot))
    monkeypatch.setenv("TRANSLATION_CONFIG__MODEL_PATH", str(snapshot))
    monkeypatch.setenv("TRANSLATION_CONFIG__MODEL_REVISION", revision)
    monkeypatch.setenv("TRANSLATION_CONFIG__MAX_NEW_TOKENS", "257")
    provider = IndicTrans2Provider(manifest_path=manifest)

    with pytest.raises(TranslationUnavailableError, match="MAX_NEW_TOKENS"):
        asyncio.run(provider.initialize())


def test_single_and_batch_translation_preserve_numeric_values(tmp_path, monkeypatch) -> None:
    provider = _ready_provider(tmp_path, monkeypatch)

    one = provider.translate("Glucose was 126 mg/dL.", "eng_Latn", "hin_Deva")
    batch = provider.translate_batch(
        ("HbA1c was 7.2%.", "Metformin dose was 500 mg on 15/10/2024."),
        "eng_Latn",
        "tam_Taml",
    )

    assert "126 mg/dL" in one
    assert len(batch) == 2
    assert "7.2%" in batch[0]
    assert "500 mg" in batch[1]
    assert "15/10/2024" in batch[1]
    assert provider._processor.calls[-1][1:] == ("eng_Latn", "tam_Taml")


def test_language_selection_and_preservation_fail_closed(tmp_path, monkeypatch) -> None:
    provider = _ready_provider(tmp_path, monkeypatch)

    with pytest.raises(TranslationUnavailableError):
        provider.translate("Text", "eng_Latn", "fra_Latn")

    unsafe = _ready_provider(tmp_path, monkeypatch, preserve_markers=False)
    with pytest.raises(TranslationPreservationError):
        unsafe.translate("Glucose was 126 mg/dL.", "eng_Latn", "hin_Deva")


def test_service_uses_provider_batch_path_once(tmp_path, monkeypatch) -> None:
    provider = _ready_provider(tmp_path, monkeypatch)
    service = TranslationService(provider)

    result = asyncio.run(
        service.process_batch(
            ("Glucose was 126 mg/dL.", "HbA1c was 7.2%."),
            "eng_Latn",
            "hin_Deva",
            uuid4(),
        )
    )

    assert len(result.translated_texts) == 2
    assert result.processing_time_ms >= 0
