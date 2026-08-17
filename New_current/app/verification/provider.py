"""Local-only PubMedBERT MedNLI provider with explicit checkpoint label semantics."""
# ruff: noqa: E501

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from app.verification.contracts import (
    BaseVerificationProvider,
    VerificationInference,
    VerificationProviderMetadata,
)
from app.verification.errors import VerificationUnavailableError
from app.verification.manifest import LABELS, VerificationManifest, load_manifest


class PubMedBERTMedNLIProvider(BaseVerificationProvider):
    def __init__(self, *, manifest_path: Path | None = None) -> None:
        self._manifest: VerificationManifest = load_manifest(manifest_path)
        root = Path(__file__).resolve().parents[3]
        configured = os.getenv(
            "VERIFICATION_CONFIG__MODEL_PATH", self._manifest.local_cache_path
        )
        candidate = Path(configured)
        self._path = candidate if candidate.is_absolute() else root / candidate
        self._device = os.getenv("VERIFICATION_CONFIG__DEVICE", "auto")
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._torch: Any | None = None
        self._loading_ms: float | None = None
        self._startup_timestamp: datetime | None = None
        self._request_count = 0
        self._detail = "Provider has not been initialized."

    async def initialize(self, *, strict: bool = True) -> None:
        try:
            self._validate()
            await asyncio.to_thread(self._load)
            self._detail = "Local PubMedBERT MedNLI checkpoint is technically ready."
        except (ImportError, OSError, RuntimeError, ValueError, VerificationUnavailableError) as exc:
            self._detail = str(exc)
            if strict:
                raise VerificationUnavailableError(self._detail) from exc

    def _validate(self) -> None:
        manifest_path = Path(self._manifest.local_cache_path)
        root = Path(__file__).resolve().parents[3]
        expected = manifest_path if manifest_path.is_absolute() else root / manifest_path
        if self._path.resolve() != expected.resolve() or not self._path.is_dir():
            raise VerificationUnavailableError("Verification checkpoint must use the approved local cache.")
        provenance = (
            self._path
            / ".cache"
            / "huggingface"
            / "trees"
            / f"{self._manifest.pinned_revision}.json"
        )
        if not provenance.is_file():
            raise VerificationUnavailableError("Pinned verification revision provenance is missing.")
        config = json.loads((self._path / "config.json").read_text(encoding="utf-8"))
        if config.get("architectures") != ["BertForSequenceClassification"]:
            raise VerificationUnavailableError("Verification checkpoint architecture is invalid.")
        mapping = {int(key): value for key, value in config.get("id2label", {}).items()}
        if mapping != LABELS or config.get("max_position_embeddings") != 512:
            raise VerificationUnavailableError("Verification checkpoint label mapping is invalid.")

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        resolved = "cuda" if self._device == "auto" and torch.cuda.is_available() else self._device
        if resolved.startswith("cuda") and not torch.cuda.is_available():
            raise VerificationUnavailableError("CUDA was requested but is unavailable.")
        started = perf_counter()
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self._path), local_files_only=True, use_fast=True
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(self._path), local_files_only=True, use_safetensors=True
        ).to(resolved)
        self._model.eval()
        self._torch = torch
        self._device = resolved
        self._loading_ms = round((perf_counter() - started) * 1000, 3)
        self._startup_timestamp = datetime.now(UTC)

    def infer(self, premise: str, hypothesis: str) -> VerificationInference:
        if self._model is None or self._tokenizer is None or self._torch is None:
            raise VerificationUnavailableError(self._detail)
        self._request_count += 1
        started = perf_counter()
        encoded = self._tokenizer(
            premise, hypothesis, truncation=True, max_length=512, return_tensors="pt"
        ).to(self._device)
        with self._torch.inference_mode():
            logits = self._model(**encoded).logits[0]
            values = self._torch.softmax(logits, dim=-1).tolist()
        probabilities = {
            LABELS[index]: round(float(value), 8) for index, value in enumerate(values)
        }
        label = LABELS[int(logits.argmax().item())]
        return VerificationInference(label, probabilities, round((perf_counter() - started) * 1000, 3))

    def metadata(self) -> VerificationProviderMetadata:
        return VerificationProviderMetadata(
            self._manifest.repository_id,
            self._manifest.pinned_revision,
            self._device,
            self._model is not None,
            self._detail,
            {
                "architecture": "BertForSequenceClassification",
                "label_mapping": LABELS,
                "max_sequence_length": 512,
                "local_files_only": True,
                "model_loading_time_ms": self._loading_ms,
                "license_status": self._manifest.license,
                "production_approval_status": "PENDING_LICENSE_VERIFICATION",
                "warm": self._model is not None,
                "request_count": self._request_count,
                "startup_timestamp": (
                    self._startup_timestamp.isoformat() if self._startup_timestamp else None
                ),
            },
        )

    async def shutdown(self) -> None:
        self._model = None
        self._tokenizer = None
        if self._torch is not None and self._device.startswith("cuda"):
            self._torch.cuda.empty_cache()
