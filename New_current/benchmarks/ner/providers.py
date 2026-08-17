"""Archived candidate adapters used only by the offline benchmark runner."""

from typing import Any

from app.ner.contracts import ENTITY_TYPES, NERProviderResult
from app.ner.errors import NERInferenceError
from app.ner.manifest import NERModelManifest
from app.ner.providers import (
    LocalTokenClassificationProvider,
    NERProviderRegistry,
    _deduplicate,
    _ManagedNERProvider,
    _normalized_entity,
)


class GLiNEREvaluationProvider(_ManagedNERProvider):
    """Local-only zero-shot GLiNER adapter for archived benchmark execution."""

    def __init__(self, entry, manifest_path) -> None:
        super().__init__(
            entry,
            manifest_path,
            device_variable="NER_BENCHMARK_DEVICE",
            threshold_variable="NER_BENCHMARK_CONFIDENCE_THRESHOLD",
            max_tokens_variable="NER_BENCHMARK_MAX_TOKENS",
            stride_variable="NER_BENCHMARK_STRIDE_TOKENS",
        )

    def _load_runtime(self) -> None:
        from gliner import GLiNER

        self._model = GLiNER.from_pretrained(str(self.cache_path))
        if self._device == "cuda":
            self._model.to("cuda")

    def extract(self, text: str) -> NERProviderResult:
        self._require_ready()
        if not text.strip():
            raise NERInferenceError("NER benchmark text must not be empty.")
        try:
            with self._lock:
                predictions = self._model.predict_entities(
                    text,
                    list(ENTITY_TYPES),
                    threshold=self._threshold,
                )
            entities = tuple(
                _normalized_entity(
                    text,
                    item.get("label", ""),
                    int(item["start"]),
                    int(item["end"]),
                    float(item.get("score", 0.0)),
                )
                for item in predictions
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise NERInferenceError("GLiNER returned an invalid entity result.") from exc
        return NERProviderResult(_deduplicate(entities), _token_count(self._model, text))


class BenchmarkTokenClassificationProvider(LocalTokenClassificationProvider):
    """Archived token classifier configured only by the offline runner."""

    def __init__(self, entry, manifest_path) -> None:
        super().__init__(
            entry,
            manifest_path,
            device_variable="NER_BENCHMARK_DEVICE",
            threshold_variable="NER_BENCHMARK_CONFIDENCE_THRESHOLD",
            max_tokens_variable="NER_BENCHMARK_MAX_TOKENS",
            stride_variable="NER_BENCHMARK_STRIDE_TOKENS",
        )


def create_evaluation_registry(manifest: NERModelManifest) -> NERProviderRegistry:
    registry = NERProviderRegistry()
    registry.register(
        "openmed-gliner",
        lambda: GLiNEREvaluationProvider(manifest.candidates["openmed-gliner"], manifest.path),
    )
    for name in ("biomedical-ner-all", "modernbert-biomedical-ner"):
        registry.register(
            name,
            lambda candidate=name: BenchmarkTokenClassificationProvider(
                manifest.candidates[candidate], manifest.path
            ),
        )
    return registry


def _token_count(model: Any, text: str) -> int | None:
    tokenizer = getattr(getattr(model, "data_processor", None), "transformer_tokenizer", None)
    if tokenizer is None:
        return None
    return len(tokenizer.encode(text, add_special_tokens=True))
