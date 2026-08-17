"""GLiNER-BioMed provider with process-wide model reuse."""

import logging
import re
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any, ClassVar

from app.clinical.models import NerEntity
from app.performance import PipelineTimings

logger = logging.getLogger(__name__)

CLINICAL_ENTITY_LABELS = (
    "disease",
    "symptom",
    "medication",
    "anatomy",
    "medical procedure",
    "laboratory test",
    "biomarker",
    "measurement",
    "vital sign",
)


class GlinerClinicalNerProvider:
    """Run a real biomedical GLiNER checkpoint over configured clinical labels."""

    _instances: ClassVar[dict[tuple[str, str, float, int, int], "GlinerClinicalNerProvider"]] = {}
    _instances_lock: ClassVar[RLock] = RLock()

    def __init__(
        self,
        model: Any,
        *,
        model_id: str,
        device: str = "cpu",
        confidence_threshold: float = 0.5,
        chunk_characters: int = 2000,
        chunk_overlap: int = 0,
        batch_size: int = 4,
    ) -> None:
        self._model = model
        self.model_id = model_id
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.chunk_characters = chunk_characters
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self._inference_lock = RLock()

    @classmethod
    def load_once(
        cls,
        model_id_or_path: str,
        *,
        device: str = "cpu",
        confidence_threshold: float = 0.5,
        chunk_characters: int = 2000,
        batch_size: int = 4,
    ) -> "GlinerClinicalNerProvider":
        key = (
            model_id_or_path,
            device,
            confidence_threshold,
            chunk_characters,
            batch_size,
        )
        with cls._instances_lock:
            existing = cls._instances.get(key)
            if existing is not None:
                return existing

            started_at = perf_counter()
            try:
                from gliner import GLiNER
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise RuntimeError(
                    "Clinical NER requires the project's 'clinical-ner' dependency extra."
                ) from exc

            try:
                source = (
                    str(Path(model_id_or_path).resolve())
                    if Path(model_id_or_path).exists()
                    else snapshot_download(
                        model_id_or_path,
                        local_files_only=True,
                    )
                )
                model = GLiNER.from_pretrained(source, local_files_only=True)
                model = model.to(device)
                model.eval()
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    "GLiNER clinical model files are unavailable or invalid. "
                    f"Download '{model_id_or_path}' into the Hugging Face cache or "
                    "set REPORT_CLINICAL_NER_MODEL to a complete local snapshot."
                ) from exc

            provider = cls(
                model,
                model_id=model_id_or_path,
                device=device,
                confidence_threshold=confidence_threshold,
                chunk_characters=chunk_characters,
                batch_size=batch_size,
            )
            cls._instances[key] = provider
            logger.info(
                "GLiNER clinical NER model loaded",
                extra={
                    "clinical_ner_model": model_id_or_path,
                    "clinical_ner_device": device,
                    "clinical_ner_load_time_ms": round(
                        (perf_counter() - started_at) * 1000,
                        3,
                    ),
                },
            )
            return provider

    def __call__(self, text: str) -> tuple[NerEntity, ...]:
        return self.infer(text)

    def infer(
        self,
        text: str,
        timings: PipelineTimings | None = None,
    ) -> tuple[NerEntity, ...]:
        if not text.strip():
            return ()
        started_at = perf_counter()
        chunk_started_at = perf_counter()
        chunks = self._chunks(text)
        chunking_ms = (perf_counter() - chunk_started_at) * 1000
        if timings is not None:
            timings.record("Chunking", chunking_ms)

        with self._inference_lock:
            inference_started_at = perf_counter()
            prediction_groups = self._predict_chunks(chunks)
            inference_ms = (perf_counter() - inference_started_at) * 1000

        for index in range(len(chunks)):
            logger.info(
                "GLiNER Chunk %d .......... %.3f ms",
                index + 1,
                inference_ms,
                extra={
                    "request_id": timings.request_id if timings is not None else None,
                    "pipeline_stage": f"GLiNER Chunk {index + 1}",
                    "stage_time_ms": round(inference_ms, 3),
                    "gliner_chunk_characters": len(chunks[index][1]),
                },
            )
        if timings is not None:
            timings.record("GLiNER", inference_ms)

        merge_started_at = perf_counter()
        entities = self._merge_predictions(chunks, prediction_groups)
        merge_ms = (perf_counter() - merge_started_at) * 1000
        if timings is not None:
            timings.record("Entity Merge", merge_ms)

        elapsed_ms = round((perf_counter() - started_at) * 1000, 3)
        logger.info(
            "GLiNER clinical NER inference complete",
            extra={
                "clinical_ner_model": self.model_id,
                "clinical_ner_entities": [
                    {
                        "text": entity.text,
                        "label": entity.label,
                        "confidence": entity.confidence,
                        "start": entity.start,
                        "end": entity.end,
                    }
                    for entity in entities
                ],
                "clinical_ner_processing_time_ms": elapsed_ms,
                "clinical_ner_inference_time_ms": round(inference_ms, 3),
                "clinical_ner_chunk_count": len(chunks),
                "clinical_ner_average_chunk_characters": round(
                    sum(len(chunk) for _, chunk in chunks) / len(chunks),
                    1,
                ),
            },
        )
        return tuple(entities)

    def _chunks(self, text: str) -> tuple[tuple[int, str], ...]:
        if len(text) <= self.chunk_characters:
            return ((0, text),)

        units = self._semantic_units(text)
        chunks: list[tuple[int, str]] = []
        chunk_start = units[0][0]
        chunk_end = chunk_start
        for unit_start, unit_end in units:
            if chunk_end > chunk_start and unit_end - chunk_start > self.chunk_characters:
                chunks.append((chunk_start, text[chunk_start:chunk_end]))
                chunk_start = unit_start
            chunk_end = unit_end
        if chunk_end > chunk_start:
            chunks.append((chunk_start, text[chunk_start:chunk_end]))
        return tuple(chunks)

    def _semantic_units(self, text: str) -> list[tuple[int, int]]:
        units: list[tuple[int, int]] = []
        start = 0
        for match in re.finditer(r"\n\s*\n|(?<=[.!?])(?:[ \t]+|\n+)", text):
            end = match.end()
            self._append_bounded_units(text, start, end, units)
            start = end
        if start < len(text):
            self._append_bounded_units(text, start, len(text), units)
        return units

    def _append_bounded_units(
        self,
        text: str,
        start: int,
        end: int,
        output: list[tuple[int, int]],
    ) -> None:
        while end - start > self.chunk_characters:
            limit = start + self.chunk_characters
            split = text.rfind(" ", start, limit)
            if split <= start:
                split = limit
            else:
                split += 1
            output.append((start, split))
            start = split
        if end > start:
            output.append((start, end))

    def _predict_chunks(
        self,
        chunks: tuple[tuple[int, str], ...],
    ) -> list[list[Mapping[str, Any]]]:
        texts = [chunk for _, chunk in chunks]
        labels = list(CLINICAL_ENTITY_LABELS)
        if len(texts) > 1:
            batch_predict = getattr(self._model, "inference", None)
            if batch_predict is None:
                batch_predict = getattr(self._model, "batch_predict_entities", None)
            if callable(batch_predict):
                return batch_predict(
                    texts,
                    labels,
                    threshold=self.confidence_threshold,
                    flat_ner=False,
                    multi_label=True,
                    batch_size=min(self.batch_size, len(texts)),
                )
            with ThreadPoolExecutor(
                max_workers=min(self.batch_size, len(texts)),
                thread_name_prefix="gliner-chunk",
            ) as executor:
                return list(
                    executor.map(
                        lambda chunk: self._predict_one(chunk, labels),
                        texts,
                    )
                )
        return [self._predict_one(texts[0], labels)]

    def _predict_one(
        self,
        text: str,
        labels: list[str],
    ) -> list[Mapping[str, Any]]:
        return self._model.predict_entities(
            text,
            labels,
            threshold=self.confidence_threshold,
            flat_ner=False,
            multi_label=True,
        )

    def _merge_predictions(
        self,
        chunks: tuple[tuple[int, str], ...],
        prediction_groups: list[list[Mapping[str, Any]]],
    ) -> list[NerEntity]:
        unique: dict[tuple[str, str], NerEntity] = {}
        for (offset, _), predictions in zip(chunks, prediction_groups, strict=True):
            for prediction in predictions:
                entity = self._to_entity(prediction, offset)
                key = (entity.text.casefold().strip(), entity.label.casefold())
                existing = unique.get(key)
                if existing is None or (entity.confidence or 0) > (existing.confidence or 0):
                    unique[key] = entity
        return sorted(
            unique.values(),
            key=lambda entity: (
                entity.start if entity.start is not None else -1,
                entity.end if entity.end is not None else -1,
            ),
        )

    @staticmethod
    def _to_entity(prediction: Mapping[str, Any], offset: int) -> NerEntity:
        return NerEntity(
            text=str(prediction["text"]),
            label=str(prediction["label"]),
            start=int(prediction["start"]) + offset,
            end=int(prediction["end"]) + offset,
            confidence=float(prediction["score"]),
        )
