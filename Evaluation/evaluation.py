import logging
from statistics import mean
from typing import Any, Dict, List

import numpy as np

from entity_metrics import (
    MedicalEntityExtractor,
    critical_safety_errors,
    entity_precision_recall_f1,
    hallucinated_entities,
    jargon_reduction_percentage,
)
from readability import fk_in_target_band, readability_scores
from semantic import SemanticScorer
from utils import safe_divide


LOGGER = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, config: Dict[str, Any]) -> None:
        evaluation_config = config.get("evaluation", {})
        device = evaluation_config.get("device", "auto")
        self.semantic = SemanticScorer(evaluation_config["embedding_model"], device=device)
        self.entities = MedicalEntityExtractor(evaluation_config["entity_model"], device=device)

    def evaluate_row(self, original: str, simplified: str, level: str) -> Dict[str, Any]:
        original_entities = self.entities.extract(original)
        simplified_entities = self.entities.extract(simplified)
        entity_metrics = entity_precision_recall_f1(original_entities, simplified_entities)
        readability = readability_scores(simplified)
        hallucinations = hallucinated_entities(original_entities, simplified_entities)
        critical_errors = critical_safety_errors(
            original, simplified, original_entities, simplified_entities
        )

        result = {
            "semantic_similarity": self.semantic.similarity(original, simplified),
            **entity_metrics,
            **readability,
            "readability_in_target_band": fk_in_target_band(level, readability["flesch_kincaid_grade"]),
            "jargon_reduction_percent": jargon_reduction_percentage(original_entities, simplified_entities),
            "hallucinated_entities": hallucinations,
            "hallucination_count": sum(len(values) for values in hallucinations.values()),
            "critical_errors": critical_errors,
            "critical_error_count": len(critical_errors),
            "original_entities": original_entities,
            "simplified_entities": simplified_entities,
        }
        return result


def aggregate_metrics(rows: List[Dict[str, Any]], readiness_weights: Dict[str, float]) -> Dict[str, Any]:
    if not rows:
        return {}

    def avg(key: str) -> float:
        return float(mean(float(row.get(key, 0.0)) for row in rows))

    latencies = [float(row.get("latency_seconds", 0.0)) for row in rows]
    critical_errors = sum(int(row.get("critical_error_count", 0)) for row in rows)
    hallucinations = sum(int(row.get("hallucination_count", 0)) for row in rows)
    generated_entities = sum(int(row.get("simplified_entity_count", 0)) for row in rows)
    readability_compliance = safe_divide(
        sum(1 for row in rows if row.get("readability_in_target_band")), len(rows)
    )

    summary = {
        "row_count": len(rows),
        "average_semantic_similarity": avg("semantic_similarity"),
        "average_entity_precision": avg("entity_precision"),
        "average_entity_recall": avg("entity_recall"),
        "average_entity_f1": avg("entity_f1"),
        "average_flesch_reading_ease": avg("flesch_reading_ease"),
        "average_flesch_kincaid_grade": avg("flesch_kincaid_grade"),
        "average_smog": avg("smog"),
        "average_gunning_fog": avg("gunning_fog"),
        "readability_compliance_rate": readability_compliance,
        "average_jargon_reduction_percent": avg("jargon_reduction_percent"),
        "hallucination_count": hallucinations,
        "hallucination_rate": safe_divide(hallucinations, generated_entities),
        "critical_error_count": critical_errors,
        "critical_error_rate": safe_divide(critical_errors, len(rows)),
        "average_latency_seconds": float(mean(latencies)) if latencies else 0.0,
        "p95_latency_seconds": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "peak_ram_mb": max(float(row.get("peak_ram_mb", 0.0)) for row in rows),
        "peak_cpu_percent": max(float(row.get("peak_cpu_percent", 0.0)) for row in rows),
        "peak_gpu_vram_mb": max(float(row.get("gpu_peak_vram_mb", 0.0)) for row in rows),
        "model_loading_time_seconds": max(float(row.get("model_loading_time_seconds", 0.0)) for row in rows),
        "model_size_mb": max(float(row.get("model_size_mb", 0.0)) for row in rows),
    }
    summary["overall_readiness_score"] = readiness_score(summary, readiness_weights)
    return summary


def readiness_score(summary: Dict[str, Any], weights: Dict[str, float]) -> float:
    total_weight = sum(float(value) for value in weights.values()) or 1.0
    semantic = max(0.0, min(1.0, summary.get("average_semantic_similarity", 0.0)))
    recall = max(0.0, min(1.0, summary.get("average_entity_recall", 0.0)))
    readability = max(0.0, min(1.0, summary.get("readability_compliance_rate", 0.0)))
    hallucination = max(0.0, 1.0 - min(1.0, summary.get("hallucination_rate", 0.0)))
    critical = 1.0 if summary.get("critical_error_count", 0) == 0 else 0.0
    latency = 1.0 / (1.0 + max(0.0, summary.get("average_latency_seconds", 0.0)) / 10.0)

    components = {
        "semantic_similarity": semantic,
        "entity_recall": recall,
        "readability_compliance": readability,
        "hallucination": hallucination,
        "critical_safety": critical,
        "latency": latency,
    }
    weighted = sum(components[key] * float(weights.get(key, 0.0)) for key in components)
    return round(100.0 * weighted / total_weight, 2)
