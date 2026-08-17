from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np


def _save_bar(path: Path, title: str, ylabel: str, values: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(values.keys())
    scores = [values[name] for name in names]
    plt.figure(figsize=(8, 5))
    plt.bar(names, scores, color=["#2E7D8A", "#D67C45"][: len(names)])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def create_plots(metrics_by_model: Dict[str, Dict[str, Any]], plots_dir: str) -> None:
    out_dir = Path(plots_dir)
    _save_bar(
        out_dir / "latency.png",
        "Average Inference Latency",
        "Seconds",
        {name: m.get("average_latency_seconds", 0.0) for name, m in metrics_by_model.items()},
    )
    _save_bar(
        out_dir / "readability.png",
        "Average Flesch-Kincaid Grade",
        "Grade",
        {name: m.get("average_flesch_kincaid_grade", 0.0) for name, m in metrics_by_model.items()},
    )
    _save_bar(
        out_dir / "semantic_similarity.png",
        "Average Semantic Similarity",
        "Cosine Similarity",
        {name: m.get("average_semantic_similarity", 0.0) for name, m in metrics_by_model.items()},
    )
    _save_bar(
        out_dir / "entity_recall.png",
        "Average Entity Recall",
        "Recall",
        {name: m.get("average_entity_recall", 0.0) for name, m in metrics_by_model.items()},
    )
    _radar_chart(metrics_by_model, out_dir / "radar_chart.png")


def _radar_chart(metrics_by_model: Dict[str, Dict[str, Any]], path: Path) -> None:
    labels = [
        "Semantic",
        "Entity Recall",
        "Readability",
        "Low Hallucination",
        "Safety",
        "Latency",
    ]
    keys = [
        "average_semantic_similarity",
        "average_entity_recall",
        "readability_compliance_rate",
        "hallucination_rate",
        "critical_error_count",
        "average_latency_seconds",
    ]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    for name, metrics in metrics_by_model.items():
        latency_score = 1.0 / (1.0 + metrics.get("average_latency_seconds", 0.0) / 10.0)
        values = [
            metrics.get("average_semantic_similarity", 0.0),
            metrics.get("average_entity_recall", 0.0),
            metrics.get("readability_compliance_rate", 0.0),
            1.0 - min(1.0, metrics.get("hallucination_rate", 0.0)),
            1.0 if metrics.get("critical_error_count", 0) == 0 else 0.0,
            latency_score,
        ]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=name)
        ax.fill(angles, values, alpha=0.12)

    ax.set_ylim(0, 1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("Model Readiness Radar")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
