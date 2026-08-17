import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set


LEVELS = ("Beginner", "Intermediate", "Advanced")


def setup_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_config(path: str = "config.json") -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_output_dirs(config: Dict[str, Any]) -> None:
    paths = config.get("paths", {})
    for key in ("results_dir", "metrics_dir", "plots_dir"):
        Path(paths.get(key, key)).mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip()).lower()


def normalize_entity(entity: str) -> str:
    entity = normalize_text(entity)
    entity = re.sub(r"^[^\w.]+|[^\w.%/-]+$", "", entity)
    return entity


def unique_normalized(values: Iterable[str]) -> Set[str]:
    return {v for v in (normalize_entity(value) for value in values) if v}


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    import numpy as np

    return float(np.percentile(np.asarray(values, dtype=float), pct))


def medical_simplification_prompt(report: str, level: str) -> str:
    return (
        "You are a medical report simplification assistant. Simplify the report for a "
        f"{level} reader.\n\n"
        "Requirements:\n"
        "- Preserve all diagnoses, drug names, dosages, dates, lab values, measurements, "
        "procedures, anatomy, symptoms, and clinical meaning exactly.\n"
        "- Do not add new medical facts.\n"
        "- Do not remove clinically important information.\n"
        "- Use plain language appropriate for the requested level.\n"
        "- Return only the simplified report.\n\n"
        f"Medical report:\n{report}"
    )


def flatten_entities(entities_by_type: Dict[str, List[str]]) -> Set[str]:
    flattened: Set[str] = set()
    for values in entities_by_type.values():
        flattened.update(unique_normalized(values))
    return flattened


def model_slug(model_key: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", model_key).strip("_").lower()
