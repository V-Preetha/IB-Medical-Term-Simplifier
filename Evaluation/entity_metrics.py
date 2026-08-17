import logging
import re
from collections import defaultdict
from typing import Dict, List, Set

from utils import flatten_entities, normalize_entity, safe_divide, unique_normalized


LOGGER = logging.getLogger(__name__)

ENTITY_TYPES = ("diseases", "drugs", "procedures", "symptoms", "anatomy", "lab_values")

DOSAGE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|g|kg|ml|l|iu|units?|tabs?|tablets?|capsules?|%)\b",
    re.IGNORECASE,
)
DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})\b",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"(?<![\w/])\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?(?![\w/])")
LAB_VALUE_RE = re.compile(
    r"\b(?:HbA1c|A1c|glucose|creatinine|hemoglobin|haemoglobin|WBC|RBC|platelets?|"
    r"cholesterol|LDL|HDL|triglycerides?|sodium|potassium|ALT|AST|TSH|INR|BP|blood pressure)"
    r"[^.\n;:]{0,40}?\d+(?:\.\d+)?(?:\s?(?:mg/dL|mmol/L|g/dL|%|mmHg|mEq/L|IU/L))?\b",
    re.IGNORECASE,
)


def extract_regex_facts(text: str) -> Dict[str, List[str]]:
    return {
        "dosages": DOSAGE_RE.findall(text),
        "dates": DATE_RE.findall(text),
        "numbers": NUMBER_RE.findall(text),
        "lab_values": LAB_VALUE_RE.findall(text),
    }


class MedicalEntityExtractor:
    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.model_name = model_name
        self.device = self._pipeline_device(device)
        self.pipeline = None
        self.load_error = None
        self._load()

    @staticmethod
    def _pipeline_device(device: str) -> int:
        if device == "auto":
            try:
                import torch

                return 0 if torch.cuda.is_available() else -1
            except Exception:
                return -1
        if device == "cpu":
            return -1
        if device.startswith("cuda"):
            return 0
        return -1

    def _load(self) -> None:
        try:
            from transformers import pipeline

            self.pipeline = pipeline(
                "token-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                aggregation_strategy="simple",
                device=self.device,
                trust_remote_code=True,
            )
        except Exception as exc:
            self.load_error = str(exc)
            LOGGER.warning(
                "OpenMed PharmaDetect entity model could not be loaded (%s). "
                "Regex safety facts will still be extracted, but entity metrics are limited.",
                exc,
            )

    @staticmethod
    def _normalize_label(label: str) -> str:
        label = label.lower()
        if any(term in label for term in ("disease", "diagnosis", "problem", "disorder")):
            return "diseases"
        if any(term in label for term in ("drug", "med", "chemical", "pharma", "treatment")):
            return "drugs"
        if any(term in label for term in ("procedure", "test", "therapy", "surgery")):
            return "procedures"
        if any(term in label for term in ("symptom", "sign")):
            return "symptoms"
        if any(term in label for term in ("anatomy", "body", "organ", "location")):
            return "anatomy"
        if any(term in label for term in ("lab", "value", "measurement")):
            return "lab_values"
        return "unmapped"

    def extract(self, text: str) -> Dict[str, List[str]]:
        entities: Dict[str, List[str]] = {entity_type: [] for entity_type in ENTITY_TYPES}
        regex_facts = extract_regex_facts(text)
        entities["lab_values"].extend(regex_facts["lab_values"])

        if self.pipeline is None:
            return entities

        try:
            for item in self.pipeline(text):
                label = item.get("entity_group") or item.get("entity") or ""
                entity_type = self._normalize_label(label)
                if entity_type in entities:
                    value = item.get("word", "")
                    if value:
                        entities[entity_type].append(value)
        except Exception as exc:
            LOGGER.exception("Entity extraction failed: %s", exc)

        return {key: sorted(unique_normalized(values)) for key, values in entities.items()}


def entity_precision_recall_f1(original: Dict[str, List[str]], simplified: Dict[str, List[str]]) -> Dict[str, float]:
    original_set = flatten_entities(original)
    simplified_set = flatten_entities(simplified)
    overlap = original_set & simplified_set
    precision = safe_divide(len(overlap), len(simplified_set))
    recall = safe_divide(len(overlap), len(original_set))
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": f1,
        "original_entity_count": len(original_set),
        "simplified_entity_count": len(simplified_set),
    }


def hallucinated_entities(original: Dict[str, List[str]], simplified: Dict[str, List[str]]) -> Dict[str, List[str]]:
    hallucinations: Dict[str, List[str]] = {}
    for entity_type in ENTITY_TYPES:
        original_set = unique_normalized(original.get(entity_type, []))
        simplified_set = unique_normalized(simplified.get(entity_type, []))
        added = sorted(simplified_set - original_set)
        if added:
            hallucinations[entity_type] = added
    return hallucinations


def jargon_reduction_percentage(original: Dict[str, List[str]], simplified: Dict[str, List[str]]) -> float:
    original_set = flatten_entities(original)
    simplified_set = flatten_entities(simplified)
    if not original_set:
        return 0.0
    removed = original_set - simplified_set
    return safe_divide(len(removed), len(original_set)) * 100


def critical_safety_errors(original_text: str, simplified_text: str, original_entities: Dict[str, List[str]], simplified_entities: Dict[str, List[str]]) -> List[Dict[str, str]]:
    errors: List[Dict[str, str]] = []

    for label, entity_type in (("Diagnoses", "diseases"), ("Drug names", "drugs")):
        original_set = unique_normalized(original_entities.get(entity_type, []))
        simplified_set = unique_normalized(simplified_entities.get(entity_type, []))
        missing = sorted(original_set - simplified_set)
        added = sorted(simplified_set - original_set)
        for value in missing:
            errors.append({"type": f"{label} missing", "value": value})
        for value in added:
            errors.append({"type": f"{label} added", "value": value})

    original_facts = extract_regex_facts(original_text)
    simplified_facts = extract_regex_facts(simplified_text)
    checks = {
        "Dosages": "dosages",
        "Numerical values": "numbers",
        "Dates": "dates",
        "Lab values": "lab_values",
    }
    for label, key in checks.items():
        original_set = unique_normalized(original_facts.get(key, []))
        simplified_set = unique_normalized(simplified_facts.get(key, []))
        for value in sorted(original_set - simplified_set):
            errors.append({"type": f"{label} missing", "value": value})
        for value in sorted(simplified_set - original_set):
            errors.append({"type": f"{label} added", "value": value})

    return errors
