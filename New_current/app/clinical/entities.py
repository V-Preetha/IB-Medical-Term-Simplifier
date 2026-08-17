"""Map model-backed clinical NER output into stable public categories."""

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from app.clinical.models import ENTITY_CATEGORIES, NerEntity

_LABEL_MAP = {
    "DISEASE": "diseases",
    "DISORDER": "diseases",
    "DIAGNOSIS": "diseases",
    "SYMPTOM": "symptoms",
    "SIGN": "symptoms",
    "LAB_TEST": "laboratory_tests",
    "TEST": "laboratory_tests",
    "BIOMARKER": "biomarkers",
    "CHEMICAL": "medications",
    "CHEM": "medications",
    "DRUG": "medications",
    "MEDICATION": "medications",
    "ANATOMY": "anatomy",
    "BODY_PART": "anatomy",
    "PROCEDURE": "procedures",
    "TREATMENT": "procedures",
    "MEASUREMENT": "measurements",
    "VITAL_SIGN": "vital_signs",
    "MEDICAL PROCEDURE": "procedures",
    "LABORATORY TEST": "laboratory_tests",
    "VITAL SIGN": "vital_signs",
}
_EXCLUDED_EXACT = frozenset(
    {
        "communication from",
        "complaint letter",
        "consultant summary",
        "consultant's summary",
        "contrast",
        "records reviewed",
        "standard of care",
        "symptom",
        "symptoms",
    }
)
_CATEGORY_OVERRIDES = {"ectasia": "diseases"}
_CATEGORY_PRIORITY = {
    category: priority
    for priority, category in enumerate(
        (
            "diseases",
            "medications",
            "procedures",
            "laboratory_tests",
            "biomarkers",
            "vital_signs",
            "symptoms",
            "anatomy",
            "measurements",
        )
    )
}
_DISTINCT_LOCATIONS = frozenset(
    {
        "abdominal",
        "celiac",
        "cerebral",
        "distal",
        "infrarenal",
        "intracranial",
        "left",
        "proximal",
        "renal",
        "right",
        "thoracic",
        "thoraco",
    }
)
_CLINICAL_HEADS = frozenset(
    {
        "aneurysm",
        "cancer",
        "disease",
        "failure",
        "fracture",
        "pain",
        "syndrome",
    }
)
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class _Candidate:
    start: int
    end: int
    surface: str
    category: str
    confidence: float


class MedicalEntityExtractor:
    """Filter and consolidate source-grounded model spans."""

    def extract(
        self,
        text: str,
        ner_entities: Iterable[NerEntity] = (),
    ) -> dict[str, tuple[str, ...]]:
        candidates = self._source_candidates(text, ner_entities)
        selected = self._resolve_overlaps(candidates)
        selected = self._resolve_cross_category_duplicates(selected)
        grouped: dict[str, list[_Candidate]] = {category: [] for category in ENTITY_CATEGORIES}
        for candidate in selected:
            grouped[candidate.category].append(candidate)
        return {
            category: self._deduplicate_category(grouped[category])
            for category in ENTITY_CATEGORIES
        }

    def _source_candidates(
        self,
        text: str,
        ner_entities: Iterable[NerEntity],
    ) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        for entity in ner_entities:
            category = _LABEL_MAP.get(entity.label.upper())
            if category is None or not entity.text.strip():
                continue
            position = self._resolve_position(text, entity)
            if position is None:
                continue
            surface = text[position : position + len(entity.text)].strip()
            canonical = self._canonical(surface)
            if not canonical or canonical in _EXCLUDED_EXACT:
                continue
            candidates.append(
                _Candidate(
                    start=position,
                    end=position + len(surface),
                    surface=surface,
                    category=_CATEGORY_OVERRIDES.get(canonical, category),
                    confidence=entity.confidence or 0.0,
                )
            )
        return candidates

    @staticmethod
    def _resolve_overlaps(candidates: list[_Candidate]) -> list[_Candidate]:
        selected: list[_Candidate] = []
        for candidate in sorted(
            candidates,
            key=lambda item: (
                -item.confidence,
                -(item.end - item.start),
                item.start,
            ),
        ):
            if any(
                candidate.start < chosen.end and candidate.end > chosen.start for chosen in selected
            ):
                continue
            selected.append(candidate)
        return selected

    def _resolve_cross_category_duplicates(
        self,
        candidates: list[_Candidate],
    ) -> list[_Candidate]:
        grouped: dict[str, list[_Candidate]] = defaultdict(list)
        for candidate in candidates:
            grouped[self._canonical(candidate.surface)].append(candidate)
        output: list[_Candidate] = []
        for canonical, variants in grouped.items():
            category = _CATEGORY_OVERRIDES.get(canonical)
            if category is None:
                category = min(
                    variants,
                    key=lambda item: (
                        -item.confidence,
                        _CATEGORY_PRIORITY[item.category],
                    ),
                ).category
            output.extend(
                _Candidate(
                    item.start,
                    item.end,
                    item.surface,
                    category,
                    item.confidence,
                )
                for item in variants
            )
        return output

    def _deduplicate_category(
        self,
        candidates: list[_Candidate],
    ) -> tuple[str, ...]:
        unique: list[_Candidate] = []
        seen: set[str] = set()
        for candidate in sorted(candidates, key=lambda item: item.start):
            canonical = self._canonical(candidate.surface)
            if canonical not in seen:
                seen.add(canonical)
                unique.append(candidate)
        return tuple(
            candidate.surface
            for candidate in unique
            if not self._is_subsumed_generic(candidate, unique)
        )

    def _is_subsumed_generic(
        self,
        candidate: _Candidate,
        all_candidates: list[_Candidate],
    ) -> bool:
        candidate_tokens = set(self._canonical(candidate.surface).split())
        candidate_head = candidate_tokens & _CLINICAL_HEADS
        if not candidate_tokens:
            return True
        for other in all_candidates:
            if other is candidate:
                continue
            other_tokens = set(self._canonical(other.surface).split())
            if not candidate_tokens < other_tokens:
                continue
            if candidate_head != other_tokens & _CLINICAL_HEADS:
                continue
            added = other_tokens - candidate_tokens
            if len(candidate_tokens) <= 2 or not added & _DISTINCT_LOCATIONS:
                return True
        return False

    @staticmethod
    def _canonical(value: str) -> str:
        normalized = _NON_WORD.sub(" ", value.casefold()).strip()
        return " ".join(
            token[:-1] if token == "aneurysms" else token for token in normalized.split()
        )

    @staticmethod
    def _resolve_position(text: str, entity: NerEntity) -> int | None:
        expected = entity.start if entity.start is not None else 0
        if (
            entity.start is not None
            and text[entity.start : entity.start + len(entity.text)].casefold()
            == entity.text.casefold()
        ):
            return entity.start
        matches = tuple(
            match.start() for match in re.finditer(re.escape(entity.text), text, re.IGNORECASE)
        )
        return min(matches, key=lambda position: abs(position - expected)) if matches else None
