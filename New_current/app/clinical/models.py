"""Typed outputs for downstream medical NLP, translation, and presentation."""

from dataclasses import dataclass, field

ENTITY_CATEGORIES = (
    "diseases",
    "symptoms",
    "laboratory_tests",
    "biomarkers",
    "medications",
    "anatomy",
    "procedures",
    "measurements",
    "vital_signs",
)


@dataclass(frozen=True, slots=True)
class NerEntity:
    """Model-neutral representation of a clinical NER entity."""

    text: str
    label: str
    start: int | None = None
    end: int | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class LabResult:
    name: str
    value: str
    unit: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "value": self.value, "unit": self.unit}


@dataclass(frozen=True, slots=True)
class SimplificationSections:
    executive_summary: str
    important_findings: tuple[str, ...] = ()
    timeline: tuple[str, ...] = ()
    medical_terms: tuple[str, ...] = ()
    simplified_explanation: str = ""
    recommended_follow_up: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, str | list[str]]:
        return {
            "executive_summary": self.executive_summary,
            "important_findings": list(self.important_findings),
            "medical_terms": list(self.medical_terms),
            "simplified_explanation": self.simplified_explanation,
            "recommended_follow_up": list(self.recommended_follow_up),
        }


@dataclass(frozen=True, slots=True)
class StructuredMedicalOutput:
    summary: str
    entities: dict[str, tuple[str, ...]] = field(default_factory=dict)
    lab_results: tuple[LabResult, ...] = ()
    simplification: SimplificationSections = field(
        default_factory=lambda: SimplificationSections("")
    )
    human_readable_report: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary,
            "entities": {
                category: list(self.entities.get(category, ())) for category in ENTITY_CATEGORIES
            },
            "lab_results": [result.to_dict() for result in self.lab_results],
            "simplification": self.simplification.to_dict(),
            "human_readable_report": self.human_readable_report,
        }
