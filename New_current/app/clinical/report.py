"""Build stable structured and human-readable views from extracted facts."""

from typing import Protocol

from app.clinical.models import (
    ENTITY_CATEGORIES,
    LabResult,
    SimplificationSections,
)


class MedicalReportSimplifier(Protocol):
    """Interface implemented by the production Qwen provider and test doubles."""

    def simplify(
        self,
        report_text: str,
        entities: dict[str, tuple[str, ...]],
        lab_results: tuple[LabResult, ...],
    ) -> SimplificationSections: ...


class StructuredReportBuilder:
    """Render sections and provide a source-grounded fallback for isolated use."""

    _DISPLAY_NAMES = {
        "diseases": "Diseases",
        "symptoms": "Symptoms",
        "laboratory_tests": "Laboratory Tests",
        "biomarkers": "Biomarkers",
        "medications": "Medications",
        "anatomy": "Anatomy",
        "procedures": "Procedures",
        "measurements": "Measurements",
        "vital_signs": "Vital Signs",
    }

    def build_sections(
        self,
        entities: dict[str, tuple[str, ...]],
        lab_results: tuple[LabResult, ...],
    ) -> SimplificationSections:
        """Build a deterministic fallback for tests and model-disabled environments."""

        populated = [
            self._DISPLAY_NAMES[category]
            for category in ENTITY_CATEGORIES
            if entities.get(category)
        ]
        summary = (
            "Explicit medical information was extracted in these categories: "
            + ", ".join(populated)
            + "."
            if populated
            else "No configured medical entities were explicitly identified."
        )
        findings = tuple(
            f"{result.name}: {result.value} {result.unit}".rstrip() for result in lab_results
        )
        medical_terms = tuple(
            value for category in ENTITY_CATEGORIES for value in entities.get(category, ())
        )
        explanations = [
            f"{self._DISPLAY_NAMES[category]} documented: {', '.join(entities[category])}."
            for category in ENTITY_CATEGORIES
            if entities.get(category)
        ]
        explanation = " ".join(explanations) or (
            "The report text is preserved for downstream clinical interpretation."
        )
        return SimplificationSections(
            executive_summary=summary,
            important_findings=findings,
            medical_terms=medical_terms,
            simplified_explanation=explanation,
            recommended_follow_up=(
                "Review the extracted findings in the context of the original report.",
            ),
        )

    def simplify(
        self,
        report_text: str,
        entities: dict[str, tuple[str, ...]],
        lab_results: tuple[LabResult, ...],
    ) -> SimplificationSections:
        """Support unit tests; the FastAPI lifespan injects the real Qwen provider."""

        del report_text
        return self.build_sections(entities, lab_results)

    def render(self, sections: SimplificationSections) -> str:
        return "\n\n".join(
            (
                self._section("Executive Summary", (sections.executive_summary,)),
                self._section("Important Findings", sections.important_findings),
                self._section("Timeline", sections.timeline),
                self._section("Medical Terms Explained", sections.medical_terms),
                self._section(
                    "Simple Explanation",
                    (sections.simplified_explanation,),
                ),
                self._section(
                    "Recommended Follow-up",
                    sections.recommended_follow_up,
                ),
            )
        )

    @staticmethod
    def _section(title: str, lines: tuple[str, ...]) -> str:
        content = "\n".join(f"• {line}" for line in lines) if lines else "None identified."
        return f"{title}\n{'-' * len(title)}\n{content}"
