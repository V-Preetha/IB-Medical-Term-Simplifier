"""Post-NER orchestration for structured medical output."""

from collections.abc import Callable, Iterable
from inspect import signature
from time import perf_counter

from app.clinical.entities import MedicalEntityExtractor
from app.clinical.labs import LabResultExtractor
from app.clinical.models import NerEntity, StructuredMedicalOutput
from app.clinical.report import MedicalReportSimplifier, StructuredReportBuilder
from app.performance import PipelineTimings


class MedicalDocumentProcessor:
    """Convert normalized text plus clinical NER spans into stable downstream JSON."""

    def __init__(
        self,
        entity_extractor: MedicalEntityExtractor | None = None,
        lab_extractor: LabResultExtractor | None = None,
        report_builder: StructuredReportBuilder | None = None,
        simplifier: MedicalReportSimplifier | None = None,
    ) -> None:
        self.entity_extractor = entity_extractor or MedicalEntityExtractor()
        self.lab_extractor = lab_extractor or LabResultExtractor()
        self.report_builder = report_builder or StructuredReportBuilder()
        self.simplifier = simplifier or self.report_builder

    def process(
        self,
        normalized_text: str,
        ner_entities: Iterable[NerEntity] = (),
        *,
        timings: PipelineTimings | None = None,
        progress: Callable[[str, int], None] | None = None,
    ) -> StructuredMedicalOutput:
        entity_started_at = perf_counter()
        entities = self.entity_extractor.extract(normalized_text, ner_entities)
        if timings is not None:
            timings.record("Entity Merge", (perf_counter() - entity_started_at) * 1000)

        if progress is not None:
            progress("Extracting Labs", 65)
        lab_started_at = perf_counter()
        lab_results = self.lab_extractor.extract(normalized_text)
        if timings is not None:
            timings.record("Lab Extraction", (perf_counter() - lab_started_at) * 1000)
        entities = dict(entities)
        for lab_result in lab_results:
            target = (
                "vital_signs"
                if lab_result.name.casefold() in {"blood pressure", "bp", "heart rate", "pulse"}
                else "laboratory_tests"
            )
            entities[target] = self._merge(
                entities[target],
                (lab_result.name,),
            )
        if progress is not None:
            progress("Generating Summary", 80)
        simplify_parameters = signature(self.simplifier.simplify).parameters
        if "timings" in simplify_parameters:
            sections = self.simplifier.simplify(
                normalized_text,
                entities,
                lab_results,
                timings=timings,
            )
        else:
            qwen_started_at = perf_counter()
            sections = self.simplifier.simplify(
                normalized_text,
                entities,
                lab_results,
            )
            if timings is not None:
                timings.record(
                    "Qwen Simplification",
                    (perf_counter() - qwen_started_at) * 1000,
                )

        if progress is not None:
            progress("Finalizing", 95)
        json_started_at = perf_counter()
        output = StructuredMedicalOutput(
            summary=sections.executive_summary,
            entities=entities,
            lab_results=lab_results,
            simplification=sections,
            human_readable_report=self.report_builder.render(sections),
        )
        if timings is not None:
            timings.record("JSON Build", (perf_counter() - json_started_at) * 1000)
        return output

    @staticmethod
    def _merge(existing: tuple[str, ...], additions: Iterable[str]) -> tuple[str, ...]:
        output = list(existing)
        seen = {value.casefold() for value in existing}
        for value in additions:
            if value.casefold() not in seen:
                seen.add(value.casefold())
                output.append(value)
        return tuple(output)
