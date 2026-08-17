"""Clinical section segmentation for extracted medical report text."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum

from app.services.document_parsing import clean_extracted_text

logger = logging.getLogger(__name__)


class SectionSegmentationError(RuntimeError):
    """Raised when clinical section segmentation cannot be completed."""


class ClinicalSectionType(StrEnum):
    """Normalized clinical section labels supported by the pipeline."""

    DIAGNOSIS = "diagnosis"
    MEDICATIONS = "medications"
    LAB_RESULTS = "lab_results"
    FINDINGS = "findings"
    IMPRESSION = "impression"
    RECOMMENDATIONS = "recommendations"
    HISTORY = "history"
    PROCEDURES = "procedures"
    OTHER = "other"


@dataclass(frozen=True)
class ClinicalSection:
    """A segmented clinical section with normalized metadata."""

    section_type: ClinicalSectionType
    title: str
    content: str
    order: int
    confidence: float


@dataclass(frozen=True)
class SegmentedReport:
    """Structured clinical sections extracted from a report."""

    sections: list[ClinicalSection]
    section_count: int
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _HeadingMatch:
    """Internal representation of a detected heading."""

    section_type: ClinicalSectionType
    title: str
    start: int
    end: int
    confidence: float


class SectionSegmentationService:
    """Split extracted medical report text into normalized clinical sections."""

    _heading_pattern = re.compile(
        r"(?im)^(?P<title>[A-Z][A-Z0-9 /,&()\-]{1,70}|[A-Z][A-Za-z0-9 /,&()\-]{1,70}):\s*"
    )

    _heading_aliases: dict[ClinicalSectionType, tuple[str, ...]] = {
        ClinicalSectionType.DIAGNOSIS: (
            "diagnosis",
            "diagnoses",
            "assessment",
            "clinical diagnosis",
            "final diagnosis",
        ),
        ClinicalSectionType.MEDICATIONS: (
            "medication",
            "medications",
            "current medications",
            "medicine",
            "drug therapy",
            "prescriptions",
        ),
        ClinicalSectionType.LAB_RESULTS: (
            "lab",
            "labs",
            "laboratory",
            "laboratory results",
            "lab results",
            "investigations",
            "blood tests",
        ),
        ClinicalSectionType.FINDINGS: (
            "findings",
            "clinical findings",
            "radiology findings",
            "exam findings",
        ),
        ClinicalSectionType.IMPRESSION: (
            "impression",
            "conclusion",
            "summary",
        ),
        ClinicalSectionType.RECOMMENDATIONS: (
            "recommendation",
            "recommendations",
            "plan",
            "treatment plan",
            "follow up",
            "follow-up",
            "advice",
        ),
        ClinicalSectionType.HISTORY: (
            "history",
            "medical history",
            "past medical history",
            "clinical history",
            "history of present illness",
            "chief complaint",
        ),
        ClinicalSectionType.PROCEDURES: (
            "procedure",
            "procedures",
            "operation",
            "surgery",
            "intervention",
        ),
    }

    def segment(self, text: str) -> SegmentedReport:
        """Segment cleaned report text into clinical sections."""
        cleaned_text = clean_extracted_text(text)
        if not cleaned_text:
            raise SectionSegmentationError("Report text is empty after cleaning.")

        heading_matches = self._find_headings(cleaned_text)
        if not heading_matches:
            logger.info("No clinical headings detected; returning one other section")
            return SegmentedReport(
                sections=[
                    ClinicalSection(
                        section_type=ClinicalSectionType.OTHER,
                        title="Other",
                        content=cleaned_text,
                        order=0,
                        confidence=0.5,
                    )
                ],
                section_count=1,
                warnings=["No known clinical section headings were detected."],
            )

        sections: list[ClinicalSection] = []
        if heading_matches[0].start > 0:
            preamble = clean_extracted_text(cleaned_text[: heading_matches[0].start])
            if preamble:
                sections.append(
                    ClinicalSection(
                        section_type=ClinicalSectionType.OTHER,
                        title="Other",
                        content=preamble,
                        order=len(sections),
                        confidence=0.5,
                    )
                )

        for index, heading in enumerate(heading_matches):
            next_start = (
                heading_matches[index + 1].start
                if index + 1 < len(heading_matches)
                else len(cleaned_text)
            )
            content = clean_extracted_text(cleaned_text[heading.end : next_start])
            if not content:
                logger.debug("Skipping empty section for heading %s", heading.title)
                continue
            sections.append(
                ClinicalSection(
                    section_type=heading.section_type,
                    title=heading.title,
                    content=content,
                    order=len(sections),
                    confidence=heading.confidence,
                )
            )

        if not sections:
            raise SectionSegmentationError("No section content could be segmented.")

        logger.info("Segmented report into %d section(s)", len(sections))
        return SegmentedReport(sections=sections, section_count=len(sections))

    def _find_headings(self, text: str) -> list[_HeadingMatch]:
        """Find clinical headings and normalize them to supported section types."""
        headings: list[_HeadingMatch] = []
        for match in self._heading_pattern.finditer(text):
            raw_title = match.group("title").strip()
            normalized = _normalize_heading(raw_title)
            section_type = self._map_heading(normalized)
            if section_type is None:
                logger.debug("Ignoring unknown heading candidate: %s", raw_title)
                continue

            confidence = 0.95 if normalized in self._all_aliases() else 0.85
            headings.append(
                _HeadingMatch(
                    section_type=section_type,
                    title=raw_title.title(),
                    start=match.start(),
                    end=match.end(),
                    confidence=confidence,
                )
            )
        return headings

    def _map_heading(self, normalized_heading: str) -> ClinicalSectionType | None:
        """Map a normalized heading string to a clinical section type."""
        for section_type, aliases in self._heading_aliases.items():
            if normalized_heading in aliases:
                return section_type
        return None

    def _all_aliases(self) -> set[str]:
        """Return every exact heading alias."""
        return {
            alias
            for aliases in self._heading_aliases.values()
            for alias in aliases
        }


def _normalize_heading(raw_heading: str) -> str:
    """Normalize report headings for robust alias matching."""
    normalized = raw_heading.strip().lower()
    normalized = normalized.replace("&", "and")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()
