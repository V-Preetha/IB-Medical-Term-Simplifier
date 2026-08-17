"""Application policy combining deterministic grounding with MedNLI."""
# ruff: noqa: E501

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from time import perf_counter
from uuid import UUID

from app.verification.contracts import BaseVerificationProvider

_NUMERIC = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
    r"\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?\s*(?:%|mg/dL|g/dL|mmol/L|mEq/L|"
    r"U/L|IU/L|mmHg|bpm|mg|mcg|mL|mm|cm)?)\b",
    re.IGNORECASE,
)
_FREQUENCY = re.compile(
    r"\b(?:once|twice|three times|every\s+\d+\s+hours|daily|weekly|"
    r"at bedtime|morning|evening|bid|tid|qid)\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(r"\b(?:no|not|denies|without|negative for)\b", re.IGNORECASE)
_LATERALITY = re.compile(r"\b(?:left|right|bilateral)\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class VerificationResult:
    request_id: UUID
    label: str
    probabilities: dict[str, float]
    status: str
    review_required: bool
    deterministic_mismatches: tuple[dict[str, object], ...]
    processing_time_ms: float
    nli_latency_ms: float
    model_name: str
    model_revision: str


class MedicalVerificationService:
    def __init__(self, provider: BaseVerificationProvider) -> None:
        self.provider = provider

    async def initialize(self, *, strict: bool = True) -> None:
        await self.provider.initialize(strict=strict)

    async def verify(self, premise: str, hypothesis: str, request_id: UUID) -> VerificationResult:
        started = perf_counter()
        mismatches = self._deterministic_mismatches(premise, hypothesis)
        inference = await asyncio.to_thread(self.provider.infer, premise, hypothesis)
        if mismatches or inference.label == "contradiction":
            status, review = "BLOCKED", True
        elif inference.label == "neutral":
            status, review = "REVIEW", True
        else:
            status, review = "PASS", False
        metadata = self.provider.metadata()
        return VerificationResult(
            request_id,
            inference.label,
            inference.probabilities,
            status,
            review,
            tuple(mismatches),
            round((perf_counter() - started) * 1000, 3),
            inference.latency_ms,
            metadata.model_name,
            metadata.model_revision,
        )

    @staticmethod
    def _deterministic_mismatches(
        premise: str, hypothesis: str
    ) -> list[dict[str, object]]:
        checks = (
            ("numeric_or_unit", _NUMERIC),
            ("medication_frequency", _FREQUENCY),
            ("negation", _NEGATION),
            ("laterality", _LATERALITY),
        )
        findings: list[dict[str, object]] = []
        for name, pattern in checks:
            source = tuple(dict.fromkeys(match.group(0) for match in pattern.finditer(premise)))
            output = tuple(dict.fromkeys(match.group(0) for match in pattern.finditer(hypothesis)))
            missing = [value for value in source if value.casefold() not in {item.casefold() for item in output}]
            added = [value for value in output if value.casefold() not in {item.casefold() for item in source}]
            if missing or added:
                findings.append({"check": name, "missing": missing, "added": added})
        return findings

    async def shutdown(self) -> None:
        await self.provider.shutdown()
