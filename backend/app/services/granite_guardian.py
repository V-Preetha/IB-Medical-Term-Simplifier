"""IBM Granite Guardian validation for generated simplifications."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from app.config.settings import Settings
from app.fusion.medical_fusion import FusedMedicalTerm
from app.services.document_parsing import clean_extracted_text

logger = logging.getLogger(__name__)


class ValidationError(RuntimeError):
    """Raised when validation cannot be completed."""


class ValidationAction(StrEnum):
    """Recommended action after validation."""

    APPROVE = "approve"
    REJECT = "reject"
    REGENERATE = "regenerate"


@dataclass(frozen=True)
class ValidationCheck:
    """One validation check result."""

    name: str
    passed: bool
    score: float
    details: str


@dataclass(frozen=True)
class GuardianAssessment:
    """Granite Guardian model risk assessment."""

    hallucination_risk: float
    factual_consistency_risk: float
    unsafe_content_risk: float
    terminology_risk: float
    raw_response: str


@dataclass(frozen=True)
class ValidationResult:
    """Validated simplification result and recommended action."""

    validation_passed: bool
    action: ValidationAction
    checks: list[ValidationCheck]
    guardian_assessment: GuardianAssessment
    warnings: list[str] = field(default_factory=list)


class GuardianBackend(Protocol):
    """Protocol for IBM Granite Guardian backends."""

    model_name: str

    def assess(
        self,
        *,
        fused_terms: list[FusedMedicalTerm],
        simplified_report: str,
    ) -> GuardianAssessment:
        """Assess generated output against structured source facts."""


class GraniteGuardianBackend:
    """HuggingFace IBM Granite Guardian generation backend."""

    def __init__(self, settings: Settings) -> None:
        """Initialize a lazy-loading Granite Guardian backend."""
        self.model_name = settings.granite_guardian_model_name
        self._max_new_tokens = settings.granite_guardian_max_new_tokens
        self._tokenizer = None
        self._model = None

    def assess(
        self,
        *,
        fused_terms: list[FusedMedicalTerm],
        simplified_report: str,
    ) -> GuardianAssessment:
        """Ask Granite Guardian to score hallucination and safety risks."""
        tokenizer, model = self._load_model()
        try:
            import torch
        except ImportError as exc:
            raise ValidationError(
                "PyTorch is not installed. Install backend requirements before Granite Guardian validation."
            ) from exc

        prompt = _build_guardian_prompt(
            fused_terms=fused_terms,
            simplified_report=simplified_report,
        )
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            rendered_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered_prompt = prompt

        encoded = tokenizer(rendered_prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][encoded["input_ids"].shape[-1] :]
        raw_response = clean_extracted_text(
            tokenizer.decode(generated_ids, skip_special_tokens=True)
        )
        return _parse_guardian_response(raw_response)

    def _load_model(self) -> tuple[object, object]:
        """Load the configured Granite Guardian model once."""
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ValidationError(
                "Transformers is not installed. Install backend requirements before Granite Guardian validation."
            ) from exc

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._model.eval()
        except Exception as exc:
            logger.exception("Granite Guardian model loading failed")
            raise ValidationError(
                f"Granite Guardian model '{self.model_name}' could not be loaded."
            ) from exc
        return self._tokenizer, self._model


class GraniteGuardianValidationService:
    """Validate Qwen3 output using Granite Guardian and deterministic checks."""

    def __init__(
        self,
        settings: Settings,
        guardian_backend: GuardianBackend | None = None,
    ) -> None:
        """Initialize validation with injectable Guardian backend."""
        self._settings = settings
        self._guardian_backend = guardian_backend or GraniteGuardianBackend(settings)

    def validate(
        self,
        *,
        fused_terms: list[FusedMedicalTerm],
        simplified_report: str,
    ) -> ValidationResult:
        """Validate generated simplification against fused source facts."""
        if not fused_terms:
            raise ValidationError("At least one fused medical term is required.")
        cleaned_report = clean_extracted_text(simplified_report)
        if not cleaned_report:
            raise ValidationError("Simplified report is empty.")

        guardian_assessment = self._guardian_backend.assess(
            fused_terms=fused_terms,
            simplified_report=cleaned_report,
        )
        checks = [
            self._check_guardian_score(
                "hallucination_detection",
                guardian_assessment.hallucination_risk,
                "Granite Guardian hallucination risk.",
            ),
            self._check_guardian_score(
                "factual_consistency",
                guardian_assessment.factual_consistency_risk,
                "Granite Guardian factual consistency risk.",
            ),
            self._check_guardian_score(
                "unsafe_generation",
                guardian_assessment.unsafe_content_risk,
                "Granite Guardian unsafe content risk.",
            ),
            self._check_guardian_score(
                "terminology_consistency",
                guardian_assessment.terminology_risk,
                "Granite Guardian terminology risk.",
            ),
            _check_terms_preserved(fused_terms, cleaned_report),
            _check_meanings_supported(fused_terms, cleaned_report),
            _check_no_unsupported_numbers(fused_terms, cleaned_report),
            _check_no_unsupported_advice(fused_terms, cleaned_report),
        ]
        failed_checks = [check for check in checks if not check.passed]
        action = _choose_action(failed_checks)
        return ValidationResult(
            validation_passed=not failed_checks,
            action=action,
            checks=checks,
            guardian_assessment=guardian_assessment,
            warnings=[
                check.details
                for check in failed_checks
            ],
        )

    def _check_guardian_score(
        self,
        name: str,
        risk_score: float,
        details: str,
    ) -> ValidationCheck:
        """Convert a Guardian risk score into a validation check."""
        return ValidationCheck(
            name=name,
            passed=risk_score < self._settings.validation_failure_threshold,
            score=round(1.0 - risk_score, 4),
            details=details,
        )


def _build_guardian_prompt(
    *,
    fused_terms: list[FusedMedicalTerm],
    simplified_report: str,
) -> str:
    """Build the Granite Guardian validation prompt."""
    source_json = json.dumps(
        [
            {
                "term": term.term,
                "meaning": term.meaning,
                "context": term.context,
                "entity_type": term.entity_type,
                "section_type": term.section_type,
                "ambiguity_resolution": term.ambiguity_resolution,
            }
            for term in fused_terms
        ],
        ensure_ascii=True,
        indent=2,
    )
    return (
        "You are IBM Granite Guardian validating a medical simplification.\n"
        "Compare the generated answer to the structured source facts.\n"
        "Return JSON only with numeric risks from 0.0 to 1.0 for keys: "
        "hallucination_risk, factual_consistency_risk, unsafe_content_risk, terminology_risk.\n\n"
        f"Structured source facts:\n{source_json}\n\n"
        f"Generated simplification:\n{simplified_report}"
    )


def _parse_guardian_response(raw_response: str) -> GuardianAssessment:
    """Parse Granite Guardian JSON risk output with conservative fallback."""
    try:
        json_match = re.search(r"\{.*\}", raw_response, flags=re.DOTALL)
        payload = json.loads(json_match.group(0) if json_match else raw_response)
        return GuardianAssessment(
            hallucination_risk=_bounded_float(payload.get("hallucination_risk", 1.0)),
            factual_consistency_risk=_bounded_float(
                payload.get("factual_consistency_risk", 1.0)
            ),
            unsafe_content_risk=_bounded_float(payload.get("unsafe_content_risk", 1.0)),
            terminology_risk=_bounded_float(payload.get("terminology_risk", 1.0)),
            raw_response=raw_response,
        )
    except Exception:
        logger.warning("Could not parse Granite Guardian response: %s", raw_response)
        return GuardianAssessment(
            hallucination_risk=1.0,
            factual_consistency_risk=1.0,
            unsafe_content_risk=1.0,
            terminology_risk=1.0,
            raw_response=raw_response,
        )


def _check_terms_preserved(
    fused_terms: list[FusedMedicalTerm],
    simplified_report: str,
) -> ValidationCheck:
    """Ensure every fused term appears in the generated explanation."""
    report_lower = simplified_report.lower()
    missing_terms = [
        term.term
        for term in fused_terms
        if term.term.lower() not in report_lower
    ]
    return ValidationCheck(
        name="required_terms_preserved",
        passed=not missing_terms,
        score=1.0 if not missing_terms else 0.0,
        details=(
            "All fused terms are preserved."
            if not missing_terms
            else "Missing required terms: " + ", ".join(missing_terms)
        ),
    )


def _check_meanings_supported(
    fused_terms: list[FusedMedicalTerm],
    simplified_report: str,
) -> ValidationCheck:
    """Ensure generated explanation includes supported meanings."""
    report_lower = simplified_report.lower()
    unsupported = [
        term.term
        for term in fused_terms
        if not _meaning_is_reflected(term.meaning, report_lower)
    ]
    return ValidationCheck(
        name="meaning_preservation",
        passed=not unsupported,
        score=1.0 if not unsupported else 0.0,
        details=(
            "All term meanings are reflected."
            if not unsupported
            else "Meanings not reflected for: " + ", ".join(unsupported)
        ),
    )


def _check_no_unsupported_numbers(
    fused_terms: list[FusedMedicalTerm],
    simplified_report: str,
) -> ValidationCheck:
    """Reject generated numbers not present in fused facts."""
    allowed_numbers = set(
        re.findall(
            r"\d+(?:\.\d+)?",
            " ".join(
                " ".join([term.term, term.meaning, term.context])
                for term in fused_terms
            ),
        )
    )
    generated_numbers = set(re.findall(r"\d+(?:\.\d+)?", simplified_report))
    unsupported_numbers = sorted(generated_numbers - allowed_numbers)
    return ValidationCheck(
        name="numerical_consistency",
        passed=not unsupported_numbers,
        score=1.0 if not unsupported_numbers else 0.0,
        details=(
            "No unsupported numbers were generated."
            if not unsupported_numbers
            else "Unsupported generated numbers: " + ", ".join(unsupported_numbers)
        ),
    )


def _check_no_unsupported_advice(
    fused_terms: list[FusedMedicalTerm],
    simplified_report: str,
) -> ValidationCheck:
    """Reject generated advice not supported by fused facts."""
    advice_terms = (
        "medication",
        "medicine",
        "lifestyle",
        "exercise",
        "diet",
        "treatment",
        "follow up",
        "follow-up",
        "emergency",
        "urgent",
    )
    source_text = " ".join(
        " ".join([term.term, term.meaning, term.context, term.ambiguity_resolution])
        for term in fused_terms
    ).lower()
    report_lower = simplified_report.lower()
    unsupported = [
        advice
        for advice in advice_terms
        if advice in report_lower and advice not in source_text
    ]
    return ValidationCheck(
        name="unsafe_or_unsupported_advice",
        passed=not unsupported,
        score=1.0 if not unsupported else 0.0,
        details=(
            "No unsupported care advice was generated."
            if not unsupported
            else "Unsupported advice terms generated: " + ", ".join(unsupported)
        ),
    )


def _meaning_is_reflected(meaning: str, report_lower: str) -> bool:
    """Check whether key meaning words are reflected in generated text."""
    meaning_tokens = [
        token
        for token in re.findall(r"[a-zA-Z]+", meaning.lower())
        if len(token) >= 4
    ]
    if not meaning_tokens:
        return True
    required_hits = max(1, min(2, len(meaning_tokens)))
    return sum(1 for token in meaning_tokens if token in report_lower) >= required_hits


def _choose_action(failed_checks: list[ValidationCheck]) -> ValidationAction:
    """Choose whether to approve, reject, or regenerate."""
    if not failed_checks:
        return ValidationAction.APPROVE
    if any(check.name == "unsafe_generation" for check in failed_checks):
        return ValidationAction.REJECT
    if any(check.name == "unsafe_or_unsupported_advice" for check in failed_checks):
        return ValidationAction.REJECT
    return ValidationAction.REGENERATE


def _bounded_float(value: object) -> float:
    """Convert a value to a bounded risk score."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 1.0
    return min(max(parsed, 0.0), 1.0)
