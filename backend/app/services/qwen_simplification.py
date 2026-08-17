"""Qwen3 Instruct simplification engine for fused medical representations."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from app.config.settings import Settings
from app.fusion.medical_fusion import FusedMedicalTerm
from app.services.document_parsing import clean_extracted_text

logger = logging.getLogger(__name__)


class SimplificationError(RuntimeError):
    """Raised when patient-friendly simplification cannot be generated."""


@dataclass(frozen=True)
class TermExplanation:
    """Patient-friendly explanation for one fused medical term."""

    term: str
    explanation: str
    difficulty: float
    confidence: float


@dataclass(frozen=True)
class SimplificationResult:
    """Generated patient-friendly simplified report."""

    simplified_report: str
    term_explanations: list[TermExplanation]
    model_name: str
    warnings: list[str] = field(default_factory=list)


class TextGenerationBackend(Protocol):
    """Protocol for Qwen3 text generation backends."""

    model_name: str

    def generate(self, prompt: str) -> str:
        """Generate text from a rendered prompt."""


class Qwen3GenerationBackend:
    """HuggingFace Qwen3 Instruct-compatible causal language model backend."""

    def __init__(self, settings: Settings) -> None:
        """Initialize a lazy-loading Qwen3 backend."""
        self.model_name = settings.qwen3_model_name
        self._max_new_tokens = settings.qwen3_max_new_tokens
        self._temperature = settings.qwen3_temperature
        self._tokenizer = None
        self._model = None

    def generate(self, prompt: str) -> str:
        """Generate patient-friendly text with Qwen3."""
        tokenizer, model = self._load_model()
        try:
            import torch
        except ImportError as exc:
            raise SimplificationError(
                "PyTorch is not installed. Install backend requirements before Qwen3 generation."
            ) from exc

        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                rendered_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
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
                do_sample=self._temperature > 0,
                temperature=self._temperature if self._temperature > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][encoded["input_ids"].shape[-1] :]
        decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return _clean_generation(clean_extracted_text(decoded_text))

    def _load_model(self) -> tuple[object, object]:
        """Load the configured Qwen3 tokenizer and model once."""
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise SimplificationError(
                "Transformers is not installed. Install backend requirements before Qwen3 generation."
            ) from exc

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._model.eval()
        except Exception as exc:
            logger.exception("Qwen3 model loading failed")
            raise SimplificationError(
                f"Qwen3 model '{self.model_name}' could not be loaded."
            ) from exc
        return self._tokenizer, self._model


class PromptTemplateLoader:
    """Load prompt templates from files under the backend project."""

    def __init__(self, template_path: str) -> None:
        """Store the template path."""
        self._template_path = Path(template_path)

    def load(self) -> str:
        """Load the Qwen3 simplification prompt template."""
        if not self._template_path.exists():
            raise SimplificationError(
                f"Prompt template not found: {self._template_path}"
            )
        return self._template_path.read_text(encoding="utf-8")


class QwenSimplificationService:
    """Generate patient-friendly explanations from fused medical terms."""

    def __init__(
        self,
        settings: Settings,
        generation_backend: TextGenerationBackend | None = None,
        prompt_loader: PromptTemplateLoader | None = None,
    ) -> None:
        """Initialize simplification with injectable generation and prompts."""
        self._settings = settings
        self._generation_backend = generation_backend or Qwen3GenerationBackend(settings)
        self._prompt_loader = prompt_loader or PromptTemplateLoader(
            settings.qwen3_prompt_template_path
        )

    def simplify(self, fused_terms: list[FusedMedicalTerm]) -> SimplificationResult:
        """Generate a simplified report using only fused structured input."""
        if not fused_terms:
            raise SimplificationError("At least one fused medical term is required.")

        prompt = self._render_prompt(fused_terms)
        generated_text = _ground_patient_output(
            text=self._generation_backend.generate(prompt),
            fused_terms=fused_terms,
        )
        if not generated_text:
            raise SimplificationError("Qwen3 generated an empty simplification.")

        logger.info("Generated simplified report with %d fused term(s)", len(fused_terms))
        return SimplificationResult(
            simplified_report=generated_text,
            term_explanations=[
                TermExplanation(
                    term=term.term,
                    explanation=term.meaning,
                    difficulty=term.difficulty,
                    confidence=term.confidence,
                )
                for term in fused_terms
            ],
            model_name=self._generation_backend.model_name,
        )

    def _render_prompt(self, fused_terms: list[FusedMedicalTerm]) -> str:
        """Render the external Qwen3 prompt template."""
        template = self._prompt_loader.load()
        fused_terms_json = json.dumps(
            [_term_to_prompt_payload(term) for term in fused_terms],
            ensure_ascii=True,
            indent=2,
        )
        return template.format(fused_terms_json=fused_terms_json)


def _term_to_prompt_payload(term: FusedMedicalTerm) -> dict[str, object]:
    """Serialize only clinically relevant fused fields for Qwen3."""
    return {
        "term": term.term,
        "difficulty": term.difficulty,
        "meaning": term.meaning,
        "context": term.context,
        "confidence": term.confidence,
        "entity_type": term.entity_type,
        "section_type": term.section_type,
        "ambiguity_resolution": term.ambiguity_resolution,
        "matched_concept": term.matched_concept,
    }


def _clean_generation(text: str) -> str:
    """Remove Qwen3 reasoning traces and echoed input from patient-facing text."""
    return _remove_echoed_structured_input(_strip_reasoning(text))


def _ground_patient_output(*, text: str, fused_terms: list[FusedMedicalTerm]) -> str:
    """Remove unsupported patient-facing statements introduced by generation."""
    grounded_text = _clean_generation(text)
    grounded_text = _remove_internal_metadata_lines(grounded_text)
    grounded_text = _remove_unsupported_care_advice(
        text=grounded_text,
        allowed_text=" ".join(
            " ".join(
                [
                    term.term,
                    term.meaning,
                    term.context,
                    term.ambiguity_resolution,
                ]
            )
            for term in fused_terms
        ),
    )
    return clean_extracted_text(grounded_text)


def _strip_reasoning(text: str) -> str:
    """Remove Qwen3 thinking traces from generated text."""
    stripped_text = re.sub(r"(?is)<think>.*?</think>", "", text)
    if stripped_text != text:
        return clean_extracted_text(stripped_text)
    if text.lower().startswith("<think>"):
        marker = "Simplified Report:"
        marker_index = text.find(marker)
        if marker_index >= 0:
            return clean_extracted_text(text[marker_index:])
    return clean_extracted_text(text)


def _remove_echoed_structured_input(text: str) -> str:
    """Remove echoed structured input if a model repeats the prompt payload."""
    marker = "Structured fused medical representation:"
    marker_index = text.find(marker)
    if marker_index < 0:
        return clean_extracted_text(text)
    return clean_extracted_text(text[:marker_index])


def _remove_internal_metadata_lines(text: str) -> str:
    """Remove model/pipeline metadata that should not be patient-facing."""
    blocked_prefixes = (
        "- confidence:",
        "- ambiguity resolution:",
        "- matched concept:",
        "- modernbert",
        "- semantic",
    )
    kept_lines = [
        line
        for line in text.splitlines()
        if not line.strip().lower().startswith(blocked_prefixes)
    ]
    return clean_extracted_text("\n".join(kept_lines))


def _remove_unsupported_care_advice(*, text: str, allowed_text: str) -> str:
    """Remove unsupported treatment or lifestyle advice not present in fused input."""
    allowed_lower = allowed_text.lower()
    advice_markers = (
        "managed with",
        "medication",
        "lifestyle",
        "exercise",
        "diet",
        "treatment",
        "follow up",
        "follow-up",
    )
    kept_lines: list[str] = []
    for line in text.splitlines():
        sentences = re.split(r"(?<=[.!?])\s+", line)
        kept_sentences: list[str] = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            has_advice = any(marker in sentence_lower for marker in advice_markers)
            if has_advice and not any(marker in allowed_lower for marker in advice_markers):
                continue
            kept_sentences.append(sentence)
        kept_lines.append(" ".join(kept_sentences))
    return clean_extracted_text("\n".join(kept_lines))
