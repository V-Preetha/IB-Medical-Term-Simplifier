"""Evaluation metrics for simplified medical reports."""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from app.config.settings import Settings
from app.fusion.medical_fusion import FusedMedicalTerm
from app.services.document_parsing import clean_extracted_text

logger = logging.getLogger(__name__)


class EvaluationError(RuntimeError):
    """Raised when report evaluation cannot be completed."""


@dataclass(frozen=True)
class BERTScoreMetrics:
    """BERTScore precision, recall, and F1 metrics."""

    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class ReadabilityMetrics:
    """Readability metrics for a generated simplification."""

    flesch_kincaid_grade_level: float
    flesch_reading_ease: float


@dataclass(frozen=True)
class MedicalConsistencyMetrics:
    """Medical consistency checks against fused structured facts."""

    score: float
    terms_preserved: bool
    meanings_preserved: bool
    unsupported_numbers: list[str]
    missing_terms: list[str]
    missing_meanings: list[str]


@dataclass(frozen=True)
class EvaluationResult:
    """Full quality evaluation result."""

    bertscore: BERTScoreMetrics
    semantic_similarity: float
    readability: ReadabilityMetrics
    medical_consistency: MedicalConsistencyMetrics
    warnings: list[str] = field(default_factory=list)


class BERTScoreBackend(Protocol):
    """Protocol for BERTScore backends."""

    def score(self, *, reference: str, candidate: str) -> BERTScoreMetrics:
        """Return BERTScore metrics."""


class SemanticSimilarityBackend(Protocol):
    """Protocol for semantic similarity backends."""

    def similarity(self, *, reference: str, candidate: str) -> float:
        """Return cosine semantic similarity."""


class HuggingFaceBERTScoreBackend:
    """BERTScore backend using the bert-score package."""

    def __init__(self, settings: Settings) -> None:
        """Initialize backend with configured model type."""
        self._model_type = settings.bertscore_model_type

    def score(self, *, reference: str, candidate: str) -> BERTScoreMetrics:
        """Compute BERTScore for one reference/candidate pair."""
        _ensure_matplotlib_cache_dir()
        try:
            from bert_score import score
        except ImportError as exc:
            raise EvaluationError(
                "bert-score is not installed. Install backend requirements before evaluation."
            ) from exc

        precision, recall, f1 = score(
            [candidate],
            [reference],
            model_type=self._model_type,
            lang="en",
            verbose=False,
        )
        return BERTScoreMetrics(
            precision=round(float(precision[0]), 4),
            recall=round(float(recall[0]), 4),
            f1=round(float(f1[0]), 4),
        )


class SentenceTransformerSimilarityBackend:
    """Cosine semantic similarity backend using SentenceTransformers."""

    def __init__(self, settings: Settings) -> None:
        """Initialize lazy-loading sentence embedding backend."""
        self._model_name = settings.semantic_similarity_model_name
        self._model = None

    def similarity(self, *, reference: str, candidate: str) -> float:
        """Compute cosine similarity between reference and candidate embeddings."""
        model = self._load_model()
        embeddings = model.encode([reference, candidate], normalize_embeddings=True)
        return round(float(_cosine_similarity(embeddings[0], embeddings[1])), 4)

    def _load_model(self) -> object:
        """Load the configured sentence-transformer model once."""
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EvaluationError(
                "sentence-transformers is not installed. Install backend requirements before evaluation."
            ) from exc
        self._model = SentenceTransformer(self._model_name)
        return self._model


class EvaluationService:
    """Evaluate simplified reports with semantic, readability, and medical metrics."""

    def __init__(
        self,
        settings: Settings,
        bertscore_backend: BERTScoreBackend | None = None,
        semantic_backend: SemanticSimilarityBackend | None = None,
    ) -> None:
        """Initialize evaluation with injectable metric backends."""
        self._bertscore_backend = bertscore_backend or HuggingFaceBERTScoreBackend(settings)
        self._semantic_backend = semantic_backend or SentenceTransformerSimilarityBackend(
            settings
        )

    def evaluate(
        self,
        *,
        reference_text: str,
        simplified_report: str,
        fused_terms: list[FusedMedicalTerm],
    ) -> EvaluationResult:
        """Compute full evaluation metrics for a simplified report."""
        reference = clean_extracted_text(reference_text)
        candidate = clean_extracted_text(simplified_report)
        if not reference:
            raise EvaluationError("Reference text is required for evaluation.")
        if not candidate:
            raise EvaluationError("Simplified report is required for evaluation.")
        if not fused_terms:
            raise EvaluationError("At least one fused medical term is required.")

        bertscore = self._bertscore_backend.score(
            reference=reference,
            candidate=candidate,
        )
        semantic_similarity = self._semantic_backend.similarity(
            reference=reference,
            candidate=candidate,
        )
        readability = _compute_readability(candidate)
        medical_consistency = _compute_medical_consistency(
            fused_terms=fused_terms,
            simplified_report=candidate,
        )
        warnings = []
        if medical_consistency.score < 1.0:
            warnings.append("Medical consistency checks found missing or unsupported content.")
        if readability.flesch_kincaid_grade_level > 8.5:
            warnings.append("Simplified report is above the target 8th-grade reading level.")

        logger.info("Computed report evaluation metrics")
        return EvaluationResult(
            bertscore=bertscore,
            semantic_similarity=semantic_similarity,
            readability=readability,
            medical_consistency=medical_consistency,
            warnings=warnings,
        )


def _compute_readability(text: str) -> ReadabilityMetrics:
    """Compute FKGL and Flesch Reading Ease."""
    sentences = max(len(re.findall(r"[.!?]+", text)), 1)
    words = re.findall(r"[A-Za-z0-9]+", text)
    word_count = max(len(words), 1)
    syllables = max(sum(_count_syllables(word) for word in words), 1)

    words_per_sentence = word_count / sentences
    syllables_per_word = syllables / word_count
    fkgl = (0.39 * words_per_sentence) + (11.8 * syllables_per_word) - 15.59
    fre = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
    return ReadabilityMetrics(
        flesch_kincaid_grade_level=round(fkgl, 4),
        flesch_reading_ease=round(fre, 4),
    )


def _compute_medical_consistency(
    *,
    fused_terms: list[FusedMedicalTerm],
    simplified_report: str,
) -> MedicalConsistencyMetrics:
    """Evaluate preservation of medical facts in generated output."""
    report_lower = simplified_report.lower()
    missing_terms = [
        term.term
        for term in fused_terms
        if term.term.lower() not in report_lower
    ]
    missing_meanings = [
        term.term
        for term in fused_terms
        if not _meaning_is_reflected(term.meaning, report_lower)
    ]
    unsupported_numbers = _unsupported_numbers(fused_terms, simplified_report)

    passed_checks = 3
    if missing_terms:
        passed_checks -= 1
    if missing_meanings:
        passed_checks -= 1
    if unsupported_numbers:
        passed_checks -= 1

    return MedicalConsistencyMetrics(
        score=round(max(passed_checks, 0) / 3, 4),
        terms_preserved=not missing_terms,
        meanings_preserved=not missing_meanings,
        unsupported_numbers=unsupported_numbers,
        missing_terms=missing_terms,
        missing_meanings=missing_meanings,
    )


def _unsupported_numbers(
    fused_terms: list[FusedMedicalTerm],
    simplified_report: str,
) -> list[str]:
    """Return generated numbers that do not appear in source facts."""
    source_text = " ".join(
        " ".join([term.term, term.meaning, term.context])
        for term in fused_terms
    )
    allowed_numbers = set(re.findall(r"\d+(?:\.\d+)?", source_text))
    generated_numbers = set(re.findall(r"\d+(?:\.\d+)?", simplified_report))
    return sorted(generated_numbers - allowed_numbers)


def _meaning_is_reflected(meaning: str, report_lower: str) -> bool:
    """Check whether key words from a term meaning appear in the report."""
    tokens = [
        token
        for token in re.findall(r"[a-zA-Z]+", meaning.lower())
        if len(token) >= 4
    ]
    if not tokens:
        return True
    required_hits = max(1, min(2, len(tokens)))
    return sum(1 for token in tokens if token in report_lower) >= required_hits


def _count_syllables(word: str) -> int:
    """Approximate syllable count for readability formulas."""
    normalized = word.lower()
    groups = re.findall(r"[aeiouy]+", normalized)
    count = len(groups)
    if normalized.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def _cosine_similarity(left: object, right: object) -> float:
    """Return cosine similarity for array-like vectors."""
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    if not left_values or not right_values or len(left_values) != len(right_values):
        return 0.0
    numerator = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(a * a for a in left_values))
    right_norm = math.sqrt(sum(b * b for b in right_values))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _ensure_matplotlib_cache_dir() -> None:
    """Keep Matplotlib cache writes inside the backend workspace by default."""
    if "MPLCONFIGDIR" in os.environ:
        return
    cache_dir = Path(".cache") / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir.resolve())
