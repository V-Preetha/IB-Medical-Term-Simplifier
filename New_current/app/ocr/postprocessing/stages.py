"""Independent, measurable medical OCR post-processing stages."""

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from app.ocr.providers.errors import ProviderConfigurationError

_WORD_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z'-]*\b")


@dataclass(frozen=True, slots=True)
class StageResult:
    text: str
    latency_ms: float
    corrections: int
    configuration: dict[str, Any]


class RegexNormalizationStage:
    """Conservative OCR cleanup that preserves line order and clinical values."""

    def __init__(self, *, rule_version: str) -> None:
        if not rule_version.strip():
            raise ProviderConfigurationError("regex rule_version must not be blank.")
        self.rule_version = rule_version
        self._rules = (
            (re.compile(r"\r\n?|\u2028|\u2029"), "\n"),
            (re.compile(r"[\t\v\f ]+"), " "),
            (re.compile(r" +\n"), "\n"),
            (re.compile(r"\n{3,}"), "\n\n"),
            (re.compile(r"\b(\d+)\s*\.\s*(\d+)\b"), r"\1.\2"),
            (re.compile(r"\b(mg|g|mcg|µg|mmol|mol)\s*/\s*(dL|L|mL|kg|day)\b", re.I), r"\1/\2"),
            (re.compile(r"\b(\d+(?:\.\d+)?)\s+%"), r"\1%"),
            (re.compile(r"\s+([,;:])"), r"\1"),
        )

    def process(self, text: str) -> StageResult:
        started = perf_counter()
        normalized = unicodedata.normalize("NFKC", text)
        corrections = int(normalized != text)
        for pattern, replacement in self._rules:
            normalized, count = pattern.subn(replacement, normalized)
            corrections += count
        return StageResult(
            text=normalized.strip(),
            latency_ms=round((perf_counter() - started) * 1000, 3),
            corrections=corrections,
            configuration={"rule_version": self.rule_version},
        )


class MedicalAbbreviationStage:
    """Apply an approved abbreviation map and expose protected vocabulary."""

    def __init__(self, *, dictionary_path: str, dictionary_version: str) -> None:
        self.dictionary_path = Path(dictionary_path).expanduser().resolve()
        self.dictionary_version = dictionary_version.strip()
        if not self.dictionary_version:
            raise ProviderConfigurationError(
                "medical abbreviation dictionary_version must not be blank."
            )
        payload = _read_json_object(self.dictionary_path, "medical abbreviation dictionary")
        raw_mapping = payload.get("abbreviations", payload)
        if not isinstance(raw_mapping, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in raw_mapping.items()
            if key != "protected_tokens"
        ):
            raise ProviderConfigurationError(
                "The medical abbreviation dictionary must contain string mappings."
            )
        self.mapping = {
            key: value
            for key, value in raw_mapping.items()
            if key != "protected_tokens" and key.strip() and value.strip()
        }
        protected = payload.get("protected_tokens", [])
        if not isinstance(protected, list) or not all(
            isinstance(value, str) for value in protected
        ):
            raise ProviderConfigurationError("protected_tokens must be a string list.")
        self.protected_terms = frozenset(
            value.casefold() for value in (*self.mapping.keys(), *protected) if value
        )
        self._pattern = (
            re.compile(
                r"(?<![A-Za-z0-9])("
                + "|".join(re.escape(key) for key in sorted(self.mapping, key=len, reverse=True))
                + r")(?![A-Za-z0-9])",
                re.IGNORECASE,
            )
            if self.mapping
            else None
        )
        self._mapping_casefold = {key.casefold(): value for key, value in self.mapping.items()}

    def process(self, text: str) -> StageResult:
        started = perf_counter()
        corrections = 0
        if self._pattern is None:
            normalized = text
        else:

            def replace(match: re.Match[str]) -> str:
                nonlocal corrections
                replacement = self._mapping_casefold[match.group(0).casefold()]
                replacement_folded = replacement.casefold()
                matched_folded = match.group(0).casefold()
                embedded_at = replacement_folded.find(matched_folded)
                if embedded_at >= 0:
                    prefix = replacement[:embedded_at]
                    suffix = replacement[embedded_at + len(match.group(0)) :]
                    existing_prefix = text[max(0, match.start() - len(prefix)) : match.start()]
                    existing_suffix = text[match.end() : match.end() + len(suffix)]
                    if (
                        existing_prefix.casefold() == prefix.casefold()
                        and existing_suffix.casefold() == suffix.casefold()
                    ):
                        return match.group(0)
                if replacement == match.group(0):
                    return match.group(0)
                corrections += 1
                return replacement

            normalized = self._pattern.sub(replace, text)
        return StageResult(
            text=normalized,
            latency_ms=round((perf_counter() - started) * 1000, 3),
            corrections=corrections,
            configuration={
                "dictionary_path": str(self.dictionary_path),
                "dictionary_version": self.dictionary_version,
                "entry_count": len(self.mapping),
            },
        )


class SymSpellStage:
    """Bounded SymSpell delete-index correction with clinical token protection."""

    def __init__(
        self,
        *,
        dictionary_path: str,
        dictionary_version: str,
        max_edit_distance: int,
        prefix_length: int,
        min_word_length: int,
        min_frequency: int,
        protected_terms: frozenset[str],
        protected_terms_path: str,
    ) -> None:
        if not 1 <= max_edit_distance <= 3:
            raise ProviderConfigurationError("max_edit_distance must be between 1 and 3.")
        if prefix_length <= max_edit_distance:
            raise ProviderConfigurationError("prefix_length must exceed max_edit_distance.")
        if min_word_length < 2 or min_frequency < 1:
            raise ProviderConfigurationError(
                "min_word_length must be at least 2 and min_frequency must be positive."
            )
        self.dictionary_path = Path(dictionary_path).expanduser().resolve()
        self.dictionary_version = dictionary_version.strip()
        if not self.dictionary_version:
            raise ProviderConfigurationError("SymSpell dictionary_version must not be blank.")
        self.max_edit_distance = max_edit_distance
        self.prefix_length = prefix_length
        self.min_word_length = min_word_length
        self.min_frequency = min_frequency
        self.frequencies = _read_frequency_dictionary(self.dictionary_path)
        configured_protected = _read_protected_terms(
            Path(protected_terms_path).expanduser().resolve()
        )
        self.protected_terms = frozenset((*protected_terms, *configured_protected))
        self.deletes: dict[str, set[str]] = {}
        for term in self.frequencies:
            for deletion in _generate_deletes(
                term[: self.prefix_length],
                self.max_edit_distance,
            ):
                self.deletes.setdefault(deletion, set()).add(term)

    def process(self, text: str) -> StageResult:
        started = perf_counter()
        corrections = 0

        def replace(match: re.Match[str]) -> str:
            nonlocal corrections
            token = match.group(0)
            normalized = token.casefold()
            if (
                len(normalized) < self.min_word_length
                or not token.islower()
                or normalized in self.protected_terms
                or normalized in self.frequencies
            ):
                return token
            suggestion = self._lookup(normalized)
            if suggestion is None or suggestion == normalized:
                return token
            corrections += 1
            return suggestion

        normalized_text = _WORD_PATTERN.sub(replace, text)
        return StageResult(
            text=normalized_text,
            latency_ms=round((perf_counter() - started) * 1000, 3),
            corrections=corrections,
            configuration={
                "dictionary_path": str(self.dictionary_path),
                "dictionary_version": self.dictionary_version,
                "dictionary_terms": len(self.frequencies),
                "max_edit_distance": self.max_edit_distance,
                "prefix_length": self.prefix_length,
                "min_word_length": self.min_word_length,
                "min_frequency": self.min_frequency,
            },
        )

    def _lookup(self, word: str) -> str | None:
        candidates: set[str] = set()
        prefix = word[: self.prefix_length]
        for deletion in _generate_deletes(prefix, self.max_edit_distance):
            candidates.update(self.deletes.get(deletion, ()))
        ranked = []
        for candidate in candidates:
            frequency = self.frequencies[candidate]
            if frequency < self.min_frequency:
                continue
            distance = _damerau_osa_distance(word, candidate, self.max_edit_distance)
            if distance <= self.max_edit_distance:
                ranked.append((distance, -frequency, candidate))
        return min(ranked)[2] if ranked else None


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProviderConfigurationError(f"The configured {label} cannot be loaded.") from exc
    if not isinstance(payload, dict):
        raise ProviderConfigurationError(f"The configured {label} must contain an object.")
    return payload


def _read_frequency_dictionary(path: Path) -> dict[str, int]:
    frequencies: dict[str, int] = {}
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                parts = line.strip().rsplit(maxsplit=1)
                if not parts:
                    continue
                if len(parts) != 2:
                    raise ProviderConfigurationError(
                        f"Invalid SymSpell dictionary row at line {line_number}."
                    )
                term, raw_frequency = parts
                try:
                    frequency = int(raw_frequency)
                except ValueError as exc:
                    raise ProviderConfigurationError(
                        f"Invalid SymSpell frequency at line {line_number}."
                    ) from exc
                if term.isalpha() and frequency > 0:
                    frequencies[term.casefold()] = max(
                        frequency,
                        frequencies.get(term.casefold(), 0),
                    )
    except OSError as exc:
        raise ProviderConfigurationError(
            "The configured SymSpell frequency dictionary cannot be loaded."
        ) from exc
    if not frequencies:
        raise ProviderConfigurationError("The SymSpell frequency dictionary is empty.")
    return frequencies


def _read_protected_terms(path: Path) -> frozenset[str]:
    payload = _read_json_object(path, "protected medical terms dictionary")
    terms = payload.get("protected_terms")
    if not isinstance(terms, list) or not all(isinstance(term, str) for term in terms):
        raise ProviderConfigurationError("protected_terms must be a string list.")
    return frozenset(term.casefold() for term in terms if term)


def _generate_deletes(word: str, max_distance: int) -> frozenset[str]:
    deletes = {word}
    frontier = {word}
    for _ in range(max_distance):
        next_frontier = {
            candidate[:index] + candidate[index + 1 :]
            for candidate in frontier
            for index in range(len(candidate))
            if candidate
        }
        next_frontier -= deletes
        if not next_frontier:
            break
        deletes.update(next_frontier)
        frontier = next_frontier
    return frozenset(deletes)


def _damerau_osa_distance(left: str, right: str, maximum: int) -> int:
    if abs(len(left) - len(right)) > maximum:
        return maximum + 1
    previous_previous: list[int] | None = None
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        row_minimum = left_index
        for right_index, right_character in enumerate(right, start=1):
            cost = int(left_character != right_character)
            value = min(
                current[right_index - 1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + cost,
            )
            if (
                previous_previous is not None
                and left_index > 1
                and right_index > 1
                and left_character == right[right_index - 2]
                and left[left_index - 2] == right_character
            ):
                value = min(value, previous_previous[right_index - 2] + 1)
            current.append(value)
            row_minimum = min(row_minimum, value)
        if row_minimum > maximum:
            return maximum + 1
        previous_previous, previous = previous, current
    return previous[-1]
