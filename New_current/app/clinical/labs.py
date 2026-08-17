"""Conservative extraction of explicit laboratory and vital-sign values."""

import re
from collections.abc import Iterable

from app.clinical.models import LabResult


class LabResultExtractor:
    """Extract only complete name/value/unit triples present in source text."""

    DEFAULT_NAMES = (
        "Hemoglobin",
        "HbA1c",
        "ALT",
        "AST",
        "Creatinine",
        "Glucose",
        "Sodium",
        "Potassium",
        "Blood Pressure",
        "BP",
        "Heart Rate",
        "Pulse",
    )
    DEFAULT_UNITS = (
        "mg/dL",
        "g/dL",
        "gm/dL",
        "mmol/L",
        "mEq/L",
        "U/L",
        "IU/L",
        "mmHg",
        "bpm",
        "%",
    )

    def __init__(
        self,
        names: Iterable[str] = DEFAULT_NAMES,
        units: Iterable[str] = DEFAULT_UNITS,
    ) -> None:
        name_pattern = "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))
        unit_pattern = "|".join(re.escape(unit) for unit in sorted(units, key=len, reverse=True))
        self._pattern = re.compile(
            rf"(?<!\w)(?P<name>{name_pattern})(?!\w)"
            rf"\s*(?:[:=\-]\s*)?"
            rf"(?P<value>[<>]?\s*(?:\d+(?:\.\d+)?|\d{{2,3}}/\d{{2,3}}))"
            rf"\s*(?P<unit>{unit_pattern})(?!\w)",
            re.IGNORECASE,
        )

    def extract(self, text: str) -> tuple[LabResult, ...]:
        results: list[LabResult] = []
        seen: set[tuple[str, str, str]] = set()
        for match in self._pattern.finditer(text):
            result = LabResult(
                name=match.group("name"),
                value=re.sub(r"\s+", "", match.group("value")),
                unit=match.group("unit"),
            )
            key = (result.name.casefold(), result.value, result.unit.casefold())
            if key not in seen:
                seen.add(key)
                results.append(result)
        return tuple(results)
