import logging
from typing import Dict

import textstat


LOGGER = logging.getLogger(__name__)


def readability_scores(text: str) -> Dict[str, float]:
    try:
        return {
            "flesch_reading_ease": float(textstat.flesch_reading_ease(text)),
            "flesch_kincaid_grade": float(textstat.flesch_kincaid_grade(text)),
            "smog": float(textstat.smog_index(text)),
            "gunning_fog": float(textstat.gunning_fog(text)),
        }
    except Exception as exc:
        LOGGER.exception("Readability scoring failed: %s", exc)
        return {
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "smog": 0.0,
            "gunning_fog": 0.0,
        }


def fk_in_target_band(level: str, fk_grade: float) -> bool:
    level = level.lower()
    if level == "beginner":
        return fk_grade <= 6
    if level == "intermediate":
        return 7 <= fk_grade <= 10
    if level == "advanced":
        return 11 <= fk_grade <= 14
    return False
