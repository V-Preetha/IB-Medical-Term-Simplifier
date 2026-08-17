"""Strict validation for LLM responses and final SFT samples."""

from __future__ import annotations

import json
from typing import Any

from config import SYSTEM_MESSAGE

VALID_ENTITY_TYPES = frozenset(
    {
        "Disease",
        "Symptom",
        "Medication",
        "Procedure",
        "Anatomy",
        "Laboratory Test",
        "Imaging Finding",
        "Clinical Measurement",
        "Medical Device",
        "Other",
    }
)


class ValidationError(ValueError):
    """Raised when a response does not match the required schema."""


def _reject_nonstandard_number(value: str) -> None:
    """Reject NaN and Infinity, which are not valid JSON values."""
    raise ValidationError(f"Non-standard JSON number is not allowed: {value}")


def parse_and_validate_response(raw_response: str) -> dict[str, Any]:
    """Parse an LLM response and validate its assistant-content schema."""
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise ValidationError("The LLM returned an empty response.")

    try:
        parsed = json.loads(
            raw_response,
            parse_constant=_reject_nonstandard_number,
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValidationError(f"The LLM response is not valid JSON: {exc}") from exc

    validate_assistant_content(parsed)
    return parsed


def _nonempty_string(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"'{field}' must be a non-empty string.")


def validate_assistant_content(content: Any) -> None:
    """Validate the JSON object produced by the LLM."""
    if not isinstance(content, dict):
        raise ValidationError("Assistant content must be a JSON object.")

    required = {"summary", "simplification", "entities"}
    missing = required.difference(content)
    if missing:
        raise ValidationError(
            f"Assistant content is missing fields: {sorted(missing)}"
        )
    unexpected = set(content).difference(required)
    if unexpected:
        raise ValidationError(
            f"Assistant content has unexpected fields: {sorted(unexpected)}"
        )

    _nonempty_string(content["summary"], "summary")

    simplification = content["simplification"]
    if not isinstance(simplification, dict):
        raise ValidationError("'simplification' must be a JSON object.")
    unexpected_levels = set(simplification).difference(
        {"clinical", "general", "child"}
    )
    if unexpected_levels:
        raise ValidationError(
            "'simplification' has unexpected fields: "
            f"{sorted(unexpected_levels)}"
        )
    for level in ("clinical", "general", "child"):
        if level not in simplification:
            raise ValidationError(
                f"'simplification' is missing the '{level}' level."
            )
        _nonempty_string(simplification[level], f"simplification.{level}")

    entities = content["entities"]
    if not isinstance(entities, list):
        raise ValidationError("'entities' must be a JSON array.")
    for index, entity in enumerate(entities):
        if not isinstance(entity, dict):
            raise ValidationError(f"Entity {index} must be a JSON object.")
        unexpected_entity_fields = set(entity).difference(
            {"term", "type", "meaning"}
        )
        if unexpected_entity_fields:
            raise ValidationError(
                f"Entity {index} has unexpected fields: "
                f"{sorted(unexpected_entity_fields)}"
            )
        for field in ("term", "type", "meaning"):
            if field not in entity:
                raise ValidationError(
                    f"Entity {index} is missing the '{field}' field."
                )
            _nonempty_string(entity[field], f"entities[{index}].{field}")
        if entity["type"] not in VALID_ENTITY_TYPES:
            allowed = ", ".join(sorted(VALID_ENTITY_TYPES))
            raise ValidationError(
                f"Entity {index} has invalid type {entity['type']!r}. "
                f"Allowed types: {allowed}."
            )


def build_and_validate_sample(
    report: str,
    assistant_content: dict[str, Any],
    instruction: str | None = None,
) -> dict[str, Any]:
    """Build the requested messages object and validate it before writing."""
    _nonempty_string(report, "report")
    validate_assistant_content(assistant_content)
    if instruction is None:
        user_content = report
    else:
        _nonempty_string(instruction, "instruction")
        user_content = f"{instruction}\n\nMedical report:\n{report}"

    sample = {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }
    validate_sample(sample)
    return sample


def validate_sample(sample: Any) -> None:
    """Validate the complete JSONL training sample."""
    if not isinstance(sample, dict) or not isinstance(
        sample.get("messages"), list
    ):
        raise ValidationError("Sample must contain a 'messages' array.")

    messages = sample["messages"]
    if len(messages) != 3:
        raise ValidationError("Sample must contain exactly three messages.")

    expected_roles = ("system", "user", "assistant")
    for index, role in enumerate(expected_roles):
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") != role:
            raise ValidationError(f"Message {index} must have role '{role}'.")
        if "content" not in message:
            raise ValidationError(f"Message {index} has no content.")

    if messages[0]["content"] != SYSTEM_MESSAGE:
        raise ValidationError("The system message does not match configuration.")
    _nonempty_string(messages[1]["content"], "messages[1].content")
    validate_assistant_content(messages[2]["content"])
