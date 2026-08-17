"""JSON logging and bounded privacy-safe log inspection."""

import json
import logging
from collections import deque
from datetime import UTC, datetime
from threading import RLock
from typing import Any

_STANDARD_FIELDS = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class JSONLogFormatter(logging.Formatter):
    """Serialize standard and structured LogRecord fields as one JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload = _record_payload(record)
        if record.exc_info:
            payload["exception_type"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))


class RecentLogStore(logging.Handler):
    """Bounded in-process engineering view; never stores clinical content fields."""

    def __init__(self, max_entries: int) -> None:
        super().__init__()
        self._records: deque[dict[str, Any]] = deque(maxlen=max_entries)
        self._lock = RLock()

    def emit(self, record: logging.LogRecord) -> None:
        with self._lock:
            self._records.append(_record_payload(record))

    def recent(self, limit: int = 100) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(list(self._records)[-limit:])


def configure_structured_logging(max_entries: int) -> RecentLogStore:
    """Configure production JSON output and return the engineering log adapter."""

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in tuple(root.handlers):
        root.removeHandler(handler)
    stream = logging.StreamHandler()
    stream.setFormatter(JSONLogFormatter())
    recent = RecentLogStore(max_entries)
    root.addHandler(stream)
    root.addHandler(recent)
    return recent


def _record_payload(record: logging.LogRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
        "level": record.levelname,
        "logger": record.name,
        "message": record.getMessage(),
    }
    for key, value in record.__dict__.items():
        if key not in _STANDARD_FIELDS and _is_safe_value(value):
            payload[key] = value
    return payload


def _is_safe_value(value: Any) -> bool:
    try:
        json.dumps(value, default=str)
    except (TypeError, ValueError):
        return False
    return True
