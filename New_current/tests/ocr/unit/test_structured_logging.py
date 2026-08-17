"""Privacy-safe JSON logging tests."""

import json
import logging

from app.ocr.observability.logging import JSONLogFormatter, RecentLogStore


def test_json_formatter_emits_structured_fields() -> None:
    record = logging.LogRecord(
        name="app.ocr.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="stage complete",
        args=(),
        exc_info=None,
    )
    record.request_id = "synthetic-request"
    record.pipeline_stage = "ocr"
    record.processing_time_ms = 12.5
    payload = json.loads(JSONLogFormatter().format(record))
    assert payload["request_id"] == "synthetic-request"
    assert payload["pipeline_stage"] == "ocr"
    assert payload["processing_time_ms"] == 12.5


def test_recent_log_store_is_bounded() -> None:
    store = RecentLogStore(max_entries=2)
    logger = logging.getLogger("app.ocr.log-store-test")
    logger.addHandler(store)
    logger.propagate = False
    try:
        logger.warning("one")
        logger.warning("two")
        logger.warning("three")
    finally:
        logger.removeHandler(store)
    assert [item["message"] for item in store.recent()] == ["two", "three"]
