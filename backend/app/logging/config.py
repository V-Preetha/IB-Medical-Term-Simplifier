"""Centralized logging setup for the backend service."""

import logging
import logging.config
from typing import Any

from app.config.settings import Settings


def configure_logging(settings: Settings) -> None:
    """Configure process-wide structured console logging."""
    logging_config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": (
                    "%(asctime)s | %(levelname)s | %(name)s | "
                    "%(message)s"
                )
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": settings.log_level.upper(),
            }
        },
        "root": {
            "handlers": ["console"],
            "level": settings.log_level.upper(),
        },
        "loggers": {
            "uvicorn.access": {
                "handlers": ["console"],
                "level": settings.log_level.upper(),
                "propagate": False,
            }
        },
    }
    logging.config.dictConfig(logging_config)

