"""ASGI entrypoint for running the backend with uvicorn."""

from app.main import app

__all__ = ["app"]

