"""Simplification composition root."""

from app.simplification.provider import Qwen3SimplificationProvider
from app.simplification.service import SimplificationService


def create_simplification_service() -> SimplificationService:
    return SimplificationService(Qwen3SimplificationProvider())
