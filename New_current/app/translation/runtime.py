"""Translation composition root."""

from app.translation.provider import IndicTrans2Provider
from app.translation.service import TranslationService


def create_translation_service() -> TranslationService:
    return TranslationService(IndicTrans2Provider())
