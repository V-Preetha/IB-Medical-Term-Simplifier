"""Verification composition root."""

from app.verification.provider import PubMedBERTMedNLIProvider
from app.verification.service import MedicalVerificationService


def create_verification_service() -> MedicalVerificationService:
    return MedicalVerificationService(PubMedBERTMedNLIProvider())
