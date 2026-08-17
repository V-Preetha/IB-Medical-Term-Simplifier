from datetime import UTC, datetime

from app.ner.contracts import (
    BaseNERProvider,
    NERHealthStatus,
    NERProviderHealth,
    NERProviderMetadata,
    NERProviderResult,
    NormalizedEntity,
)


class FakeNERProvider(BaseNERProvider):
    def __init__(self, name: str = "openmed-gliner") -> None:
        self.name = name
        self.ready = False

    async def initialize(self) -> None:
        self.ready = True

    def extract(self, text: str) -> NERProviderResult:
        folded = text.casefold()
        if "diabetes" not in folded:
            return NERProviderResult((), len(text.split()))
        start = folded.index("diabetes")
        entity = NormalizedEntity(text[start : start + 8], "Disease", start, start + 8, 0.95)
        return NERProviderResult((entity,), len(text.split()))

    def metadata(self) -> NERProviderMetadata:
        return NERProviderMetadata(
            self.name,
            "synthetic-test-model",
            "a" * 40,
            "test",
            "cpu",
            1.25 if self.ready else None,
            datetime.now(UTC) if self.ready else None,
            {"candidate_only": True, "confidence_threshold": 0.5},
        )

    def health(self) -> NERProviderHealth:
        status = NERHealthStatus.READY if self.ready else NERHealthStatus.NOT_INITIALIZED
        return NERProviderHealth(self.name, status, "Synthetic test provider.", self.metadata())

    async def shutdown(self) -> None:
        self.ready = False
