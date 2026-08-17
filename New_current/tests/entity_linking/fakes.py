from datetime import UTC, datetime

from app.entity_linking.contracts import (
    BaseEntityLinkingProvider,
    ConceptCandidate,
    EntityLink,
    LinkerHealth,
    LinkerHealthStatus,
    LinkerMetadata,
    LinkStatus,
    ProviderLinkResult,
    SourceEntity,
)


class FakeEntityLinkingProvider(BaseEntityLinkingProvider):
    def __init__(self) -> None:
        self.ready = False

    async def initialize(self) -> None:
        self.ready = True

    def link(self, entities: tuple[SourceEntity, ...]) -> ProviderLinkResult:
        links = []
        for entity in entities:
            concept = ConceptCandidate("C0011849", "Diabetes Mellitus", ("T047",), 0.94, "UMLS")
            links.append(EntityLink(entity, LinkStatus.LINKED, concept, (concept,), False))
        return ProviderLinkResult(tuple(links))

    def metadata(self) -> LinkerMetadata:
        return LinkerMetadata(
            "scispacy-umls",
            "0.5.4",
            "synthetic-scispacy-model",
            "1.0.0",
            "UMLS",
            "2025AA",
            "scispacy_candidate_similarity",
            "uncalibrated-scispacy-umls-v1",
            datetime.now(UTC) if self.ready else None,
            1.2 if self.ready else None,
            {"local_files_only": True},
        )

    def health(self) -> LinkerHealth:
        return LinkerHealth(
            LinkerHealthStatus.READY if self.ready else LinkerHealthStatus.NOT_INITIALIZED,
            "Synthetic linker.",
            self.metadata(),
        )

    async def shutdown(self) -> None:
        self.ready = False
