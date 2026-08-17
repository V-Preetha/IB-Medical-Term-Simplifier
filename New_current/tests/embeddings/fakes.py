from datetime import UTC, datetime

from app.embeddings.contracts import (
    BaseEmbeddingProvider,
    EmbeddingHealthStatus,
    EmbeddingInput,
    EmbeddingProviderHealth,
    EmbeddingProviderMetadata,
    EmbeddingVector,
    ProviderEmbeddingResult,
)


class FakeEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self) -> None:
        self.ready = False

    async def initialize(self) -> None:
        self.ready = True

    def embed_batch(self, inputs: tuple[EmbeddingInput, ...]) -> ProviderEmbeddingResult:
        return ProviderEmbeddingResult(
            tuple(EmbeddingVector(item.input_id, (0.6, 0.8), 5, 1.0) for item in inputs)
        )

    def metadata(self) -> EmbeddingProviderMetadata:
        return EmbeddingProviderMetadata(
            provider_name="bioclinical-modernbert",
            provider_version="test-transformers",
            model_name="approved/test-biomedical-modernbert",
            model_revision="a" * 40,
            framework="transformers-feature-extraction",
            device="cpu",
            dimensions=2,
            pooling_method="attention-mask-mean-v1",
            normalized=True,
            startup_timestamp=datetime.now(UTC) if self.ready else None,
            loading_time_ms=1.0 if self.ready else None,
            configuration={"local_files_only": True, "batch_size": 16},
        )

    def health(self) -> EmbeddingProviderHealth:
        return EmbeddingProviderHealth(
            EmbeddingHealthStatus.READY if self.ready else EmbeddingHealthStatus.NOT_INITIALIZED,
            "Synthetic embedding provider.",
            self.metadata(),
        )

    async def shutdown(self) -> None:
        self.ready = False
