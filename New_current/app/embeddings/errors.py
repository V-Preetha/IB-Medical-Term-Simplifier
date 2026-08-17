"""Safe typed errors for medical embeddings."""


class EmbeddingError(Exception):
    code = "embedding_error"
    status_code = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class EmbeddingConfigurationError(EmbeddingError):
    code = "embedding_configuration_error"
    status_code = 503


class EmbeddingProviderUnavailableError(EmbeddingError):
    code = "embedding_provider_unavailable"
    status_code = 503


class EmbeddingInferenceError(EmbeddingError):
    code = "embedding_inference_error"
    status_code = 500


class UnsupportedEmbeddingProviderError(EmbeddingError):
    code = "unsupported_embedding_provider"
    status_code = 422
