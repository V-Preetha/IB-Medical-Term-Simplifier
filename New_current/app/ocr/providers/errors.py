"""Typed failures raised by OCR provider infrastructure."""


class ProviderError(Exception):
    """Base provider failure carrying a stable machine-readable code."""

    code = "provider_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ProviderInitializationError(ProviderError):
    code = "provider_initialization_failed"


class ProviderConfigurationError(ProviderError):
    code = "provider_configuration_invalid"


class ProviderUnavailableError(ProviderError):
    code = "provider_unavailable"


class InferenceError(ProviderError):
    code = "provider_inference_failed"


class UnsupportedDocumentError(ProviderError):
    code = "unsupported_document"
