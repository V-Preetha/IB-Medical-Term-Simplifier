"""Safe errors for the medical NER boundary."""


class NERError(Exception):
    """Base error safe to expose through the production NER API."""

    code = "ner_error"
    status_code = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class NERConfigurationError(NERError):
    code = "ner_configuration_error"
    status_code = 503


class NERProviderUnavailableError(NERError):
    code = "ner_provider_unavailable"
    status_code = 503


class NERInferenceError(NERError):
    code = "ner_inference_error"
    status_code = 500


class UnsupportedNERModelError(NERError):
    code = "unsupported_ner_model"
    status_code = 422
