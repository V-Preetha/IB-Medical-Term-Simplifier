"""Safe typed errors for relation extraction."""


class RelationExtractionError(Exception):
    code = "relation_extraction_error"
    status_code = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class RelationConfigurationError(RelationExtractionError):
    code = "relation_configuration_error"
    status_code = 503


class RelationProviderUnavailableError(RelationExtractionError):
    code = "relation_provider_unavailable"
    status_code = 503


class RelationInferenceError(RelationExtractionError):
    code = "relation_inference_error"
    status_code = 500


class UnsupportedRelationProviderError(RelationExtractionError):
    code = "unsupported_relation_provider"
    status_code = 422
