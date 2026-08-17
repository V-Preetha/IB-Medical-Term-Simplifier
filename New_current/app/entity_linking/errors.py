"""Safe errors for the entity-linking boundary."""


class EntityLinkingError(Exception):
    code = "entity_linking_error"
    status_code = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class EntityLinkingConfigurationError(EntityLinkingError):
    code = "entity_linking_configuration_error"
    status_code = 503


class EntityLinkingUnavailableError(EntityLinkingError):
    code = "entity_linking_unavailable"
    status_code = 503


class EntityLinkingInferenceError(EntityLinkingError):
    code = "entity_linking_inference_error"
    status_code = 500


class UnsupportedEntityLinkingProviderError(EntityLinkingError):
    code = "unsupported_entity_linking_provider"
    status_code = 422
