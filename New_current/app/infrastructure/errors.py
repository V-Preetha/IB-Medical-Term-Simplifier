"""Typed, client-safe infrastructure failures."""


class InfrastructureError(Exception):
    """Base failure exposed through the infrastructure API."""

    code = "infrastructure_error"
    status_code = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InfrastructureConfigurationError(InfrastructureError):
    code = "infrastructure_not_configured"
    status_code = 503


class InfrastructureUnavailableError(InfrastructureError):
    code = "infrastructure_unavailable"
    status_code = 503


class JobNotFoundError(InfrastructureError):
    code = "job_not_found"
    status_code = 404


class JobConflictError(InfrastructureError):
    code = "job_conflict"
    status_code = 409
