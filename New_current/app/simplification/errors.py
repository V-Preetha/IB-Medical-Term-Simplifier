"""Safe simplification failures."""


class SimplificationError(Exception):
    def __init__(self, message: str, *, code: str = "simplification_error", status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class SimplificationUnavailableError(SimplificationError):
    def __init__(self, message: str):
        super().__init__(message, code="simplification_unavailable", status_code=503)


class SimplificationOutputError(SimplificationError):
    def __init__(self, message: str):
        super().__init__(message, code="unsafe_simplification_output", status_code=422)
