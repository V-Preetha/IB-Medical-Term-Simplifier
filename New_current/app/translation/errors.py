"""Safe translation failures."""


class TranslationError(Exception):
    def __init__(self, message: str, *, code: str = "translation_error", status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class TranslationUnavailableError(TranslationError):
    def __init__(self, message: str):
        super().__init__(message, code="translation_unavailable", status_code=503)


class TranslationPreservationError(TranslationError):
    def __init__(self):
        super().__init__(
            "Translation could not prove preservation of protected medical content.",
            code="translation_preservation_failed",
            status_code=422,
        )
