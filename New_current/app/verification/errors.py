"""Safe typed errors for medical verification."""


class VerificationError(Exception):
    status_code = 500
    code = "verification_error"


class VerificationUnavailableError(VerificationError):
    status_code = 503
    code = "verification_unavailable"


class VerificationInputError(VerificationError):
    status_code = 422
    code = "verification_input_invalid"
