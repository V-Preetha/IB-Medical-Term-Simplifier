"""Application-level OCR failures."""


class OCRApplicationError(Exception):
    code = "ocr_application_error"
    status_code = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class OCRNotFoundError(OCRApplicationError):
    code = "ocr_not_found"
    status_code = 404


class OCRUploadError(OCRApplicationError):
    code = "invalid_upload"
    status_code = 400


class OCRUploadTooLargeError(OCRApplicationError):
    code = "upload_too_large"
    status_code = 413
