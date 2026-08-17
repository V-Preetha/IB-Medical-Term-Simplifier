"""Domain exceptions translated to API responses by the FastAPI application."""


class IngestionError(Exception):
    """Base exception for expected report-processing failures."""

    status_code = 422
    code = "processing_failed"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class EmptyFileError(IngestionError):
    status_code = 400
    code = "empty_file"


class FileTooLargeError(IngestionError):
    status_code = 413
    code = "file_too_large"


class UnsupportedDocumentError(IngestionError):
    status_code = 415
    code = "unsupported_document"


class CorruptDocumentError(IngestionError):
    status_code = 422
    code = "corrupt_document"


class OcrUnavailableError(IngestionError):
    status_code = 503
    code = "ocr_unavailable"


class UnreadableDocumentError(IngestionError):
    status_code = 422
    code = "unreadable_document"


class InvalidManualReviewError(IngestionError):
    status_code = 400
    code = "invalid_manual_review"


class ManualReviewNotFoundError(IngestionError):
    status_code = 404
    code = "manual_review_not_found"


class ReportJobNotFoundError(IngestionError):
    status_code = 404
    code = "report_job_not_found"


class ReportJobQueueFullError(IngestionError):
    status_code = 503
    code = "report_job_queue_full"
