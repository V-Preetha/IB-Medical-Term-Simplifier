"""Document parsing service for text-bearing PDFs and text files."""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePath

import fitz
import pdfplumber

from app.config.settings import Settings

logger = logging.getLogger(__name__)


class DocumentParsingError(RuntimeError):
    """Raised when a document cannot be parsed into text."""


class SourceType(StrEnum):
    """Supported document source types."""

    TEXT = "text"
    PDF = "pdf"
    IMAGE = "image"


class ExtractionMethod(StrEnum):
    """Extraction strategy used for a document."""

    TEXT_DIRECT = "text_direct"
    PDF_TEXT = "pdf_text"
    PDF_TEXT_WITH_OCR = "pdf_text_with_ocr"
    OCR = "ocr"


@dataclass(frozen=True)
class ParsedDocument:
    """Clean extracted text and metadata from a parsed document."""

    text: str
    source_type: SourceType
    extraction_method: ExtractionMethod
    ocr_applied: bool
    page_count: int | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _PdfTextExtraction:
    """Intermediate per-page PDF extraction state."""

    page_texts: list[str]
    page_count: int
    warnings: list[str]


class DocumentParsingService:
    """Extract clean text from medical report documents."""

    _text_extensions = {".txt", ".text"}
    _pdf_extensions = {".pdf"}
    _image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    def __init__(self, settings: Settings) -> None:
        """Initialize parser with runtime settings."""
        self._settings = settings

    def parse_text(self, text: str) -> ParsedDocument:
        """Clean directly submitted plain text."""
        cleaned_text = clean_extracted_text(text)
        if not cleaned_text:
            raise DocumentParsingError("Submitted text is empty after cleaning.")

        logger.info("Parsed plain text input")
        return ParsedDocument(
            text=cleaned_text,
            source_type=SourceType.TEXT,
            extraction_method=ExtractionMethod.TEXT_DIRECT,
            ocr_applied=False,
        )

    def parse_file(
        self,
        *,
        filename: str,
        content: bytes,
        content_type: str | None = None,
    ) -> ParsedDocument:
        """Extract clean text from an uploaded report file."""
        if not content:
            raise DocumentParsingError("Uploaded file is empty.")

        source_type = self._detect_source_type(filename, content_type)
        logger.info("Parsing %s as %s", filename, source_type.value)

        if source_type is SourceType.TEXT:
            return self.parse_text(_decode_text(content))
        if source_type is SourceType.PDF:
            return self._parse_pdf(content)
        if source_type is SourceType.IMAGE:
            return self._parse_image(content)

        raise DocumentParsingError(f"Unsupported document type for {filename}.")

    def _detect_source_type(
        self,
        filename: str,
        content_type: str | None,
    ) -> SourceType:
        """Infer source type from filename extension and content type."""
        extension = PurePath(filename).suffix.lower()
        normalized_content_type = (content_type or "").lower()

        if extension in self._pdf_extensions or normalized_content_type == "application/pdf":
            return SourceType.PDF
        if extension in self._image_extensions or normalized_content_type.startswith("image/"):
            return SourceType.IMAGE
        if extension in self._text_extensions or normalized_content_type.startswith("text/"):
            return SourceType.TEXT

        raise DocumentParsingError(
            "Unsupported file type. Supported inputs are PDF, image, and text."
        )

    def _parse_pdf(self, content: bytes) -> ParsedDocument:
        """Extract embedded text from a PDF and fail closed for image-only pages."""
        extraction = self._extract_pdf_text(content)
        page_texts = extraction.page_texts
        pages_needing_ocr = [
            index
            for index, text in enumerate(page_texts)
            if self._page_needs_ocr(text)
        ]

        warnings = list(extraction.warnings)
        if pages_needing_ocr:
            warnings.append(
                "One or more image-only pages require the approved OCR service."
            )

        cleaned_text = clean_extracted_text("\n\n".join(page_texts))
        if not cleaned_text:
            raise DocumentParsingError("No readable text could be extracted from PDF.")

        return ParsedDocument(
            text=cleaned_text,
            source_type=SourceType.PDF,
            extraction_method=ExtractionMethod.PDF_TEXT,
            ocr_applied=False,
            page_count=extraction.page_count,
            warnings=warnings,
        )

    def _extract_pdf_text(self, content: bytes) -> _PdfTextExtraction:
        """Extract per-page text using PyMuPDF with pdfplumber fallback."""
        try:
            pdf_document = fitz.open(stream=content, filetype="pdf")
        except Exception as exc:
            logger.exception("PyMuPDF could not open PDF")
            raise DocumentParsingError("Uploaded PDF could not be opened.") from exc

        warnings: list[str] = []
        page_texts: list[str] = []
        with pdf_document:
            for page in pdf_document:
                page_texts.append(page.get_text("text") or "")

        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for index, page in enumerate(pdf.pages):
                    existing_text = page_texts[index] if index < len(page_texts) else ""
                    if self._page_needs_ocr(existing_text):
                        fallback_text = page.extract_text() or ""
                        if fallback_text and len(fallback_text) > len(existing_text):
                            page_texts[index] = fallback_text
        except Exception as exc:
            logger.warning("pdfplumber fallback failed: %s", exc)
            warnings.append("pdfplumber fallback failed; PyMuPDF text was used.")

        return _PdfTextExtraction(
            page_texts=page_texts,
            page_count=len(page_texts),
            warnings=warnings,
        )

    def _parse_image(self, content: bytes) -> ParsedDocument:
        """Reject image extraction outside the approved OCR service boundary."""
        del content
        raise DocumentParsingError(
            "Image extraction must be performed by the approved OCR service."
        )

    def _page_needs_ocr(self, text: str) -> bool:
        """Return whether extracted PDF page text is too sparse to trust."""
        normalized_text = clean_extracted_text(text)
        return len(normalized_text) < self._settings.pdf_text_min_chars_per_page


def clean_extracted_text(text: str) -> str:
    """Normalize extracted text without altering clinical values."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _decode_text(content: bytes) -> str:
    """Decode text bytes using UTF-8 with a safe Latin-1 fallback."""
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("UTF-8 decode failed; falling back to latin-1")
        return content.decode("latin-1")
