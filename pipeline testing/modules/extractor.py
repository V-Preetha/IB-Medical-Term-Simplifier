"""Input detection, native PDF extraction, and OCR."""
from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger(__name__)

class ExtractionError(RuntimeError):
    """Raised when a supported document cannot be read."""

@dataclass(frozen=True)
class ExtractionResult:
    file_type: str
    method: str
    text: str
    page_count: int = 1

class TextExtractor:
    SUPPORTED = {".pdf", ".png", ".jpg", ".jpeg", ".txt"}
    def __init__(self, min_pdf_chars_per_page: int = 40, ocr_dpi: int = 300) -> None:
        self.min_chars, self.ocr_dpi = min_pdf_chars_per_page, ocr_dpi

    def extract(self, input_path: str | Path) -> ExtractionResult:
        path = Path(input_path).expanduser().resolve()
        if not path.is_file(): raise ExtractionError(f"Input file does not exist: {path}")
        ext = path.suffix.lower()
        if ext not in self.SUPPORTED: raise ExtractionError(f"Unsupported type {ext}; use PDF, PNG, JPG, JPEG, or TXT")
        if ext == ".txt": return self._check(ExtractionResult("text", "direct_text", path.read_text(encoding="utf-8-sig", errors="replace")))
        if ext == ".pdf": return self._pdf(path)
        raise ExtractionError("Image extraction must use the approved OCR service")

    def _pdf(self, path: Path) -> ExtractionResult:
        try: import fitz
        except ImportError as exc: raise ExtractionError("Install PyMuPDF from requirements.txt") from exc
        try:
            with fitz.open(path) as doc:
                pages = [page.get_text("text").strip() for page in doc]
                weak = [i for i, text in enumerate(pages) if len(text) < self.min_chars]
                if weak:
                    raise ExtractionError(
                        "Image-only PDF pages must use the approved OCR service"
                    )
                return self._check(
                    ExtractionResult("pdf", "pdf_text", "\n\n".join(pages), len(doc))
                )
        except ExtractionError: raise
        except Exception as exc: raise ExtractionError(f"Could not extract {path.name}: {exc}") from exc

    @staticmethod
    def _check(result: ExtractionResult) -> ExtractionResult:
        if not result.text.strip(): raise ExtractionError("No readable text was extracted")
        return result
