"""Safe in-memory decoding for the multimodal OCR provider."""

import re
from dataclasses import dataclass
from io import BytesIO

import fitz
from PIL import Image, ImageOps, ImageSequence, UnidentifiedImageError
from pillow_heif import register_heif_opener

from app.ocr.providers.contracts import ProviderDocument
from app.ocr.providers.errors import UnsupportedDocumentError

_IMAGE_FORMATS = {
    "PNG": "png",
    "JPEG": "jpeg",
    "TIFF": "tiff",
    "BMP": "bmp",
    "WEBP": "webp",
    "HEIF": "heic",
}
_FILE_TYPE_ALIASES = {
    "application/pdf": "pdf",
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpg",
    "image/tiff": "tiff",
    "image/bmp": "bmp",
    "image/webp": "webp",
    "image/heic": "heic",
    "image/heif": "heif",
    ".pdf": "pdf",
    ".png": "png",
    ".jpeg": "jpeg",
    ".jpg": "jpg",
    ".tiff": "tiff",
    ".tif": "tiff",
    "tif": "tiff",
    ".bmp": "bmp",
    ".webp": "webp",
    ".heic": "heic",
    ".heif": "heif",
}


@dataclass(frozen=True, slots=True)
class DigitalPDFText:
    page_texts: tuple[str, ...]
    dimensions: tuple[tuple[int, int], ...]

    @property
    def text(self) -> str:
        return "\n\n".join(self.page_texts)


register_heif_opener()


@dataclass(frozen=True, slots=True)
class DecodedPage:
    page_number: int
    image: Image.Image
    original_width: int
    original_height: int


def decode_document_pages(
    document: ProviderDocument,
    *,
    pdf_render_dpi: int,
    max_image_size: int,
    max_pages: int,
) -> tuple[DecodedPage, ...]:
    """Decode supported content with bounded pages and dimensions."""

    normalized_type = _normalize_file_type(document.file_type)
    if normalized_type == "pdf":
        pages = _decode_pdf(document.content, pdf_render_dpi, max_pages)
    elif normalized_type in {"png", "jpeg", "jpg", "tiff", "bmp", "webp", "heic", "heif"}:
        pages = _decode_image(document.content, normalized_type, max_pages)
    else:
        raise UnsupportedDocumentError(
            f"File type {document.file_type!r} is not supported by this provider."
        )
    return tuple(_bounded_page(page, max_image_size) for page in pages)


def extract_digital_pdf_text(
    document: ProviderDocument,
    *,
    max_pages: int,
    minimum_characters_per_page: int = 40,
) -> DigitalPDFText | None:
    """Return a usable native PDF text layer, or None when OCR is required."""

    if _normalize_file_type(document.file_type) != "pdf":
        return None
    if not document.content.startswith(b"%PDF-"):
        raise UnsupportedDocumentError("The uploaded PDF signature is invalid.")
    try:
        pdf = fitz.open(stream=document.content, filetype="pdf")
    except Exception as exc:
        raise UnsupportedDocumentError("The uploaded PDF cannot be decoded.") from exc
    try:
        if pdf.needs_pass:
            raise UnsupportedDocumentError("Encrypted PDF documents are not supported.")
        if pdf.page_count <= 0:
            raise UnsupportedDocumentError("The uploaded PDF contains no pages.")
        if pdf.page_count > max_pages:
            raise UnsupportedDocumentError(
                f"The PDF exceeds the configured {max_pages}-page limit."
            )
        page_texts: list[str] = []
        dimensions: list[tuple[int, int]] = []
        for page in pdf:
            text = _normalize_pdf_text(page.get_text("text", sort=True))
            if len(text) < minimum_characters_per_page:
                return None
            page_texts.append(text)
            dimensions.append((round(page.rect.width), round(page.rect.height)))
        return DigitalPDFText(tuple(page_texts), tuple(dimensions))
    finally:
        pdf.close()


def _normalize_pdf_text(value: str) -> str:
    lines = []
    for line in value.splitlines():
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+", " ", line)
        cleaned = " ".join(cleaned.split())
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def _normalize_file_type(value: str) -> str:
    normalized = value.strip().casefold()
    return _FILE_TYPE_ALIASES.get(normalized, normalized.removeprefix("."))


def _decode_pdf(data: bytes, dpi: int, max_pages: int) -> tuple[DecodedPage, ...]:
    if not data.startswith(b"%PDF-"):
        raise UnsupportedDocumentError("The uploaded PDF signature is invalid.")
    try:
        pdf = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise UnsupportedDocumentError("The uploaded PDF cannot be decoded.") from exc
    try:
        if pdf.needs_pass:
            raise UnsupportedDocumentError("Encrypted PDF documents are not supported.")
        if pdf.page_count <= 0:
            raise UnsupportedDocumentError("The uploaded PDF contains no pages.")
        if pdf.page_count > max_pages:
            raise UnsupportedDocumentError(
                f"The PDF exceeds the configured {max_pages}-page limit."
            )
        scale = dpi / 72
        matrix = fitz.Matrix(scale, scale)
        pages: list[DecodedPage] = []
        for index in range(pdf.page_count):
            try:
                pixmap = pdf.load_page(index).get_pixmap(matrix=matrix, alpha=False)
                image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            except Exception as exc:
                raise UnsupportedDocumentError(f"PDF page {index + 1} cannot be rendered.") from exc
            pages.append(
                DecodedPage(
                    page_number=index + 1,
                    image=image,
                    original_width=image.width,
                    original_height=image.height,
                )
            )
        return tuple(pages)
    finally:
        pdf.close()


def _decode_image(
    data: bytes,
    expected_type: str,
    max_pages: int,
) -> tuple[DecodedPage, ...]:
    try:
        with Image.open(BytesIO(data)) as source:
            detected_type = _IMAGE_FORMATS.get(source.format or "")
            if detected_type is None:
                raise UnsupportedDocumentError("The uploaded image format is unsupported.")
            if expected_type in {"jpeg", "jpg"}:
                expected_type = "jpeg"
            if expected_type in {"heic", "heif"}:
                expected_type = "heic"
            if detected_type != expected_type:
                raise UnsupportedDocumentError(
                    "The declared image type does not match the decoded content."
                )
            frame_count = getattr(source, "n_frames", 1)
            if frame_count > max_pages:
                raise UnsupportedDocumentError(
                    f"The image exceeds the configured {max_pages}-page limit."
                )
            pages = []
            for index, frame in enumerate(ImageSequence.Iterator(source), start=1):
                normalized = ImageOps.exif_transpose(frame).convert("RGB").copy()
                pages.append(
                    DecodedPage(
                        page_number=index,
                        image=normalized,
                        original_width=normalized.width,
                        original_height=normalized.height,
                    )
                )
            return tuple(pages)
    except UnsupportedDocumentError:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
        raise UnsupportedDocumentError("The uploaded image cannot be decoded safely.") from exc


def _bounded_page(page: DecodedPage, max_image_size: int) -> DecodedPage:
    image = page.image
    longest_edge = max(image.size)
    if longest_edge > max_image_size:
        ratio = max_image_size / longest_edge
        target = (
            max(1, round(image.width * ratio)),
            max(1, round(image.height * ratio)),
        )
        image = image.resize(target, Image.Resampling.LANCZOS)
    return DecodedPage(
        page_number=page.page_number,
        image=image,
        original_width=page.original_width,
        original_height=page.original_height,
    )
