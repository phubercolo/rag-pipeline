from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from pypdf import PdfReader

from rag_pipeline.config import Settings
from rag_pipeline.models import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def _load_ocr_dependencies() -> tuple[object, object, object]:
    """Import OCR dependencies with a friendly error if they are unavailable."""

    try:
        import fitz
        import pytesseract
        from PIL import Image
    except ModuleNotFoundError as exc:
        missing_module = exc.name or "an OCR dependency"
        logger.error(
            "OCR dependencies are missing (%s). Install them in your virtual "
            "environment with `python -m pip install -r requirements.txt`, then "
            "run ingest again.",
            missing_module,
        )
        raise RuntimeError("OCR dependencies are not installed") from exc

    return fitz, pytesseract, Image


def load_documents(documents_dir: Path, settings: Settings) -> list[Document]:
    """Load supported files from a folder into normalized Document objects."""

    documents: list[Document] = []

    for path in sorted(documents_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            text = extract_text(path, settings)
            if not text.strip():
                logger.warning(
                    "Skipping document with no extractable text: %s", path
                )
                continue

            doc_hash = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()
            documents.append(
                Document(
                    doc_id=doc_hash,
                    source_path=str(path),
                    content=normalize_whitespace(text),
                    metadata={"extension": path.suffix.lower(), "filename": path.name},
                )
            )
        except Exception as exc:
            logger.exception("Failed to ingest %s: %s", path, exc)

    logger.info("Loaded %s documents from %s", len(documents), documents_dir)
    return documents


def extract_text(path: Path, settings: Settings) -> str:
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        return extract_pdf_text(path, settings)

    raise ValueError(f"Unsupported file type: {path.suffix}")


def extract_pdf_text(path: Path, settings: Settings) -> str:
    """Extract text from PDFs, with optional OCR fallback for scanned files."""

    reader = PdfReader(str(path))
    pages: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(page_text)
            else:
                logger.debug("No extractable text on page %s of %s", page_number, path)
        except Exception as exc:
            logger.warning("Failed to read page %s of %s: %s", page_number, path, exc)

    extracted_text = "\n".join(pages).strip()
    if extracted_text:
        return extracted_text

    logger.info("No embedded PDF text found in %s", path)
    if not settings.enable_ocr:
        logger.info(
            "OCR is disabled. Set RAG_ENABLE_OCR=true to try OCR for scanned PDFs."
        )
        return ""

    return extract_pdf_text_with_ocr(path, settings)


def extract_pdf_text_with_ocr(path: Path, settings: Settings) -> str:
    """OCR a PDF by rendering each page to an image and sending it to Tesseract."""

    import io

    try:
        fitz, pytesseract, Image = _load_ocr_dependencies()
    except RuntimeError:
        return ""

    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    logger.info("Attempting OCR for %s", path)
    pages: list[str] = []

    with fitz.open(path) as pdf_document:
        for page_number, page in enumerate(pdf_document, start=1):
            try:
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                image = Image.open(io.BytesIO(pixmap.pil_tobytes(format="PNG")))
                page_text = pytesseract.image_to_string(image).strip()
                if page_text:
                    pages.append(page_text)
                else:
                    logger.debug("OCR found no text on page %s of %s", page_number, path)
            except pytesseract.TesseractNotFoundError:
                logger.error(
                    "Tesseract OCR is not installed or not on PATH. "
                    "Install Tesseract or set TESSERACT_CMD."
                )
                return ""
            except Exception as exc:
                logger.warning("OCR failed on page %s of %s: %s", page_number, path, exc)

    ocr_text = "\n".join(pages).strip()
    if ocr_text:
        logger.info("OCR extracted text from %s", path)
    else:
        logger.warning("OCR completed but found no text in %s", path)
    return ocr_text


def normalize_whitespace(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r\n", "\n").splitlines()]
    collapsed = "\n".join(line for line in lines if line)
    return collapsed.strip()
