"""PDF / DOCX / TXT / image extraction for uploaded CVs.

Docling is the primary backend — it covers PDFs (text + scanned), DOCX,
PPTX, HTML, MD and images in one call, with built-in OCR via RapidOCR's
torch backend. pypdf remains as a fallback for the rare PDF Docling
chokes on.
"""

from __future__ import annotations

import os
from typing import Optional

import pypdf

from .docling_reader import convert_to_text


PDF_OCR_MIN_CHARS = int(os.environ.get("PDF_OCR_MIN_CHARS", "100"))
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")
DOCLING_EXTS = (".pdf", ".docx", ".doc", ".pptx", ".html", ".md", *IMAGE_EXTS)


def _read_pdf_pypdf(path: str) -> str:
    reader = pypdf.PdfReader(path)
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts).strip()


def read_pdf(path: str) -> str:
    """Docling first, pypdf as fallback for malformed PDFs."""
    docling_text = convert_to_text(path)
    if len(docling_text) >= PDF_OCR_MIN_CHARS:
        return docling_text

    pypdf_text = _read_pdf_pypdf(path)
    if len(pypdf_text) > len(docling_text):
        return pypdf_text
    return docling_text


def read_docx(path: str) -> str:
    """Try Docling first; fall back to python-docx for very simple files."""
    text = convert_to_text(path)
    if text:
        return text
    import docx  # python-docx
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs).strip()


def read_image(path: str) -> str:
    return convert_to_text(path)


def read_any(path: str) -> Optional[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    if ext == ".pdf":
        return read_pdf(path)
    if ext in (".docx", ".doc"):
        return read_docx(path)
    if ext in IMAGE_EXTS:
        return read_image(path)
    if ext in DOCLING_EXTS:
        return convert_to_text(path)
    return None
