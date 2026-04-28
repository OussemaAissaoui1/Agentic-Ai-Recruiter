"""PDF / DOCX / TXT / image extraction for uploaded CVs.

PDFs are first tried with PaddleOCR (most accurate on real-world CVs that
mix glyphs, columns and embedded images). If OCR fails or returns
essentially nothing, we fall back to pypdf. Pure image uploads
(.jpg/.jpeg/.png/.bmp/.tiff/.webp) go straight to OCR.
"""

from __future__ import annotations

import os
from typing import Optional

import pypdf


PDF_OCR_MIN_CHARS = int(os.environ.get("PDF_OCR_MIN_CHARS", "100"))
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")


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
    """OCR first, pypdf as fallback.

    PaddleOCR yields cleaner text on the real CVs we see (mixed columns,
    icons, embedded images), so we run it first. If OCR is unavailable or
    produces less text than pypdf, we fall back to pypdf.
    """
    ocr_text = ""
    try:
        from .ocr import ocr_pdf
        ocr_text = ocr_pdf(path)
    except Exception as e:
        print(f"[pdf_reader] OCR failed for {path}: {e}", flush=True)

    if len(ocr_text) >= PDF_OCR_MIN_CHARS:
        return ocr_text

    pypdf_text = _read_pdf_pypdf(path)
    if len(pypdf_text) > len(ocr_text):
        return pypdf_text
    return ocr_text


def read_docx(path: str) -> str:
    import docx  # python-docx
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs).strip()


def read_image(path: str) -> str:
    from .ocr import ocr_image
    return ocr_image(path)


def read_any(path: str) -> Optional[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return read_pdf(path)
    if ext in (".docx", ".doc"):
        return read_docx(path)
    if ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    if ext in IMAGE_EXTS:
        return read_image(path)
    return None
