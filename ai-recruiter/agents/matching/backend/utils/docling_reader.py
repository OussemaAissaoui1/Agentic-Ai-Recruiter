"""Docling-backed multi-format document reader.

Replaces the previous PaddleOCR + PyMuPDF subprocess pipeline. Docling
parses PDF / DOCX / PPTX / HTML / MD / PNG / JPG / TIFF / WEBP through a
single `DocumentConverter` entry point and runs OCR (RapidOCR torch
backend) when needed for image-only pages. Layout, tables, columns and
embedded images all collapse to plain text via `export_to_text()`.

Why not import directly: model weights are downloaded from HuggingFace
on first use (~250 MB for the layout + tableformer + RapidOCR set), and
torch CUDA init is ~1 s, so we lazy-init a single converter and reuse it.

Config:
    OCR_USE_GPU       1 / 0 / auto (default auto, follows torch.cuda)
    OCR_DEVICE_INDEX  GPU index when use_gpu (default 0)
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

log = logging.getLogger("agents.matching.docling_reader")

_converter = None  # type: ignore[var-annotated]
_converter_lock = threading.Lock()
_init_error: Optional[str] = None


def _resolve_device() -> str:
    """Pick "cuda" / "cpu" honouring OCR_USE_GPU.

    Default is CPU: this process shares the GPU with vLLM, which holds
    ~85 GB once warm. Docling-on-GPU OOMs in that state. Set
    OCR_USE_GPU=1 explicitly when running OCR before vLLM warms up,
    or when running the matching agent on a dedicated GPU.
    """
    pref = os.environ.get("OCR_USE_GPU", "0").strip().lower()
    if pref in {"1", "true", "yes", "gpu"}:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
    return "cpu"


def _build_converter():
    """Construct a singleton DocumentConverter with GPU acceleration where possible."""
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    device = _resolve_device()
    accel = AcceleratorOptions(
        device=AcceleratorDevice.CUDA if device == "cuda" else AcceleratorDevice.CPU,
        cuda_use_flash_attention2=False,
    )

    pdf_opts = PdfPipelineOptions()
    pdf_opts.accelerator_options = accel
    pdf_opts.do_ocr = True
    pdf_opts.do_table_structure = True
    pdf_opts.generate_page_images = False
    pdf_opts.generate_picture_images = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
        },
    )
    log.info("docling converter ready (device=%s)", device)
    return converter


def _get_converter():
    """Lazy singleton. Sticky error cache prevents repeated bad inits."""
    global _converter, _init_error
    if _converter is not None:
        return _converter
    if _init_error is not None:
        raise RuntimeError(_init_error)
    with _converter_lock:
        if _converter is not None:
            return _converter
        try:
            _converter = _build_converter()
        except Exception as e:  # noqa: BLE001 — surface init failure once
            _init_error = f"docling init failed: {type(e).__name__}: {e}"
            log.exception("docling init failed")
            raise RuntimeError(_init_error) from e
        return _converter


def convert_to_text(path: str) -> str:
    """Convert any supported document to plain text via Docling.

    Returns an empty string on failure (callers may fall back to pypdf).
    """
    try:
        conv = _get_converter()
        result = conv.convert(path)
        doc = result.document
        # Prefer plain text; fall back to markdown if the export helper
        # isn't available in older docling-core builds.
        try:
            text = doc.export_to_text()
        except AttributeError:
            text = doc.export_to_markdown()
        return (text or "").strip()
    except Exception as e:  # noqa: BLE001
        log.warning("docling convert failed for %s: %s", os.path.basename(path), e)
        return ""


__all__ = ["convert_to_text"]
