"""OCR backend — runs PaddleOCR in a subprocess for cuDNN isolation.

Why subprocess:
    paddlepaddle-gpu 3.0 ships cuDNN 9.5.x; PyTorch 2.x ships cuDNN 9.8.x.
    Loading both into the same process picks paddle's cuDNN first and
    breaks the next torch CUDA op with:
        cuDNN version incompatibility: PyTorch was compiled against
        (9, 8, 0) but found runtime version (9, 5, 1).
    The fix is to keep paddle's cuDNN out of the FastAPI process by
    running paddle in a long-lived child process. PDF rendering with
    PyMuPDF stays in the parent (CPU only) and we only ship per-page
    PNG paths to the child.

Config:
    OCR_USE_GPU       1 / 0 / auto (default auto)
    PADDLE_OCR_LANG   passed through to PaddleOCR (default "en")
    PDF_OCR_DPI       PDF render DPI (default 150)
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from typing import List, Optional


_worker: Optional[subprocess.Popen] = None
_worker_lock = threading.Lock()
_atexit_registered = False


def _resolve_use_gpu() -> bool:
    """Pick CPU vs GPU. OCR_USE_GPU=1/0/auto (default auto).

    Auto only returns True when torch reports CUDA available. We can't
    probe paddle here (it's in the child), so the worker itself does a
    safe fallback if its GPU build isn't installed.
    """
    pref = os.environ.get("OCR_USE_GPU", "auto").strip().lower()
    if pref in {"0", "false", "no", "cpu"}:
        return False
    if pref in {"1", "true", "yes", "gpu"}:
        return True
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _spawn_worker() -> subprocess.Popen:
    global _atexit_registered

    env = os.environ.copy()
    env["_OCR_WORKER_USE_GPU"] = "1" if _resolve_use_gpu() else "0"
    # Make sure the worker can find the backend package as a module.
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{project_root}{os.pathsep}{pythonpath}" if pythonpath else project_root
    )

    cmd = [sys.executable, "-u", "-m", "backend.utils._ocr_worker"]
    print(
        f"[ocr] spawning worker (use_gpu={env['_OCR_WORKER_USE_GPU']}); "
        f"first call will pay model-load latency",
        flush=True,
    )

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,  # forward worker stderr (paddle logs, tracebacks)
        text=True,
        bufsize=1,
        env=env,
        cwd=project_root,
    )

    # Wait for READY (paddle warm-up happens before this line is sent).
    deadline = time.monotonic() + 300.0
    line = ""
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            # EOF — child died during warm-up.
            raise RuntimeError(
                "OCR worker exited during startup. Check stderr above for "
                "the actual paddle error."
            )
        if line.strip() == "READY":
            break
        # Anything before READY is unexpected; surface it but keep waiting.
        sys.stderr.write(f"[ocr-worker:stdout-preamble] {line}")
    else:
        proc.terminate()
        raise RuntimeError("OCR worker timed out during startup (>300 s)")

    if not _atexit_registered:
        atexit.register(_terminate_worker)
        _atexit_registered = True

    print("[ocr] worker ready", flush=True)
    return proc


def _terminate_worker() -> None:
    global _worker
    if _worker is None:
        return
    try:
        if _worker.stdin and not _worker.stdin.closed:
            _worker.stdin.close()
        _worker.terminate()
        _worker.wait(timeout=5)
    except Exception:
        try:
            _worker.kill()
        except Exception:
            pass
    finally:
        _worker = None


def _get_worker() -> subprocess.Popen:
    global _worker
    if _worker is None or _worker.poll() is not None:
        _worker = _spawn_worker()
    return _worker


def _send(req: dict) -> str:
    payload = json.dumps(req) + "\n"
    with _worker_lock:
        w = _get_worker()
        try:
            w.stdin.write(payload)
            w.stdin.flush()
            line = w.stdout.readline()
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError(f"OCR worker pipe broken: {e}")
    if not line:
        raise RuntimeError("OCR worker closed unexpectedly")
    try:
        resp = json.loads(line)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"OCR worker bad response: {line[:300]!r} ({e})")
    if "error" in resp:
        raise RuntimeError(f"OCR worker error: {resp['error']}")
    return resp.get("text", "")


def ocr_image(path: str) -> str:
    name = os.path.basename(path)
    print(f"[ocr] image {name} ...", flush=True)
    t0 = time.monotonic()
    text = _send({"action": "image", "path": path})
    print(
        f"[ocr] image {name} done in {time.monotonic() - t0:.1f}s "
        f"({len(text)} chars)",
        flush=True,
    )
    return text


def ocr_pdf(path: str, dpi: Optional[int] = None) -> str:
    """Render every PDF page to PNG in the parent (PyMuPDF, CPU only),
    then ship each page path to the OCR worker."""
    import fitz  # pymupdf

    if dpi is None:
        dpi = int(os.environ.get("PDF_OCR_DPI", "150"))

    pieces: List[str] = []
    name = os.path.basename(path)
    doc = fitz.open(path)
    try:
        n_pages = len(doc)
        print(f"[ocr] pdf {name}: {n_pages} page(s) at {dpi} dpi", flush=True)
        for i, page in enumerate(doc):
            t0 = time.monotonic()
            pix = page.get_pixmap(dpi=dpi)
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            try:
                tmp.write(pix.tobytes("png"))
                tmp.flush()
                tmp.close()
                page_text = _send({"action": "image", "path": tmp.name})
                pieces.append(page_text)
                print(
                    f"[ocr] pdf {name} page {i + 1}/{n_pages} "
                    f"{time.monotonic() - t0:.1f}s ({len(page_text)} chars)",
                    flush=True,
                )
            finally:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass
    finally:
        doc.close()
    return "\n\n".join(p for p in pieces if p).strip()
