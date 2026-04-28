"""Subprocess worker that runs PaddleOCR in isolation.

Why a subprocess:
    paddlepaddle-gpu 3.0 bundles cuDNN 9.5.x in its wheel. PyTorch 2.x is
    compiled against cuDNN 9.8.x. Loading both into the same process means
    paddle's cuDNN wins (it dlopens its own libs first), and the next time
    torch tries to do anything CUDA-related it fails with:
        cuDNN version incompatibility: PyTorch was compiled against
        (9, 8, 0) but found runtime version (9, 5, 1).
    Running paddle in a child process keeps its cuDNN out of the FastAPI
    process entirely.

Protocol (line-delimited JSON over stdin/stdout):
    parent -> child: {"action": "image", "path": "/tmp/page.png"}
    child  -> parent: {"text": "..."}            on success
    child  -> parent: {"error": "ExcType: msg"}  on failure
    A single line "READY" is sent on stdout before the request loop
    starts so the parent can wait for paddle warm-up to finish.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import List


# Disable PIR + oneDNN in the child too — the converter unimplemented op
# crash applies to GPU paddle as well on some configurations.
for _k, _v in {
    "FLAGS_enable_pir_in_executor": "0",
    "FLAGS_enable_pir_api": "0",
    "FLAGS_enable_new_ir_in_executor": "0",
    "FLAGS_enable_new_ir_api": "0",
    "FLAGS_enable_new_executor": "0",
    "FLAGS_use_mkldnn": "0",
}.items():
    os.environ.setdefault(_k, _v)


_paddle = None


def _patch_inference() -> None:
    try:
        import paddle.inference as pi
    except Exception:
        return
    Config = pi.Config
    if getattr(Config, "_cv_ranking_patched", False):
        return

    def _post_init(cfg) -> None:
        for name in ("disable_mkldnn", "disable_new_ir"):
            fn = getattr(cfg, name, None)
            if fn is not None:
                try:
                    fn()
                except Exception:
                    pass
        for name, args in (("enable_new_ir", (False,)), ("switch_ir_optim", (False,))):
            fn = getattr(cfg, name, None)
            if fn is None:
                continue
            try:
                fn(*args)
            except Exception:
                pass

    try:
        class _PatchedConfig(Config):  # type: ignore[misc]
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                _post_init(self)

        _PatchedConfig._cv_ranking_patched = True  # type: ignore[attr-defined]
        pi.Config = _PatchedConfig
    except Exception:
        pass


def _get_paddle():
    global _paddle
    if _paddle is None:
        try:
            import paddle
            for k, v in {
                "FLAGS_enable_pir_in_executor": False,
                "FLAGS_enable_new_ir_in_executor": False,
                "FLAGS_use_mkldnn": False,
            }.items():
                try:
                    paddle.set_flags({k: v})
                except Exception:
                    continue
        except Exception:
            pass
        _patch_inference()

        from paddleocr import PaddleOCR

        lang = os.environ.get("PADDLE_OCR_LANG", "en")
        use_gpu = os.environ.get("_OCR_WORKER_USE_GPU", "0") == "1"
        gpu_kw = {"use_gpu": use_gpu}
        mkldnn_kw = {"enable_mkldnn": False} if not use_gpu else {}

        candidates = [
            {"use_angle_cls": True, "lang": lang, **gpu_kw, **mkldnn_kw, "show_log": False},
            {"use_angle_cls": True, "lang": lang, **gpu_kw, **mkldnn_kw},
            {"use_angle_cls": True, "lang": lang, **mkldnn_kw, "show_log": False},
            {"use_angle_cls": True, "lang": lang, **mkldnn_kw},
            {"use_angle_cls": True, "lang": lang},
            {"lang": lang},
            {},
        ]
        last: Exception | None = None
        for kw in candidates:
            try:
                _paddle = PaddleOCR(**kw)
                break
            except (TypeError, ValueError) as e:
                last = e
            except Exception as e:
                last = e
        if _paddle is None:
            raise RuntimeError(f"PaddleOCR init failed: {last}")
    return _paddle


def _run_ocr(ocr, path: str):
    if hasattr(ocr, "ocr"):
        try:
            return ocr.ocr(path, cls=True)
        except TypeError:
            return ocr.ocr(path)
    if hasattr(ocr, "predict"):
        return ocr.predict(path)
    raise RuntimeError("PaddleOCR has no .ocr or .predict method")


def _flatten(result) -> str:
    if not result:
        return ""
    lines: List[str] = []
    pages = result if isinstance(result, list) else [result]
    for page in pages:
        if not page:
            continue
        for entry in page:
            try:
                payload = entry[1] if len(entry) >= 2 else entry
                text = payload[0] if isinstance(payload, (list, tuple)) else payload
                text = str(text).strip()
                if text:
                    lines.append(text)
            except Exception:
                continue
    return "\n".join(lines)


def _ocr_image(path: str) -> str:
    return _flatten(_run_ocr(_get_paddle(), path))


def main() -> None:
    # Eagerly load paddle so warm-up happens before READY — that way the
    # parent's first OCR call doesn't pay the model-load latency on the
    # critical path.
    try:
        _get_paddle()
    except Exception:
        # If load fails, still send READY and surface the error on first
        # request so parent gets a useful traceback instead of a hang.
        sys.stderr.write("[ocr-worker] init failed:\n")
        sys.stderr.write(traceback.format_exc())
        sys.stderr.flush()

    sys.stdout.write("READY\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            action = req.get("action")
            if action == "image":
                text = _ocr_image(req["path"])
            elif action == "ping":
                text = "pong"
            else:
                raise ValueError(f"unknown action: {action}")
            resp = {"text": text}
        except Exception as e:
            resp = {"error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
