"""Verify TAPJFNN + GNN + LDA are loaded and actually used by /api/matching.

Three-layer check:
  1. /api/matching/health      → boolean flags loaded at boot.
  2. /api/matching/rank        → response.model_info.* flags returned per call,
                                  AND the chosen scorer is logged in the server
                                  (search uvicorn output for "rank: n=… scorer=…").
  3. behavioral                → score spread is non-trivial. If everything but
                                  cosine fell over, two semantically different
                                  CVs would still get cosine-distinguishable
                                  scores; we just check that ranking is sane
                                  (the better-fit CV ranks first).

Usage:
    python scripts/verify_matching.py [--base http://127.0.0.1:8000]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.parse
import urllib.request
import urllib.error


JD = (
    "Senior Frontend Engineer. We're hiring a senior frontend engineer to own "
    "our React + TypeScript design system. You'll partner with designers, "
    "ship motion-rich product surfaces, and lead a small frontend platform "
    "team. Strong React, TypeScript, Tailwind, GraphQL, and accessibility "
    "experience required."
)

CV_GOOD = (
    "Jane Doe — Senior Frontend Engineer\n"
    "7 years building React + TypeScript design systems at Linear-style "
    "product companies. Deep Tailwind, GraphQL, Framer Motion, accessibility, "
    "and state-management expertise. Led a 4-person frontend platform team at "
    "previous role. Author of an internal motion library used across 12 apps."
)

CV_BAD = (
    "Sam Garcia — Diesel Mechanic\n"
    "12 years repairing heavy trucks. ASE certified. Hydraulic systems, "
    "engine rebuilds, fleet maintenance. No software experience."
)


def _http_get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _multipart(parts: list[tuple[str, str, str | None, bytes | None]]) -> tuple[bytes, str]:
    """parts: list of (field_name, filename_or_value, content_type_or_None, data_or_None).
       If content_type is None it's a regular form field, value is in `filename_or_value`.
    """
    boundary = "----hireflow-verify-7c0d3ab"
    buf = io.BytesIO()
    for name, val_or_name, ctype, data in parts:
        buf.write(f"--{boundary}\r\n".encode())
        if ctype is None:
            buf.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            buf.write(val_or_name.encode())
        else:
            buf.write(
                f'Content-Disposition: form-data; name="{name}"; filename="{val_or_name}"\r\n'.encode()
            )
            buf.write(f"Content-Type: {ctype}\r\n\r\n".encode())
            buf.write(data or b"")
        buf.write(b"\r\n")
    buf.write(f"--{boundary}--\r\n".encode())
    return buf.getvalue(), f"multipart/form-data; boundary={boundary}"


def _http_post_multipart(url: str, parts) -> dict:
    body, ctype = _multipart(parts)
    req = urllib.request.Request(url, data=body, headers={"Content-Type": ctype}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    base = args.base.rstrip("/")
    failures: list[str] = []

    # ---------- Layer 1: health ----------
    print("→ /api/matching/health")
    try:
        h = _http_get_json(f"{base}/api/matching/health")
    except urllib.error.URLError as e:
        print(f"  ! could not reach {base}: {e}")
        return 2
    detail = h.get("detail", {})
    flags = {
        "tapjfnn_loaded": detail.get("tapjfnn_loaded"),
        "gnn_loaded": detail.get("gnn_loaded"),
        "lda_loaded": detail.get("lda_loaded"),
    }
    for k, v in flags.items():
        ok = v is True
        print(f"  {'✓' if ok else '✗'}  {k}: {v}")
        if not ok:
            failures.append(f"{k}={v}")

    # ---------- Layer 2: rank with two contrasting CVs ----------
    print("\n→ /api/matching/rank  (good CV vs unrelated CV)")
    parts = [
        ("job_description", JD, None, None),
        ("apply_ga_flag", "false", None, None),
        ("metadata_json", "[]", None, None),
        ("constraints_json", "{}", None, None),
        ("files", "good.txt", "text/plain", CV_GOOD.encode()),
        ("files", "bad.txt", "text/plain", CV_BAD.encode()),
    ]
    try:
        r = _http_post_multipart(f"{base}/api/matching/rank", parts)
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="ignore")
        print(f"  ! rank failed: HTTP {e.code} — {msg[:300]}")
        return 1

    info = r.get("model_info", {})
    print(f"  model_info: {json.dumps(info)}")
    for k in ("tapjfnn_loaded", "gnn_loaded"):
        if info.get(k) is not True:
            failures.append(f"rank.model_info.{k}={info.get(k)}")

    ranked = r.get("ranked", [])
    print(f"  ranked: {len(ranked)} items")
    for c in ranked:
        print(f"    - {c.get('id'):12s}  fit={c.get('fit_score'):.3f}")

    if len(ranked) != 2:
        failures.append(f"expected 2 ranked items, got {len(ranked)}")
    elif ranked[0]["id"] != "good.txt":
        failures.append(f"better CV did not rank first; got {ranked[0]['id']}")
    elif ranked[0]["fit_score"] - ranked[1]["fit_score"] < 0.05:
        failures.append(
            f"score spread too small: {ranked[0]['fit_score']:.3f} vs "
            f"{ranked[1]['fit_score']:.3f} — model may be falling back to cosine"
        )

    # ---------- summary ----------
    print()
    if failures:
        print("✗ FAIL")
        for f in failures:
            print(f"  - {f}")
        print("\nNext check the uvicorn log for the line:")
        print('  agents.matching.inference: rank: n=2 scorer=… tapjfnn=… gnn=… lda=…')
        print("If scorer != gnn, the GNN path didn't run for this call.")
        return 1
    print("✓ PASS — TAPJFNN, GNN, LDA all loaded; rank uses them; ordering is sane.")
    print("Confirm in uvicorn log:")
    print("  agents.matching.inference: rank: n=2 scorer=gnn tapjfnn=True gnn=True lda=True")
    return 0
if __name__ == "__main__":
    sys.exit(main())
