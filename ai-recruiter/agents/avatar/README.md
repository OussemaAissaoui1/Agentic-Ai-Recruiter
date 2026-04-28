# Avatar Agent

Serves the 3D recruiter avatar (Ready Player Me GLB) and extracts
viseme timelines from TTS audio for browser-side lip sync.

## Routes (`/api/avatar`)

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/health` | Readiness + GLB size |
| `GET`  | `/glb`    | Returns the GLB binary (cached 24 h) |
| `POST` | `/visemes` | `{ wav_b64 }` → viseme timeline (list of frames) |

## A2A capabilities

`get_glb`, `extract_visemes`.

## Files

- `brunette.glb` — checked-in avatar model.
- `viseme_engine.py` — spectral analysis → ARKit-compatible visemes.

## Used by

The interview frontend renders the GLB with Three.js and consumes the
viseme timeline produced from each TTS chunk to drive blend-shape lip
sync. Avatar runs alongside (not instead of) the audio playback.
