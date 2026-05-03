# Vision Agent

**Stage 2 — runs alongside the live interview.** Streams the candidate's
webcam + microphone over a WebSocket and produces a real-time behavioral
profile, plus a PDF/JSON report at session end.

## What it does

| Modality | Pipeline | Source |
|---|---|---|
| Face / gaze | MediaPipe FaceMesh → 20-D AU+gaze features → temporal stats → MLP | `stress_detection/visual_model.py` |
| Body | MediaPipe Pose → 12-D pose features (openness, lean, stability, head nods) | `stress_detection/visual_model.py` |
| Emotion | HSEmotion EfficientNet-B0 → valence / arousal | `stress_detection/visual_model.py` |
| Hands | MediaPipe Hands → fidget / self-touch | `stress_detection/body_language.py` |
| Audio | Wav2Vec2-base → 512-D embedding → StudentNet MLP → stress prob | `stress_detection/audio_model.py` |
| Baseline | Welford 45s online calibration | `stress_detection/baseline.py` |
| Fusion | Multi-dimension weighted fusion → 4-D profile | `stress_detection/fusion.py` |

The 4-D profile is `Cognitive Load`, `Emotional Arousal`, `Engagement Level`,
`Confidence Level` — see the upstream `REPORT.md` for the dimension recipes.

## HTTP surface (mounted at `/api/vision`)

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/health` | Per-agent readiness + fallback reasons |
| `POST` | `/session/start` | Open a new analysis session |
| `POST` | `/session/{id}/question` | Mark an interviewer question on the timeline |
| `GET`  | `/report/{id}` | JSON behavioral report |
| `GET`  | `/report/{id}/pdf` | PDF behavioral report |

## WebSocket (mounted at `/ws/vision`)

The browser sends frames + audio chunks; the server replies with per-frame
results and dimension scores.

**Client → server messages:**
```jsonc
{ "type": "video_frame", "data": "<data:image/jpeg;base64,…>" }
{ "type": "audio_chunk", "data": "<base64 float32 PCM>", "sample_rate": 44100 }
{ "type": "calibration_start" }
{ "type": "calibration_complete" }
{ "type": "question_marker", "label": "Tell me about project X" }
{ "type": "session_end" }
{ "type": "ping" }
```

**Server → client messages:**
```jsonc
{ "type": "result", "frame_id": 42, "visual": {…}, "audio": {…},
  "dimension_scores": {"cognitive_load": 0.42, …}, "is_calibrated": true }
{ "type": "calibration_progress", "progress": 0.73 }
{ "type": "calibration_done", "success": true }
{ "type": "notable_moment", "dimension": "engagement", "magnitude": 1.8, "timestamp": … }
{ "type": "report_ready", "report_json": {…} }
{ "type": "pong" }
```

## Live wiring with the interview

The interview UI (built from `agents/avatar/demo/`, deployed to
`apps/static/interview/`) opens both connections concurrently:

1. `POST /api/nlp/session` for the chat session (Llama recruiter)
2. `POST /api/vision/session/start` for behavioral analysis
3. WebSocket to `/ws/vision?session_id=<vision-id>` streaming JPEG frames
   from `getUserMedia()` and 44.1 kHz PCM from the AudioContext
4. After each generated question, the UI fires
   `POST /api/vision/session/{id}/question` with the question text so the
   timeline shows where each question landed

Behavioral signals stay client-side (live overlay) until session end, when
the report is generated and saved to `artifacts/vision/reports/`.

## A2A capabilities

`start_session`, `mark_question`, `get_report`. See `agent_card.json`.

## Trained artifacts

| Artifact | Path | Required? |
|---|---|---|
| `face_landmarker.task` | `models/` | yes — visual pipeline returns 503 without it |
| `pose_landmarker.task` | `models/` | yes — same |
| `hand_landmarker.task` | `models/` | optional — gesture analyzer disabled if missing |
| `studentnet.pt` | `weights/` | optional — audio uses untrained head as fallback |
| `visual_mlp.pt`| `weights/` | optional — visual stress label disabled |

The MediaPipe `.task` files ship with the repo. StudentNet has not been
trained yet (see project-level fallback notes). The audio pipeline still
runs Wav2Vec2 and surfaces a probability; only the calibration uses it.

## Performance notes

- Visual inference: ~30 ms / frame on CPU, ~12 ms on GPU (RTX 4090).
- Audio: Wav2Vec2 ~1-3 s on CPU per 1 s window; backgrounded so it does
  not block the receiver loop.
- Frames are dropped if the processor is busy — only the latest one is
  kept (latency over throughput).
