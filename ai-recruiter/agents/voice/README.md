# Voice Agent (stub)

Reserved for ASR + prosody features that the NLP agent does not already
cover (e.g. faster-whisper transcription of candidate audio for offline
scoring). All capabilities return 501 today.

The behavioral / vocal-stress signal is **not** owned by this agent —
it lives in `agents/vision/` because the multimodal stress-detection
pipeline fuses face, pose, gesture *and* audio under one calibration.

## Routes (`/api/voice`)

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/health` | Reports `state=fallback` until ASR is wired |

## Future

- faster-whisper transcription endpoint.
- Prosody features (pitch, tempo, rhythm) for scoring.
