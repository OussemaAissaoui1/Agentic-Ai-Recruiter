# WebSocket API Definitions

Message schemas for real-time WebSocket communication.

## Key Files

| File | Purpose |
|------|---------|
| `messages.json` | JSON Schema for WebSocket messages |
| `events.py` | Event type enums and Pydantic models |
| `protocol.md` | WebSocket protocol documentation |

## Message Types

### Client → Server
- `audio_chunk` - Raw audio data from candidate
- `video_frame` - Video frame from candidate
- `text_input` - Text fallback input
- `control` - Session control (pause, resume, end)

### Server → Client
- `avatar_video` - Generated avatar video chunk
- `avatar_audio` - Generated avatar audio chunk
- `transcript` - Real-time transcription
- `status` - Interview status updates
