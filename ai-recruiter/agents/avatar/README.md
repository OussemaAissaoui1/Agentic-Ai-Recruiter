# Avatar Agent

Synthetic interviewer generation for visual and audio representation.

## Purpose

- Generate realistic talking head video of interviewer (Alex)
- Text-to-speech synthesis with natural prosody
- Lip sync synchronization with generated audio
- Adaptive facial expressions based on interview context
- Emotion-responsive avatar behavior

## Key Components

| File | Purpose |
|------|---------|
| `agent.py` | Main Avatar agent with A2A task handler |
| `tts_engine.py` | Text-to-speech synthesis |
| `lip_sync.py` | Audio-to-lip-sync generation |
| `face_generator.py` | Talking head video synthesis |
| `expression_controller.py` | Facial expression adaptation |
| `avatar_renderer.py` | Final video composition and streaming |
| `agent_card.json` | A2A discovery metadata |

## Models Used

- TTS: XTTS / Coqui TTS / ElevenLabs
- Lip Sync: Wav2Lip / SadTalker
- Face Generation: First Order Motion Model

## Data Flows

- **Input from NLP**: Generated question text for TTS + lip sync
- **Input from Vision**: Emotion signals to adapt avatar expressions
- **Output**: Video/audio stream to client
