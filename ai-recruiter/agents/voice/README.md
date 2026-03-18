# Voice Agent

Audio processing agent for speech-to-text and voice analysis.

## Purpose

- Real-time speech-to-text transcription (ASR)
- Prosody analysis (pitch, tempo, rhythm)
- Stress and confidence detection from voice
- Voice activity detection (VAD) for turn-taking
- Generate prosody scores for evaluation

## Key Components

| File | Purpose |
|------|---------|
| `agent.py` | Main Voice agent with A2A task handler |
| `transcriber.py` | ASR engine (Whisper / Conformer) |
| `prosody_analyzer.py` | Pitch, tempo, and rhythm extraction |
| `stress_detector.py` | Vocal stress and confidence analysis |
| `vad.py` | Voice activity detection for turn-taking |
| `audio_processor.py` | Audio preprocessing and normalization |
| `agent_card.json` | A2A discovery metadata |

## Models Used

- ASR: Whisper / faster-whisper
- Prosody: Custom acoustic feature extractor
- VAD: Silero VAD / WebRTC VAD

## Data Flows

- **Input**: Audio stream from client WebSocket
- **Output to NLP**: Candidate transcript (ASR output)
- **Output to Scoring**: Prosody and stress scores
