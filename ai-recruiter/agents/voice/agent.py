"""
Voice Agent - Audio Processing and Analysis

Purpose:
    Handles all audio processing including speech-to-text transcription
    and voice-based analysis.

Key Responsibilities:
    - Real-time speech-to-text (ASR)
    - Prosody analysis (pitch, tempo, rhythm)
    - Vocal stress and confidence detection
    - Voice activity detection for turn-taking

Models Used:
    - ASR: Whisper or faster-whisper
    - VAD: Silero VAD or WebRTC VAD
    - Prosody: Custom acoustic feature extractor

A2A Tasks Handled:
    - transcribe: Convert audio to text
    - analyze_prosody: Extract voice features
    - detect_stress: Vocal stress indicators
"""
