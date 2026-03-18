"""
Avatar Agent - Synthetic Interviewer Generation

Purpose:
    Generates the visual and audio representation of the AI interviewer (Alex).
    Creates realistic talking head video synchronized with generated speech.

Key Responsibilities:
    - Text-to-speech synthesis with natural prosody
    - Lip sync generation from audio
    - Talking head video synthesis
    - Adaptive facial expressions based on context

Models Used:
    - TTS: XTTS, Coqui TTS, or ElevenLabs
    - Lip Sync: Wav2Lip or SadTalker
    - Face Generation: First Order Motion Model

A2A Tasks Handled:
    - generate_speech: TTS from question text
    - generate_video: Full avatar video with lip sync
    - update_expression: Adapt avatar based on context
"""
