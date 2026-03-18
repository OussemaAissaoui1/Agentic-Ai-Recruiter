"""
Vision Agent - Non-Verbal Analysis

Purpose:
    Computer vision agent for analyzing candidate's non-verbal communication
    during the video interview.

Key Responsibilities:
    - Detect and track faces in video frames
    - Classify facial emotions (confidence, nervousness, engagement)
    - Analyze posture and body language
    - Monitor eye contact and attention
    - Generate behavioral scores

Models Used:
    - Face Detection: MediaPipe or MTCNN
    - Emotion Recognition: Custom CNN or FER model
    - Posture Analysis: BlazePose or OpenPose

A2A Tasks Handled:
    - analyze_frame: Process single video frame
    - get_emotion_summary: Aggregate emotions over time window
    - get_attention_score: Eye contact and engagement metrics
"""
