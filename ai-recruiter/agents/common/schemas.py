"""
Shared Pydantic Schemas

Purpose:
    Common data models used for communication between agents.
    Ensures type safety and validation across the system.

Key Models:
    - InterviewSession: Interview session metadata
    - TranscriptTurn: Single turn in conversation
    - QuestionRequest: Request to generate a question
    - ScorePayload: Evaluation scores from an agent
    - A2ATask: Task request/response for A2A protocol

Usage:
    These models are imported by agents and core modules
    for serialization/deserialization of inter-agent messages.
"""
