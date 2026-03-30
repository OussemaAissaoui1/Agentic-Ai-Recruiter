"""
Orchestrator State Schema

Purpose:
    Defines the TypedDict state schema for the LangGraph state machine.
    This state is passed between nodes and persisted across the interview.

Key Fields:
    - session_id: Unique interview session identifier
    - phase: Current interview phase (GREETING, QUESTIONING, etc.)
    - messages: Conversation history
    - candidate_info: Candidate metadata (name, role)
    - scores: Accumulated scores from agents
    - pending_tasks: Tasks dispatched but not yet completed

Usage:
    Used by LangGraph to type-check state transitions and ensure
    all nodes receive and return consistent state.
"""
