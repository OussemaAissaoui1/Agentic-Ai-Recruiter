"""
Orchestrator Agent - Central Coordination Layer

Purpose:
    The brain of the interview system. Implemented as a LangGraph state machine
    that coordinates all specialized agents through the interview lifecycle.

Key Responsibilities:
    - Receive interview requests from API Gateway (gRPC)
    - Manage interview session state machine (GREETING -> QUESTIONING -> CLOSING)
    - Dispatch tasks to specialized agents via A2A protocol
    - Aggregate responses and manage conversation flow
    - Handle errors, timeouts, and recovery

LangGraph Integration:
    Uses LangGraph to define a graph-based workflow where:
    - Nodes represent processing steps (e.g., generate_question, analyze_response)
    - Edges define transitions based on state and conditions
    - State persists across the interview session

Dependencies:
    - Dispatches to: NLP, Vision, Voice, Avatar, Scoring agents
    - Uses: core/memory, core/messaging, core/state
"""
