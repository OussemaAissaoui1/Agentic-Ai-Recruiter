"""
LangGraph Edges for Orchestrator

Purpose:
    Defines conditional routing logic between nodes in the state machine.
    Determines the next node based on current state.

Key Routing Logic:
    - After greeting -> start questioning
    - After question -> wait for response
    - After N questions -> move to closing
    - On error -> recovery or end
    - After all scores -> generate report

Usage:
    Edges are defined using LangGraph's add_conditional_edges()
    to create dynamic, state-dependent workflows.
"""
