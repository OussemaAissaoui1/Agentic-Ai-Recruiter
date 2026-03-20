"""
LangGraph Nodes for Orchestrator

Purpose:
    Defines the processing nodes in the orchestrator's state machine graph.
    Each node performs a specific action and updates the state.

Key Nodes:
    - start_interview: Initialize session, dispatch greeting
    - generate_question: Request NLP agent to generate next question
    - process_response: Handle candidate answer, request analysis
    - evaluate_scores: Aggregate scores from all agents
    - end_interview: Finalize session, generate report

Usage:
    Nodes are registered with the LangGraph StateGraph and connected
    via conditional edges defined in edges.py.
"""
