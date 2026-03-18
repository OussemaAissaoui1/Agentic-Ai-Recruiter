"""
A2A Protocol Client

Purpose:
    Client implementation for the Agent-to-Agent (A2A) communication protocol.
    Used by the Orchestrator to dispatch tasks to specialized agents.

Key Responsibilities:
    - Discover agents via agent_card.json endpoints
    - Send tasks to agents (tasks/send endpoint)
    - Subscribe to streaming task results (tasks/sendSubscribe)
    - Handle task lifecycle (submitted, working, completed, failed)

Protocol:
    Implements the A2A specification for inter-agent communication.
    See: api/a2a/README.md for protocol details.

Usage:
    client = A2AClient()
    result = await client.send_task(agent_url, task)
"""
