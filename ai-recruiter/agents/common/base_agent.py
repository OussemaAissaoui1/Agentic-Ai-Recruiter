"""
Base Agent Abstract Class

Purpose:
    Defines the common interface and lifecycle methods that ALL agents must implement.
    Ensures consistency across the Orchestrator, NLP, Vision, Voice, Avatar, and Scoring agents.

Key Responsibilities:
    - Define abstract methods for A2A task handling
    - Provide common initialization (logging, metrics, health checks)
    - Implement shared utilities (retry logic, error handling)

Usage:
    All agent implementations (e.g., NLPAgent, VisionAgent) inherit from this class
    and implement the abstract methods.

Example:
    class NLPAgent(BaseAgent):
        async def handle_task(self, task: A2ATask) -> A2ATaskResult:
            # Agent-specific implementation
            pass
"""
