"""Recruit agent — backend for the HireFlow frontend.

Provides CRUD over jobs/applications/notifications/interviews plus
AI-powered JD generation that streams tokens via SSE.
"""

from __future__ import annotations

from .agent import RecruitAgent

__all__ = ["RecruitAgent"]
