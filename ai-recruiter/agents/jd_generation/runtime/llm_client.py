"""Groq client with tool-calling.

Mirrors `agents/scoring/llm_client.py` (httpx + retries + JSON
response_format) and adds `chat_with_tools` for the agent loop. Groq's API
is OpenAI-compatible for both the JSON-mode and tools surfaces.

We never import the official `openai` or `anthropic` SDK — the parent
project's policy is one tiny httpx wrapper per upstream service. Easier to
audit, fewer transitive deps, faster cold start.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

log = logging.getLogger("agents.jd_generation.runtime.llm_client")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqError(RuntimeError):
    """Any non-recoverable failure from the Groq client."""


# ---------------------------------------------------------------------------
# Normalized tool-call shape
# ---------------------------------------------------------------------------
@dataclass
class ToolCall:
    """A single tool call requested by the model.

    `arguments` is the PARSED dict — we lenient-parse the JSON string the
    model returns. Llama-class models occasionally emit malformed JSON; we
    log the raw string but surface an empty dict so the caller can decide
    whether to abort or recover.
    """
    id:        str
    name:      str
    arguments: Dict[str, Any]
    raw_arguments: str = ""


@dataclass
class ChatResult:
    """Normalized chat response.

    `content` is None when the model returned tool_calls without prose.
    `tool_calls` is empty when the model returned a final answer.
    """
    content:       Optional[str]
    tool_calls:    List[ToolCall]
    finish_reason: str
    usage:         Dict[str, Any] = field(default_factory=dict)
    raw:           Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class GroqClient:
    """Async Groq chat-completions client.

    Reuse a single instance for the lifetime of the process. Calls
    `chat_with_tools` for the agent loop, `chat_json` for one-shot
    JSON tasks (e.g. rejection-category classification).
    """

    def __init__(
        self,
        api_key: str,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise GroqError("GROQ_API_KEY is required")
        self._client = httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(timeout, connect=5.0),
        )
        self._api_key = api_key

    @classmethod
    def from_env(cls) -> "GroqClient":
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise GroqError("GROQ_API_KEY is not set")
        return cls(api_key=key)

    async def aclose(self) -> None:
        await self._client.aclose()

    # -----------------------------------------------------------------------
    # Tool-calling
    # -----------------------------------------------------------------------
    async def chat_with_tools(
        self,
        *,
        model:        str,
        messages:     List[Dict[str, Any]],
        tools:        List[Dict[str, Any]],
        tool_choice:  str = "auto",
        temperature:  float = 0.0,
        max_tokens:   Optional[int] = None,
        timeout:      float = 60.0,
        max_retries:  int = 1,
    ) -> ChatResult:
        """Single tool-using turn. Caller orchestrates the loop.

        Returns a `ChatResult`. If the model issued tool calls, `tool_calls`
        is populated; if it returned a final answer, `content` is set and
        `tool_calls` is empty.
        """
        payload: Dict[str, Any] = {
            "model":        model,
            "messages":     messages,
            "tools":        tools,
            "tool_choice":  tool_choice,
            "temperature":  temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        data = await self._post_with_retry(payload, timeout=timeout, max_retries=max_retries)
        return _parse_chat_response(data)

    # -----------------------------------------------------------------------
    # JSON-only one-shot (rejection classifier etc.)
    # -----------------------------------------------------------------------
    async def chat_json(
        self,
        *,
        system:      str,
        user:        str,
        model:       str,
        temperature: float = 0.0,
        timeout:     float = 30.0,
        max_retries: int = 1,
    ) -> Dict[str, Any]:
        """Force a JSON-object response. Mirrors scoring/llm_client.chat_json."""
        payload = {
            "model":          model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "temperature":    temperature,
            "response_format": {"type": "json_object"},
        }
        data = await self._post_with_retry(payload, timeout=timeout, max_retries=max_retries)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GroqError(f"unexpected groq response shape: {exc}") from exc
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise GroqError(f"non-json content from groq: {exc}") from exc

    # -----------------------------------------------------------------------
    # Internal: POST with retry on 5xx / timeout. 4xx never retries.
    # -----------------------------------------------------------------------
    async def _post_with_retry(
        self,
        payload:     Dict[str, Any],
        *,
        timeout:     float,
        max_retries: int,
    ) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type":  "application/json",
        }
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                resp = await self._client.post(
                    GROQ_URL, json=payload, headers=headers, timeout=timeout,
                )
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt >= max_retries:
                    raise GroqError(
                        f"timeout after {max_retries + 1} attempts: {exc}"
                    ) from exc
                await asyncio.sleep(0.5 * (2 ** attempt))
                continue

            if resp.status_code >= 500:
                last_exc = GroqError(f"groq {resp.status_code}: {resp.text[:200]}")
                if attempt >= max_retries:
                    raise last_exc
                await asyncio.sleep(0.5 * (2 ** attempt))
                continue

            if resp.status_code >= 400:
                # 4xx never retries
                raise GroqError(f"groq {resp.status_code}: {resp.text[:200]}")

            try:
                return resp.json()
            except json.JSONDecodeError as exc:
                raise GroqError(f"non-json wire response: {exc}") from exc

        raise GroqError(f"unreachable: {last_exc}")


# ---------------------------------------------------------------------------
# Response parsing — defensive against malformed tool-call JSON
# ---------------------------------------------------------------------------
def _parse_chat_response(data: Dict[str, Any]) -> ChatResult:
    """Pull content + tool_calls out of an OpenAI-shaped response.

    The OpenAI/Groq shape:
        data["choices"][0]["message"]["content"]      str | None
        data["choices"][0]["message"]["tool_calls"]  list[{
            "id": str,
            "type": "function",
            "function": {"name": str, "arguments": str},  # arguments is JSON-string
        }]
        data["choices"][0]["finish_reason"]           "stop" | "tool_calls" | ...

    Defensive: lenient JSON parsing of `arguments`. If it fails, we keep
    the raw string and set arguments to an empty dict; the caller can decide
    to abort or recover.
    """
    try:
        choice = data["choices"][0]
        msg = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise GroqError(f"unexpected groq response shape: {exc}") from exc

    content: Optional[str] = msg.get("content")
    finish_reason: str = choice.get("finish_reason", "stop") or "stop"
    tool_calls: List[ToolCall] = []

    for tc in msg.get("tool_calls") or []:
        if tc.get("type") != "function":
            continue
        fn = tc.get("function", {}) or {}
        name = fn.get("name", "")
        raw_args = fn.get("arguments", "") or ""
        parsed_args: Dict[str, Any] = {}
        if raw_args:
            try:
                parsed_args = json.loads(raw_args)
                if not isinstance(parsed_args, dict):
                    log.warning("tool_call %s: arguments parsed but not a dict; treating as empty",
                                name)
                    parsed_args = {}
            except json.JSONDecodeError:
                log.warning("tool_call %s: arguments not JSON; raw=%r", name, raw_args[:200])
                parsed_args = {}
        tool_calls.append(ToolCall(
            id=tc.get("id", ""),
            name=name,
            arguments=parsed_args,
            raw_arguments=raw_args,
        ))

    usage = data.get("usage", {}) or {}
    return ChatResult(
        content=content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage,
        raw=data,
    )
