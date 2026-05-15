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
import time
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

        # Local view of the server's TPM budget, refreshed from response
        # headers. Lets us pre-sleep when we know we'd 429 instead of paying
        # the 429 round-trip (which on Groq free-tier costs 13–17s per hit).
        # See _wait_for_tpm_budget + _update_rate_state below.
        self._tpm_remaining: Optional[int] = None
        self._tpm_reset_at_monotonic: float = 0.0
        self._tpm_lock = asyncio.Lock()

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
    # TPM budget governor
    # -----------------------------------------------------------------------
    @staticmethod
    def _estimate_payload_tokens(payload: Dict[str, Any]) -> int:
        """Rough char/4 heuristic + a response-budget cushion.

        Groq's tokenizer isn't identical to OpenAI's but the ~4-chars-per-token
        rule is close enough for budgeting (within ~20%). We add 600 tokens of
        headroom for the response so we don't have to model max_tokens precisely.
        """
        try:
            blob = json.dumps(payload, separators=(",", ":"))
        except (TypeError, ValueError):
            return 2000
        return max(400, len(blob) // 4 + 600)

    async def _wait_for_tpm_budget(self, estimated_tokens: int) -> None:
        """Sleep until the server should have enough budget for `estimated_tokens`.

        No-op when:
        - we've never seen a rate-limit header yet (first request of the process)
        - we know we have enough remaining budget
        - the reset time has already passed (the bucket has refilled)
        """
        async with self._tpm_lock:
            if self._tpm_remaining is None or self._tpm_remaining >= estimated_tokens:
                return
            now = time.monotonic()
            wait = self._tpm_reset_at_monotonic - now
            if wait <= 0:
                # Bucket should have refilled by clock time; pessimistically
                # clear our local view and proceed — the next response header
                # will refresh it.
                self._tpm_remaining = None
                return
            log.info(
                "groq tpm low (have %d, need ~%d) — pre-sleeping %.1fs",
                self._tpm_remaining, estimated_tokens, wait,
            )
            # Clear the cached "remaining" so concurrent callers don't all
            # decide to wait sequentially after they wake.
            self._tpm_remaining = None
            sleep_s = min(60.0, wait)
        # Release the lock before sleeping so other tasks can read state.
        await asyncio.sleep(sleep_s)

    def _update_rate_state(
        self, headers: httpx.Headers, *, exhausted: bool = False
    ) -> None:
        """Refresh local TPM view from response headers.

        Called after every response. On 429, `exhausted=True` forces remaining
        to 0 so other in-flight tasks don't race past us into another 429.
        """
        try:
            if exhausted:
                self._tpm_remaining = 0
            else:
                rem_h = headers.get("x-ratelimit-remaining-tokens")
                if rem_h is not None:
                    self._tpm_remaining = max(0, int(float(rem_h)))
            reset_h = (
                headers.get("x-ratelimit-reset-tokens")
                or headers.get("retry-after")
            )
            if reset_h is not None:
                reset_s = max(0.0, float(reset_h))
                self._tpm_reset_at_monotonic = time.monotonic() + reset_s
        except (ValueError, TypeError):
            # Headers are best-effort — never crash the request on a bad header.
            pass

    # -----------------------------------------------------------------------
    # Internal: POST with retry on 5xx / timeout / 429. Other 4xx never retry.
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
        # 429s get extra retry budget because Groq's TPM window slides over a
        # minute — a single sleep usually lets us through.
        rate_limit_retries_left = 3
        attempt = 0
        last_exc: Optional[Exception] = None
        estimated = self._estimate_payload_tokens(payload)
        while True:
            # Proactive throttle: if we already know the budget is too low,
            # sleep until it refills instead of paying a 429 round-trip.
            await self._wait_for_tpm_budget(estimated)

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
                attempt += 1
                continue

            if resp.status_code == 429:
                # Rate-limited. Update local view + back off using server hint.
                self._update_rate_state(resp.headers, exhausted=True)
                if rate_limit_retries_left <= 0:
                    raise GroqError(f"groq 429 after retries: {resp.text[:200]}")
                wait_s = _parse_retry_after(resp.headers, default=20.0)
                log.warning("groq 429 — sleeping %.1fs before retry", wait_s)
                await asyncio.sleep(wait_s)
                rate_limit_retries_left -= 1
                continue

            if resp.status_code >= 500:
                last_exc = GroqError(f"groq {resp.status_code}: {resp.text[:200]}")
                if attempt >= max_retries:
                    raise last_exc
                await asyncio.sleep(0.5 * (2 ** attempt))
                attempt += 1
                continue

            if resp.status_code >= 400:
                # Other 4xx (auth, bad request) never retry
                raise GroqError(f"groq {resp.status_code}: {resp.text[:200]}")

            # Success — refresh budget view for the next caller.
            self._update_rate_state(resp.headers)
            try:
                return resp.json()
            except json.JSONDecodeError as exc:
                raise GroqError(f"non-json wire response: {exc}") from exc


def _parse_retry_after(headers: httpx.Headers, default: float) -> float:
    """Groq returns either Retry-After (seconds or HTTP-date) or
    x-ratelimit-reset-tokens. Honor either. Cap at 60s."""
    ra = headers.get("retry-after")
    if ra:
        try:
            return min(60.0, max(1.0, float(ra)))
        except ValueError:
            pass
    reset = headers.get("x-ratelimit-reset-tokens") or headers.get("x-ratelimit-reset")
    if reset:
        try:
            return min(60.0, max(1.0, float(reset)))
        except ValueError:
            pass
    return default


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
