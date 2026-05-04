"""Async Groq chat-completions client with strict-JSON response_format.

Thin wrapper over httpx so we don't take a dependency on the groq SDK for
one endpoint. One automatic retry on 5xx / timeout, then raise.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Optional

import httpx

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqError(RuntimeError):
    """Any non-recoverable failure from the Groq client."""


class GroqClient:
    """Minimal async client. Reuse the same instance across calls."""

    def __init__(
        self,
        api_key: str,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        if not api_key:
            raise GroqError("GROQ_API_KEY is required")
        self._client = httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(60.0, connect=5.0),
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

    async def chat_json(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float = 0.2,
        timeout: float = 30.0,
        max_retries: int = 1,
    ) -> dict[str, Any]:
        """Send one chat completion and parse the assistant message as JSON.

        response_format is forced to json_object. On 5xx or read timeout, we
        retry up to max_retries with exponential backoff. On JSON parse
        failure or malformed response shape we raise GroqError. The
        4xx-never-retries policy is structural: 4xx responses raise directly
        without entering the retry/sleep path, so the decision does not depend
        on parsing the exception message.
        """
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
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
                # 4xx — never retry
                raise GroqError(f"groq {resp.status_code}: {resp.text[:200]}")

            data = resp.json()
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise GroqError(f"unexpected groq response shape: {exc}") from exc
            try:
                return json.loads(content)
            except json.JSONDecodeError as exc:
                raise GroqError(f"non-json content from groq: {exc}") from exc

        raise GroqError(f"unreachable: {last_exc}")
