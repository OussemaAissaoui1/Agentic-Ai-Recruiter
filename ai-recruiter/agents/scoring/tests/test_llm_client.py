import os

import httpx
import pytest

from agents.scoring.llm_client import GroqClient, GroqError


@pytest.fixture
def fake_transport_ok():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-key"
        body = request.read().decode()
        assert '"response_format"' in body
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"hello": "world"}'}}]},
        )
    return httpx.MockTransport(handler)


@pytest.fixture
def fake_transport_5xx_then_ok():
    calls = {"n": 0}
    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="overloaded")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok": true}'}}]},
        )
    return httpx.MockTransport(handler)


@pytest.fixture
def fake_transport_always_5xx():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")
    return httpx.MockTransport(handler)


async def test_chat_json_returns_parsed_dict(fake_transport_ok):
    client = GroqClient(api_key="test-key", transport=fake_transport_ok)
    out = await client.chat_json(
        system="be precise", user="say hi", model="m-1", timeout=5.0,
    )
    assert out == {"hello": "world"}
    await client.aclose()


async def test_chat_json_retries_once_on_5xx(fake_transport_5xx_then_ok):
    client = GroqClient(api_key="test-key", transport=fake_transport_5xx_then_ok)
    out = await client.chat_json(system="s", user="u", model="m-1", timeout=5.0)
    assert out == {"ok": True}
    await client.aclose()


async def test_chat_json_raises_after_retry_exhausted(fake_transport_always_5xx):
    client = GroqClient(api_key="test-key", transport=fake_transport_always_5xx)
    with pytest.raises(GroqError):
        await client.chat_json(system="s", user="u", model="m-1", timeout=5.0)
    await client.aclose()


def test_groqclient_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(GroqError):
        GroqClient.from_env()
