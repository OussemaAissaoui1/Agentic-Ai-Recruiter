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


@pytest.fixture
def fake_transport_4xx():
    calls = {"n": 0}
    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, text="invalid api key")
    handler.calls = calls
    return httpx.MockTransport(handler), calls


async def test_chat_json_does_not_retry_on_4xx(fake_transport_4xx):
    transport, calls = fake_transport_4xx
    client = GroqClient(api_key="test-key", transport=transport)
    with pytest.raises(GroqError) as ei:
        await client.chat_json(system="s", user="u", model="m-1", timeout=5.0)
    assert "401" in str(ei.value)
    assert calls["n"] == 1, "4xx must not retry"
    await client.aclose()


async def test_chat_json_raises_on_non_json_content():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "this is not json"}}]},
        )
    transport = httpx.MockTransport(handler)
    client = GroqClient(api_key="test-key", transport=transport)
    with pytest.raises(GroqError) as ei:
        await client.chat_json(system="s", user="u", model="m-1", timeout=5.0)
    assert "non-json" in str(ei.value).lower()
    await client.aclose()


async def test_chat_json_raises_on_malformed_response_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        # No "choices" key — caller must get a GroqError, not a KeyError leak
        return httpx.Response(200, json={"oops": True})
    transport = httpx.MockTransport(handler)
    client = GroqClient(api_key="test-key", transport=transport)
    with pytest.raises(GroqError):
        await client.chat_json(system="s", user="u", model="m-1", timeout=5.0)
    await client.aclose()


async def test_chat_json_retries_on_timeout():
    calls = {"n": 0}
    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("simulated timeout", request=request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok": true}'}}]},
        )
    transport = httpx.MockTransport(handler)
    client = GroqClient(api_key="test-key", transport=transport)
    out = await client.chat_json(system="s", user="u", model="m-1", timeout=5.0)
    assert out == {"ok": True}
    assert calls["n"] == 2
    await client.aclose()
