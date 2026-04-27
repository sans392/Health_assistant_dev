"""Tests for OpenAICompatibleClient (app/services/llm_client.py).

We use httpx.MockTransport so the client speaks real /v1/chat/completions wire
format without an actual server.
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.services import llm_client as llm_client_module
from app.services.llm_client import (
    LLMMessage,
    OpenAICompatibleClient,
    ToolCall,
)


def _install_mock_transport(
    monkeypatch: pytest.MonkeyPatch,
    response: dict | None = None,
    capture: dict | None = None,
) -> None:
    """Patch OpenAICompatibleClient._client to use an httpx MockTransport."""

    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture["url"] = str(request.url)
            capture["headers"] = dict(request.headers)
            capture["body"] = json.loads(request.content)
        return httpx.Response(200, json=response or _ok_payload())

    transport = httpx.MockTransport(handler)

    def _client(self: OpenAICompatibleClient) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout),
            headers={"Authorization": f"Bearer {self._api_key}"},
            transport=transport,
        )

    monkeypatch.setattr(OpenAICompatibleClient, "_client", _client)


def _ok_payload(content: str = "ok", tool_calls: list[dict] | None = None) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "qwen2.5:7b",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
    }


@pytest.fixture(autouse=True)
def _silence_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip llm_call_logger DB writes in unit tests."""
    monkeypatch.setattr(
        llm_client_module.llm_call_logger, "record", lambda *a, **kw: None
    )


class TestBaseUrl:
    def test_appends_v1_when_missing(self) -> None:
        c = OpenAICompatibleClient(base_url="http://ollama:11434", model="m")
        assert c.base_url == "http://ollama:11434/v1"

    def test_keeps_v1_when_already_present(self) -> None:
        c = OpenAICompatibleClient(base_url="http://vllm:8000/v1", model="m")
        assert c.base_url == "http://vllm:8000/v1"


@pytest.mark.asyncio
async def test_chat_sends_openai_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    capture: dict = {}
    _install_mock_transport(monkeypatch, capture=capture)
    client = OpenAICompatibleClient(base_url="http://x/v1", model="qwen2.5:7b")

    reply = await client.chat(
        messages=[
            LLMMessage(role="system", content="be concise"),
            LLMMessage(role="user", content="hi"),
        ],
        temperature=0.2,
    )

    assert capture["url"].endswith("/v1/chat/completions")
    body = capture["body"]
    assert body["model"] == "qwen2.5:7b"
    assert body["temperature"] == 0.2
    assert body["stream"] is False
    assert body["messages"][0] == {"role": "system", "content": "be concise"}
    assert body["messages"][1] == {"role": "user", "content": "hi"}

    assert reply.content == "ok"
    assert reply.finish_reason == "stop"
    assert reply.tool_calls == []


@pytest.mark.asyncio
async def test_chat_passes_tools_and_tool_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    capture: dict = {}
    _install_mock_transport(monkeypatch, capture=capture)
    client = OpenAICompatibleClient(base_url="http://x", model="m")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "ping",
                "description": "check health",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    await client.chat(
        messages=[LLMMessage(role="user", content="check")],
        tools=tools,
        tool_choice="auto",
    )

    body = capture["body"]
    assert body["tools"] == tools
    assert body["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_chat_parses_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _ok_payload(
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_activities",
                    "arguments": json.dumps(
                        {"date_from": "2026-04-01", "date_to": "2026-04-07"}
                    ),
                },
            }
        ],
    )
    _install_mock_transport(monkeypatch, response=payload)
    client = OpenAICompatibleClient(base_url="http://x", model="m")

    reply = await client.chat(messages=[LLMMessage(role="user", content="acts?")])

    assert len(reply.tool_calls) == 1
    tc = reply.tool_calls[0]
    assert isinstance(tc, ToolCall)
    assert tc.id == "call_1"
    assert tc.name == "get_activities"
    assert tc.arguments == {"date_from": "2026-04-01", "date_to": "2026-04-07"}


@pytest.mark.asyncio
async def test_extra_body_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """vLLM-specific knobs (guided_json, etc.) ride along via extra_body."""
    capture: dict = {}
    _install_mock_transport(monkeypatch, capture=capture)
    client = OpenAICompatibleClient(base_url="http://x", model="m")

    await client.chat(
        messages=[LLMMessage(role="user", content="x")],
        extra_body={"guided_json": {"type": "object"}},
    )
    assert capture["body"]["guided_json"] == {"type": "object"}


@pytest.mark.asyncio
async def test_response_format_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    capture: dict = {}
    _install_mock_transport(monkeypatch, capture=capture)
    client = OpenAICompatibleClient(base_url="http://x", model="m")

    await client.chat(
        messages=[LLMMessage(role="user", content="x")],
        response_format={"type": "json_object"},
    )
    assert capture["body"]["response_format"] == {"type": "json_object"}
