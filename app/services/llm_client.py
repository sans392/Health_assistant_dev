"""OpenAI-compatible LLM client (skeleton).

A thin async wrapper around `/v1/chat/completions` with optional `tools` /
`tool_choice` support. Talks to anything that speaks the OpenAI Chat Completions
protocol — currently Ollama (`/v1`) for development, vLLM for production once
we move off single-node Ollama.

Design notes:
- Provider-neutral DTOs (`LLMMessage`, `ToolCall`, `LLMReply`) so the rest of
  the pipeline does not import provider SDKs.
- Reuses `llm_call_logger` so calls show up in admin diagnostics next to legacy
  Ollama-native calls. The role/model labels survive the migration.
- No streaming yet. The legacy `OllamaClient.chat_stream` keeps serving the
  response role; this client is meant for tool-using planner-style flows where
  we want structured `tool_calls` back.
- No guided-decoding hooks yet. vLLM's `guided_json` / `guided_grammar` will
  hang off a future `extra_body` parameter without changing this surface.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import settings
from app.services.llm_call_logger import llm_call_logger

logger = logging.getLogger(__name__)


@dataclass
class LLMMessage:
    """OpenAI-style chat message.

    `tool_call_id` is set on role="tool" replies; `tool_calls` on role="assistant"
    when the model decided to call tools.
    """

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    def to_wire(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            msg["content"] = self.content
        if self.name is not None:
            msg["name"] = self.name
        if self.tool_call_id is not None:
            msg["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_wire() for tc in self.tool_calls]
        return msg


@dataclass
class ToolCall:
    """Single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]

    def to_wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "ToolCall":
        fn = data.get("function") or {}
        raw_args = fn.get("arguments")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}
        return cls(
            id=str(data.get("id") or ""),
            name=str(fn.get("name") or ""),
            arguments=args,
        )


@dataclass
class LLMReply:
    """Result of a chat completion call."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    model: str = ""
    prompt_length: int = 0
    response_length: int = 0
    duration_ms: float = 0.0
    raw: dict[str, Any] | None = None


class OpenAICompatibleClient:
    """Async client for OpenAI Chat Completions–compatible endpoints.

    Backend selection is just a base URL. Defaults to Ollama's `/v1` so this
    works in the current dev setup; point it at vLLM by changing `base_url`.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: int | None = None,
        role: str = "planner",
    ) -> None:
        host = (base_url or settings.ollama_host).rstrip("/")
        if not host.endswith("/v1"):
            host = f"{host}/v1"
        self._base_url = host
        self._model = model or settings.ollama_model
        # Ollama ignores the key; vLLM accepts any non-empty token by default.
        self._api_key = api_key or "ollama"
        self._timeout = timeout or settings.ollama_timeout
        self._role = role

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout),
            headers={"Authorization": f"Bearer {self._api_key}"},
        )

    async def chat(
        self,
        messages: list[LLMMessage] | list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> LLMReply:
        """Single-shot chat completion.

        Args:
            messages: Either typed `LLMMessage` or already-wire-formatted dicts.
            tools: OpenAI tool specs (`{"type":"function","function":{...}}`).
            tool_choice: "auto" | "none" | "required" | {"type":"function",...}.
            response_format: e.g. `{"type": "json_object"}` for JSON mode.
            extra_body: Backend-specific extras passed through verbatim.
                Reserved slot for vLLM `guided_json`/`guided_grammar` later.
        """
        wire_messages: list[dict[str, Any]] = [
            m.to_wire() if isinstance(m, LLMMessage) else dict(m) for m in messages
        ]
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": wire_messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if response_format is not None:
            payload["response_format"] = response_format
        if extra_body:
            payload.update(extra_body)

        prompt_for_log = " ".join(
            m.get("content") or "" for m in wire_messages if isinstance(m.get("content"), str)
        )

        start_ms = time.monotonic() * 1000
        http_status: int | None = None
        try:
            async with self._client() as client:
                resp = await client.post("/chat/completions", json=payload)
                http_status = resp.status_code
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            duration_ms = int(time.monotonic() * 1000 - start_ms)
            self._log_failure(
                payload=payload,
                prompt_for_log=prompt_for_log,
                error=f"{type(exc).__name__}: {exc}",
                http_status=http_status,
                duration_ms=duration_ms,
            )
            raise

        duration_ms = time.monotonic() * 1000 - start_ms
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        finish_reason = choice.get("finish_reason")
        raw_tool_calls = message.get("tool_calls") or []
        tool_calls = [ToolCall.from_wire(tc) for tc in raw_tool_calls]

        prompt_length = len(prompt_for_log)
        response_length = len(content) + sum(
            len(tc.name) + len(json.dumps(tc.arguments, ensure_ascii=False))
            for tc in tool_calls
        )

        logger.info(
            "LLM(openai) call: model=%s prompt_len=%d response_len=%d "
            "tool_calls=%d duration_ms=%.1f",
            self._model, prompt_length, response_length, len(tool_calls), duration_ms,
        )

        llm_call_logger.record(
            role=self._role,
            model=self._model,
            prompt=prompt_for_log or None,
            response=content or None,
            prompt_length=prompt_length,
            response_length=response_length,
            duration_ms=int(duration_ms),
            endpoint="/v1/chat/completions",
            stream=False,
            http_status=http_status,
            request_body=payload,
            response_body=data,
        )

        return LLMReply(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            model=self._model,
            prompt_length=prompt_length,
            response_length=response_length,
            duration_ms=round(duration_ms, 1),
            raw=data,
        )

    def _log_failure(
        self,
        payload: dict[str, Any],
        prompt_for_log: str,
        error: str,
        http_status: int | None,
        duration_ms: int,
    ) -> None:
        llm_call_logger.record(
            role=self._role,
            model=self._model,
            prompt=prompt_for_log or None,
            response=None,
            prompt_length=len(prompt_for_log or ""),
            response_length=0,
            duration_ms=duration_ms,
            endpoint="/v1/chat/completions",
            stream=False,
            http_status=http_status,
            request_body=payload,
            response_body=None,
            error=error,
        )
