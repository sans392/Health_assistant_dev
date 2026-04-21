"""Async-клиент для взаимодействия с Ollama API.

Phase 2: мульти-модельный роутинг реализован через LLMRegistry
(app/services/llm_registry.py). Там используй get_client(role) для новых компонентов.

Backward-compat: глобальный `ollama_client` соответствует
`llm_registry.get_client("response")` — существующий код не требует изменений.
"""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import settings
from app.services.llm_call_logger import llm_call_logger

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Результат вызова LLM."""
    content: str
    model: str
    prompt_length: int
    response_length: int
    duration_ms: float


class OllamaClient:
    """Async-клиент для Ollama REST API.

    Поддерживает generate и chat endpoints, health check, список моделей.
    Логирует каждый вызов: модель, длину промпта, длину ответа, время.
    """

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
        role: str = "response",
    ) -> None:
        self._host = (host or settings.ollama_host).rstrip("/")
        self._model = model or settings.ollama_model
        self._timeout = timeout or settings.ollama_timeout
        self._role = role

    @property
    def model(self) -> str:
        return self._model

    def _make_client(self) -> httpx.AsyncClient:
        """Создать httpx AsyncClient с настроенным таймаутом."""
        return httpx.AsyncClient(
            base_url=self._host,
            timeout=httpx.Timeout(self._timeout),
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        format: str | None = None,
    ) -> LLMResponse:
        """Генерация текста через /api/generate endpoint.

        Args:
            prompt: Пользовательский промпт.
            system_prompt: Системный промпт (опционально).
            temperature: Температура генерации (0.0–1.0).
            max_tokens: Максимальная длина ответа в токенах.
            format: Формат ответа Ollama ("json" для JSON mode).

        Returns:
            LLMResponse с содержимым ответа и метриками.

        Raises:
            httpx.HTTPError: При сетевых ошибках после retry.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system_prompt:
            payload["system"] = system_prompt
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        if format:
            payload["format"] = format

        return await self._call_with_retry("/api/generate", payload, prompt)

    async def chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        system_prompts: list[str] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        format: str | None = None,
    ) -> LLMResponse:
        """Генерация ответа через /api/chat endpoint (chat-формат).

        Args:
            messages: История сообщений [{"role": "user"|"assistant", "content": "..."}].
            system_prompt: Одиночный системный промпт (backward compat).
            system_prompts: Список системных промптов — кладутся как отдельные
                role="system" сообщения в начале истории. Удобно, когда нужно
                разделить базовую инструкцию, данные профиля и tool/RAG контекст.
            temperature: Температура генерации.
            max_tokens: Максимальная длина ответа в токенах.
            format: Формат ответа Ollama ("json" для JSON mode).

        Returns:
            LLMResponse с содержимым ответа и метриками.
        """
        msgs: list[dict[str, str]] = []
        if system_prompts:
            for sp in system_prompts:
                if sp:
                    msgs.append({"role": "system", "content": sp})
        elif system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.extend(messages)

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": msgs,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        if format:
            payload["format"] = format

        # Для логирования — объединяем тексты в один промпт
        combined_prompt = " ".join(m.get("content", "") for m in msgs)
        return await self._call_with_retry("/api/chat", payload, combined_prompt)

    async def _call_with_retry(
        self,
        endpoint: str,
        payload: dict[str, Any],
        prompt_for_log: str,
    ) -> LLMResponse:
        """Отправить запрос к Ollama с одним retry при таймауте.

        Args:
            endpoint: API endpoint (/api/generate или /api/chat).
            payload: Тело запроса.
            prompt_for_log: Промпт для подсчёта длины в логе.

        Returns:
            LLMResponse.
        """
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                return await self._send_request(endpoint, payload, prompt_for_log)
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt == 0:
                    logger.warning(
                        "Ollama timeout (attempt %d/2), retrying... endpoint=%s",
                        attempt + 1,
                        endpoint,
                    )
                    continue
                logger.error("Ollama timeout after 2 attempts: endpoint=%s", endpoint)
                self._log_failure(
                    endpoint=endpoint,
                    payload=payload,
                    prompt_for_log=prompt_for_log,
                    error=f"Timeout после 2 попыток: {exc}",
                    http_status=None,
                    duration_ms=0,
                    stream=False,
                )
                raise
            except httpx.HTTPStatusError as exc:
                self._log_failure(
                    endpoint=endpoint,
                    payload=payload,
                    prompt_for_log=prompt_for_log,
                    error=f"HTTP {exc.response.status_code}: {exc.response.text[:500]}",
                    http_status=exc.response.status_code,
                    duration_ms=0,
                    stream=False,
                )
                raise
            except Exception as exc:
                self._log_failure(
                    endpoint=endpoint,
                    payload=payload,
                    prompt_for_log=prompt_for_log,
                    error=f"{type(exc).__name__}: {exc}",
                    http_status=None,
                    duration_ms=0,
                    stream=False,
                )
                raise
        # На случай неожиданного выхода из цикла (не должно случиться)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Неожиданный выход из _call_with_retry")

    async def _send_request(
        self,
        endpoint: str,
        payload: dict[str, Any],
        prompt_for_log: str,
    ) -> LLMResponse:
        """Выполнить HTTP запрос к Ollama и вернуть LLMResponse."""
        start_ms = time.monotonic() * 1000

        async with self._make_client() as client:
            resp = await client.post(endpoint, json=payload)
            http_status = resp.status_code
            resp.raise_for_status()
            data = resp.json()

        duration_ms = time.monotonic() * 1000 - start_ms

        # Извлечь текст ответа (разные поля для generate и chat)
        if endpoint == "/api/chat":
            content = data.get("message", {}).get("content", "")
        else:
            content = data.get("response", "")

        prompt_length = len(prompt_for_log)
        response_length = len(content)

        logger.info(
            "LLM вызов: model=%s endpoint=%s prompt_len=%d response_len=%d duration_ms=%.1f",
            self._model,
            endpoint,
            prompt_length,
            response_length,
            duration_ms,
        )

        llm_call_logger.record(
            role=self._role,
            model=self._model,
            prompt=prompt_for_log or None,
            response=content or None,
            prompt_length=prompt_length,
            response_length=response_length,
            duration_ms=int(duration_ms),
            endpoint=endpoint,
            stream=False,
            http_status=http_status,
            request_body=payload,
            response_body=data,
        )

        return LLMResponse(
            content=content,
            model=self._model,
            prompt_length=prompt_length,
            response_length=response_length,
            duration_ms=round(duration_ms, 1),
        )

    def _log_failure(
        self,
        endpoint: str,
        payload: dict[str, Any],
        prompt_for_log: str,
        error: str,
        http_status: int | None,
        duration_ms: int,
        stream: bool,
    ) -> None:
        """Записать в llm_calls провалившийся вызов (timeout/HTTP-ошибка/другое)."""
        llm_call_logger.record(
            role=self._role,
            model=self._model,
            prompt=prompt_for_log or None,
            response=None,
            prompt_length=len(prompt_for_log or ""),
            response_length=0,
            duration_ms=duration_ms,
            endpoint=endpoint,
            stream=stream,
            http_status=http_status,
            request_body=payload,
            response_body=None,
            error=error,
        )

    async def health_check(self) -> dict[str, Any]:
        """Проверить доступность Ollama и наличие нужной модели.

        Returns:
            Словарь с полями:
            - available: bool — Ollama доступен
            - model: str — имя модели
            - model_loaded: bool | None — модель найдена в списке
            - error: str | None — текст ошибки, если недоступен
        """
        try:
            async with self._make_client() as client:
                resp = await client.get("/api/tags")
                resp.raise_for_status()
                data = resp.json()

            models = [m.get("name", "") for m in data.get("models", [])]
            model_loaded = any(self._model in m for m in models)

            logger.debug("Ollama доступен. Модели: %s", models)
            return {
                "available": True,
                "model": self._model,
                "model_loaded": model_loaded,
                "error": None,
            }
        except httpx.ConnectError as exc:
            logger.warning("Ollama недоступен (ConnectError): %s", exc)
            return {
                "available": False,
                "model": self._model,
                "model_loaded": None,
                "error": "Ollama недоступен: не удалось подключиться",
            }
        except httpx.TimeoutException as exc:
            logger.warning("Ollama health check timeout: %s", exc)
            return {
                "available": False,
                "model": self._model,
                "model_loaded": None,
                "error": "Ollama недоступен: таймаут",
            }
        except Exception as exc:
            logger.error("Ollama health check неожиданная ошибка: %s", exc)
            return {
                "available": False,
                "model": self._model,
                "model_loaded": None,
                "error": f"Ошибка: {exc}",
            }

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> "LLMResponse":
        """Генерация текста через /api/generate с включённым стримингом.

        Вызывает on_token(token) на каждый полученный токен.
        Собирает полный ответ и возвращает его как LLMResponse.

        Args:
            prompt: Пользовательский промпт.
            system_prompt: Системный промпт.
            temperature: Температура генерации.
            max_tokens: Максимальная длина ответа в токенах.
            on_token: Callback — вызывается для каждого токена строки ответа.

        Returns:
            LLMResponse с полным ответом и метриками.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature},
        }
        if system_prompt:
            payload["system"] = system_prompt
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        start_ms = time.monotonic() * 1000
        full_content: list[str] = []
        http_status: int | None = None
        final_chunk: dict[str, Any] | None = None
        chunk_count = 0

        try:
            async with self._make_client() as client:
                async with client.stream("POST", "/api/generate", json=payload) as response:
                    http_status = response.status_code
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        chunk_count += 1
                        token = chunk.get("response", "")
                        if token:
                            full_content.append(token)
                            if on_token is not None:
                                on_token(token)
                        if chunk.get("done"):
                            final_chunk = chunk
                            break
        except httpx.TimeoutException as exc:
            logger.error("Ollama stream timeout: model=%s", self._model)
            self._log_failure(
                endpoint="/api/generate",
                payload=payload,
                prompt_for_log=prompt,
                error=f"Stream timeout: {exc}",
                http_status=http_status,
                duration_ms=int(time.monotonic() * 1000 - start_ms),
                stream=True,
            )
            raise
        except httpx.HTTPStatusError as exc:
            self._log_failure(
                endpoint="/api/generate",
                payload=payload,
                prompt_for_log=prompt,
                error=f"Stream HTTP {exc.response.status_code}",
                http_status=exc.response.status_code,
                duration_ms=int(time.monotonic() * 1000 - start_ms),
                stream=True,
            )
            raise
        except Exception as exc:
            self._log_failure(
                endpoint="/api/generate",
                payload=payload,
                prompt_for_log=prompt,
                error=f"Stream error {type(exc).__name__}: {exc}",
                http_status=http_status,
                duration_ms=int(time.monotonic() * 1000 - start_ms),
                stream=True,
            )
            raise

        content = "".join(full_content)
        duration_ms = time.monotonic() * 1000 - start_ms

        logger.info(
            "LLM stream: model=%s prompt_len=%d response_len=%d duration_ms=%.1f",
            self._model, len(prompt), len(content), duration_ms,
        )

        # Для стриминга как response_body сохраняем финальный chunk с метаданными
        # Ollama (done_reason, eval_count, prompt_eval_count и т.д.) + агрегат.
        response_body: dict[str, Any] = {
            "stream": True,
            "chunks_received": chunk_count,
            "response": content,
        }
        if final_chunk is not None:
            response_body["final_chunk"] = final_chunk

        llm_call_logger.record(
            role=self._role,
            model=self._model,
            prompt=prompt or None,
            response=content or None,
            prompt_length=len(prompt),
            response_length=len(content),
            duration_ms=int(duration_ms),
            endpoint="/api/generate",
            stream=True,
            http_status=http_status,
            request_body=payload,
            response_body=response_body,
        )

        return LLMResponse(
            content=content,
            model=self._model,
            prompt_length=len(prompt),
            response_length=len(content),
            duration_ms=round(duration_ms, 1),
        )

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        system_prompts: list[str] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> "LLMResponse":
        """Генерация ответа через /api/chat с включённым стримингом.

        Вызывает on_token(token) на каждый полученный токен.

        Args:
            messages: История сообщений.
            system_prompt: Одиночный системный промпт.
            system_prompts: Список системных промптов (несколько role=system).
            temperature: Температура генерации.
            max_tokens: Максимальная длина ответа в токенах.
            on_token: Callback для каждого токена.

        Returns:
            LLMResponse с полным ответом и метриками.
        """
        msgs: list[dict[str, str]] = []
        if system_prompts:
            for sp in system_prompts:
                if sp:
                    msgs.append({"role": "system", "content": sp})
        elif system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.extend(messages)

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": msgs,
            "stream": True,
            "options": {"temperature": temperature},
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        combined_prompt = " ".join(m.get("content", "") for m in msgs)
        start_ms = time.monotonic() * 1000
        full_content: list[str] = []
        http_status: int | None = None
        final_chunk: dict[str, Any] | None = None
        chunk_count = 0

        try:
            async with self._make_client() as client:
                async with client.stream("POST", "/api/chat", json=payload) as response:
                    http_status = response.status_code
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        chunk_count += 1
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            full_content.append(token)
                            if on_token is not None:
                                on_token(token)
                        if chunk.get("done"):
                            final_chunk = chunk
                            break
        except httpx.TimeoutException as exc:
            logger.error("Ollama chat stream timeout: model=%s", self._model)
            self._log_failure(
                endpoint="/api/chat",
                payload=payload,
                prompt_for_log=combined_prompt,
                error=f"Chat stream timeout: {exc}",
                http_status=http_status,
                duration_ms=int(time.monotonic() * 1000 - start_ms),
                stream=True,
            )
            raise
        except httpx.HTTPStatusError as exc:
            self._log_failure(
                endpoint="/api/chat",
                payload=payload,
                prompt_for_log=combined_prompt,
                error=f"Chat stream HTTP {exc.response.status_code}",
                http_status=exc.response.status_code,
                duration_ms=int(time.monotonic() * 1000 - start_ms),
                stream=True,
            )
            raise
        except Exception as exc:
            self._log_failure(
                endpoint="/api/chat",
                payload=payload,
                prompt_for_log=combined_prompt,
                error=f"Chat stream error {type(exc).__name__}: {exc}",
                http_status=http_status,
                duration_ms=int(time.monotonic() * 1000 - start_ms),
                stream=True,
            )
            raise

        content = "".join(full_content)
        duration_ms = time.monotonic() * 1000 - start_ms

        logger.info(
            "LLM chat stream: model=%s prompt_len=%d response_len=%d duration_ms=%.1f",
            self._model, len(combined_prompt), len(content), duration_ms,
        )

        response_body: dict[str, Any] = {
            "stream": True,
            "chunks_received": chunk_count,
            "message": {"content": content},
        }
        if final_chunk is not None:
            response_body["final_chunk"] = final_chunk

        llm_call_logger.record(
            role=self._role,
            model=self._model,
            prompt=combined_prompt or None,
            response=content or None,
            prompt_length=len(combined_prompt),
            response_length=len(content),
            duration_ms=int(duration_ms),
            endpoint="/api/chat",
            stream=True,
            http_status=http_status,
            request_body=payload,
            response_body=response_body,
        )

        return LLMResponse(
            content=content,
            model=self._model,
            prompt_length=len(combined_prompt),
            response_length=len(content),
            duration_ms=round(duration_ms, 1),
        )

    async def list_models(self) -> list[str]:
        """Получить список доступных моделей в Ollama.

        Returns:
            Список имён моделей. Пустой список при ошибке.
        """
        try:
            async with self._make_client() as client:
                resp = await client.get("/api/tags")
                resp.raise_for_status()
                data = resp.json()
            return [m.get("name", "") for m in data.get("models", [])]
        except Exception as exc:
            logger.error("Ошибка получения списка моделей: %s", exc)
            return []


# Глобальный экземпляр клиента (singleton)
ollama_client = OllamaClient()
