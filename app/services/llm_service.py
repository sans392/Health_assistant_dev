"""Async-клиент для взаимодействия с Ollama API.

В MVP используется одна модель для всех задач.
Multi-model routing отложен на v2.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import settings

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
    ) -> None:
        self._host = (host or settings.ollama_host).rstrip("/")
        self._model = model or settings.ollama_model
        self._timeout = timeout or settings.ollama_timeout

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
    ) -> LLMResponse:
        """Генерация текста через /api/generate endpoint.

        Args:
            prompt: Пользовательский промпт.
            system_prompt: Системный промпт (опционально).
            temperature: Температура генерации (0.0–1.0).
            max_tokens: Максимальная длина ответа в токенах.

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

        return await self._call_with_retry("/api/generate", payload, prompt)

    async def chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Генерация ответа через /api/chat endpoint (chat-формат).

        Args:
            messages: История сообщений [{"role": "user"|"assistant", "content": "..."}].
            system_prompt: Системный промпт (опционально).
            temperature: Температура генерации.

        Returns:
            LLMResponse с содержимым ответа и метриками.
        """
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.extend(messages)

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": msgs,
            "stream": False,
            "options": {"temperature": temperature},
        }

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
        for attempt in range(2):
            try:
                return await self._send_request(endpoint, payload, prompt_for_log)
            except httpx.TimeoutException as exc:
                if attempt == 0:
                    logger.warning(
                        "Ollama timeout (attempt %d/2), retrying... endpoint=%s",
                        attempt + 1,
                        endpoint,
                    )
                    continue
                logger.error("Ollama timeout after 2 attempts: endpoint=%s", endpoint)
                raise

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

        return LLMResponse(
            content=content,
            model=self._model,
            prompt_length=prompt_length,
            response_length=response_length,
            duration_ms=round(duration_ms, 1),
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
