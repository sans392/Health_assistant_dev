"""Сервис эмбеддингов через Ollama /api/embeddings (Phase 2, Issue #22).

Модель: nomic-embed-text (из .env EMBEDDING_MODEL).
Используется для RAG (knowledge_base) и semantic memory.
"""

import logging
import time
from typing import Any, Union

import httpx

from app.config import settings
from app.services.llm_call_logger import llm_call_logger

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Async-клиент для получения эмбеддингов через Ollama /api/embeddings.

    Поддерживает одиночные строки и батчи (поочерёдно — Ollama не поддерживает
    нативный batch в /api/embeddings).
    Retry 1 раз при таймауте.
    """

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout: int = 60,
    ) -> None:
        self._host = (host or settings.ollama_host).rstrip("/")
        self._model = model or settings.embedding_model
        self._timeout = timeout

    def _make_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._host,
            timeout=httpx.Timeout(self._timeout),
        )

    async def embed(self, text: Union[str, list[str]]) -> list[list[float]]:
        """Получить эмбеддинги для строки или списка строк.

        Args:
            text: Одна строка или список строк.

        Returns:
            Список векторов float. Длина == количество входных строк.

        Raises:
            ValueError: При ошибке Ollama после retry.
            httpx.HTTPError: При сетевой ошибке.
        """
        texts = [text] if isinstance(text, str) else list(text)
        results: list[list[float]] = []
        for t in texts:
            vector = await self._embed_single(t)
            results.append(vector)
        return results

    async def _embed_single(self, text: str) -> list[float]:
        """Получить эмбеддинг для одной строки с 1 retry при таймауте."""
        for attempt in range(2):
            try:
                return await self._send_request(text)
            except httpx.TimeoutException as exc:
                if attempt == 0:
                    logger.warning(
                        "Embedding timeout (попытка %d/2), retry... model=%s",
                        attempt + 1,
                        self._model,
                    )
                    continue
                logger.error(
                    "Embedding timeout после 2 попыток: model=%s", self._model
                )
                self._log_call(
                    text=text,
                    payload={"model": self._model, "prompt": text},
                    duration_ms=0,
                    http_status=None,
                    response_body=None,
                    embedding_dim=0,
                    error=f"Timeout после 2 попыток: {exc}",
                )
                raise ValueError(
                    f"Ошибка получения эмбеддинга: таймаут (model={self._model})"
                ) from exc
        # Этой точки не достичь, но для mypy
        raise ValueError("Ошибка embedding: неожиданный выход из цикла retry")

    async def _send_request(self, text: str) -> list[float]:
        """Выполнить HTTP запрос к /api/embeddings."""
        start_ms = time.monotonic() * 1000
        payload = {"model": self._model, "prompt": text}
        http_status: int | None = None

        try:
            async with self._make_client() as client:
                resp = await client.post("/api/embeddings", json=payload)
                http_status = resp.status_code
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            self._log_call(
                text=text,
                payload=payload,
                duration_ms=int(time.monotonic() * 1000 - start_ms),
                http_status=exc.response.status_code,
                response_body=None,
                embedding_dim=0,
                error=f"HTTP {exc.response.status_code}: {exc.response.text[:500]}",
            )
            raise

        duration_ms = time.monotonic() * 1000 - start_ms
        embedding: list[float] = data.get("embedding", [])

        logger.info(
            "Embedding: model=%s text_len=%d dim=%d duration_ms=%.1f",
            self._model,
            len(text),
            len(embedding),
            duration_ms,
        )

        self._log_call(
            text=text,
            payload=payload,
            duration_ms=int(duration_ms),
            http_status=http_status,
            response_body=data,
            embedding_dim=len(embedding),
            error=None,
        )

        return embedding

    def _log_call(
        self,
        text: str,
        payload: dict[str, Any],
        duration_ms: int,
        http_status: int | None,
        response_body: dict[str, Any] | None,
        embedding_dim: int,
        error: str | None,
    ) -> None:
        """Записать embedding-вызов в llm_calls (если per-request трекинг активен).

        В response сохраняем не весь вектор (сотни floats), а краткую сводку
        «[embedding dim=768]» — сам вектор доступен в response_body.
        """
        if response_body is not None and isinstance(response_body.get("embedding"), list):
            # В raw response_body тоже заменяем массив на сводку, чтобы не раздувать хранилище:
            # полную картину (все float'ы) видеть в админке не нужно, достаточно dim.
            safe_body = {k: v for k, v in response_body.items() if k != "embedding"}
            safe_body["embedding_dim"] = embedding_dim
            safe_body["embedding"] = "<vector omitted>"
        else:
            safe_body = response_body

        response_summary = (
            f"[embedding dim={embedding_dim}]" if embedding_dim and error is None else None
        )

        llm_call_logger.record(
            role="embedding",
            model=self._model,
            prompt=text or None,
            response=response_summary,
            prompt_length=len(text or ""),
            response_length=embedding_dim,
            duration_ms=duration_ms,
            endpoint="/api/embeddings",
            stream=False,
            http_status=http_status,
            request_body=payload,
            response_body=safe_body,
            error=error,
        )

    async def check_model_available(self) -> bool:
        """Проверить наличие модели эмбеддингов в Ollama.

        Returns:
            True если модель найдена в /api/tags.
        """
        try:
            async with self._make_client() as client:
                resp = await client.get("/api/tags")
                resp.raise_for_status()
                data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            return any(self._model in m for m in models)
        except Exception as exc:
            logger.warning("Embedding check_model_available error: %s", exc)
            return False


# Глобальный синглтон
embedding_service = EmbeddingService()
