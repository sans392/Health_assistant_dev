"""LLM Call Logger — накопление и сохранение вызовов LLM в таблицу llm_calls.

Phase 2, Issue #31.

Архитектура (per-request accumulator через contextvars):
  * orchestrator вызывает llm_call_logger.start() в начале pipeline.
  * OllamaClient._send_request() вызывает llm_call_logger.record() после каждого вызова.
  * В конце pipeline orchestrator вызывает llm_call_logger.stop() + flush_to_db().

Пример:
    token = llm_call_logger.start()
    # ... pipeline с LLM вызовами ...
    calls = llm_call_logger.stop(token)
    await llm_call_logger.flush_to_db(request_id, calls, db)
"""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LLMCallData:
    """Данные одного LLM-вызова для записи в llm_calls."""

    role: str
    model: str
    prompt: str | None
    response: str | None
    prompt_length: int
    response_length: int
    duration_ms: int
    iteration: int | None = None
    # Полная сырая картина HTTP-вызова (см. llm_call.py)
    endpoint: str | None = None
    stream: bool | None = None
    http_status: int | None = None
    request_body: dict[str, Any] | None = None
    response_body: dict[str, Any] | None = None
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


# Контекстная переменная: список накопленных вызовов (None = трекинг не активен)
_current_calls: ContextVar[list[LLMCallData] | None] = ContextVar(
    "_current_calls", default=None,
)


class LLMCallLogger:
    """Аккумулятор LLM-вызовов per-request через contextvars."""

    def start(self) -> Any:
        """Начать трекинг. Возвращает токен для последующего stop()."""
        calls: list[LLMCallData] = []
        return _current_calls.set(calls)

    def stop(self, token: Any) -> list[LLMCallData]:
        """Завершить трекинг. Возвращает накопленные вызовы и сбрасывает контекст."""
        calls = _current_calls.get(None) or []
        _current_calls.reset(token)
        return calls

    def record(
        self,
        role: str,
        model: str,
        prompt: str | None,
        response: str | None,
        prompt_length: int,
        response_length: int,
        duration_ms: int,
        iteration: int | None = None,
        endpoint: str | None = None,
        stream: bool | None = None,
        http_status: int | None = None,
        request_body: dict[str, Any] | None = None,
        response_body: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Добавить запись о LLM-вызове (если трекинг активен)."""
        calls = _current_calls.get(None)
        if calls is None:
            return
        calls.append(LLMCallData(
            role=role,
            model=model,
            prompt=prompt,
            response=response,
            prompt_length=prompt_length,
            response_length=response_length,
            duration_ms=int(duration_ms),
            iteration=iteration,
            endpoint=endpoint,
            stream=stream,
            http_status=http_status,
            request_body=request_body,
            response_body=response_body,
            error=error,
        ))

    def build_role_usage(self, calls: list[LLMCallData]) -> dict[str, int]:
        """Сформировать сводку {role: count} из накопленных вызовов."""
        usage: dict[str, int] = {}
        for call in calls:
            usage[call.role] = usage.get(call.role, 0) + 1
        return usage

    async def flush_to_db(
        self,
        request_id: str,
        calls: list[LLMCallData],
        db: Any,  # AsyncSession
    ) -> None:
        """Записать накопленные вызовы в таблицу llm_calls."""
        if not calls:
            return
        try:
            from app.models.llm_call import LLMCall

            for call_data in calls:
                db.add(LLMCall(
                    id=str(uuid.uuid4()),
                    request_id=request_id,
                    role=call_data.role,
                    model=call_data.model,
                    prompt=call_data.prompt,
                    response=call_data.response,
                    prompt_length=call_data.prompt_length,
                    response_length=call_data.response_length,
                    duration_ms=call_data.duration_ms,
                    iteration=call_data.iteration,
                    endpoint=call_data.endpoint,
                    stream=call_data.stream,
                    http_status=call_data.http_status,
                    request_body=call_data.request_body,
                    response_body=call_data.response_body,
                    error=call_data.error,
                    timestamp=call_data.timestamp,
                ))
            await db.commit()
            logger.debug(
                "LLMCallLogger: записано %d вызовов для request_id=%s",
                len(calls),
                request_id,
            )
        except Exception as exc:
            logger.error(
                "LLMCallLogger: ошибка записи в БД request_id=%s: %s",
                request_id,
                exc,
                exc_info=True,
            )
            await db.rollback()


# Глобальный синглтон
llm_call_logger = LLMCallLogger()
