"""Tool Call Logger — накопление и сохранение вызовов tools в таблицу tool_calls.

Архитектура (per-request accumulator через contextvars) — зеркалит
llm_call_logger:
  * orchestrator вызывает tool_call_logger.start() в начале pipeline.
  * Tool Executor / Planner / Template Executor вызывают record() после каждого
    выполненного tool-вызова.
  * В конце pipeline orchestrator вызывает stop() + flush_to_db().

Пример:
    token = tool_call_logger.start()
    # ... pipeline с tool-вызовами ...
    calls = tool_call_logger.stop(token)
    await tool_call_logger.flush_to_db(request_id, calls, db)
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
class ToolCallData:
    """Данные одного Tool-вызова для записи в tool_calls."""

    name: str
    source: str  # tool_executor | planner | template
    args: dict[str, Any] | None = None
    result: Any | None = None
    success: bool | None = None
    error: str | None = None
    duration_ms: int | None = None
    iteration: int | None = None
    step_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


# Контекстная переменная: список накопленных вызовов (None = трекинг не активен)
_current_calls: ContextVar[list[ToolCallData] | None] = ContextVar(
    "_current_tool_calls", default=None,
)


class ToolCallLogger:
    """Аккумулятор Tool-вызовов per-request через contextvars."""

    def start(self) -> Any:
        """Начать трекинг. Возвращает токен для последующего stop()."""
        calls: list[ToolCallData] = []
        return _current_calls.set(calls)

    def stop(self, token: Any) -> list[ToolCallData]:
        """Завершить трекинг. Возвращает накопленные вызовы и сбрасывает контекст."""
        calls = _current_calls.get(None) or []
        _current_calls.reset(token)
        return calls

    def record(
        self,
        name: str,
        source: str,
        args: dict[str, Any] | None = None,
        result: Any | None = None,
        success: bool | None = None,
        error: str | None = None,
        duration_ms: int | None = None,
        iteration: int | None = None,
        step_id: str | None = None,
    ) -> None:
        """Добавить запись о Tool-вызове (если трекинг активен)."""
        calls = _current_calls.get(None)
        if calls is None:
            return
        calls.append(ToolCallData(
            name=name,
            source=source,
            args=args,
            result=result,
            success=success,
            error=error,
            duration_ms=int(duration_ms) if duration_ms is not None else None,
            iteration=iteration,
            step_id=step_id,
        ))

    async def flush_to_db(
        self,
        request_id: str,
        calls: list[ToolCallData],
        db: Any,  # AsyncSession
    ) -> None:
        """Записать накопленные вызовы в таблицу tool_calls."""
        if not calls:
            return
        try:
            from app.models.tool_call import ToolCall

            for call_data in calls:
                db.add(ToolCall(
                    id=str(uuid.uuid4()),
                    request_id=request_id,
                    name=call_data.name,
                    source=call_data.source,
                    iteration=call_data.iteration,
                    step_id=call_data.step_id,
                    args=call_data.args,
                    result=call_data.result,
                    success=call_data.success,
                    error=call_data.error,
                    duration_ms=call_data.duration_ms,
                    timestamp=call_data.timestamp,
                ))
            await db.commit()
            logger.debug(
                "ToolCallLogger: записано %d вызовов для request_id=%s",
                len(calls),
                request_id,
            )
        except Exception as exc:
            logger.error(
                "ToolCallLogger: ошибка записи в БД request_id=%s: %s",
                request_id,
                exc,
                exc_info=True,
            )
            await db.rollback()


# Глобальный синглтон
tool_call_logger = ToolCallLogger()
