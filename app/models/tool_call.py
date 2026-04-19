"""Модель детального лога вызовов Tool Executor (Phase 2, Tool Results v2).

Каждый tool-вызов в пайплайне (напрямую через Tool Executor, из Planner loop,
или из Template Plan Executor) записывается сюда с привязкой к request_id.
Используется для observability в админке (страница Pipeline Trace → Tool Results).

В отличие от старого поведения (хранили только имя инструмента в pipeline_logs),
теперь сохраняем полные аргументы вызова, сырой результат и метаданные
(status/error/duration), чтобы админка показывала полную картину.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class ToolCall(Base):
    """Лог одного Tool-вызова в рамках запроса пайплайна."""

    __tablename__ = "tool_calls"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    # Привязка к запросу в pipeline_logs
    request_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("pipeline_logs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    # Имя инструмента: get_activities, rag_retrieve, compute_recovery, …
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    # Источник вызова: tool_executor | planner | template
    source: Mapped[str] = mapped_column(String(20), nullable=False)
    # Номер итерации planner loop (null для tool_executor / template)
    iteration: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Идентификатор шага шаблона (null для tool_executor / planner)
    step_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    # Аргументы вызова (сырой JSON, без обрезки)
    args: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    # Результат вызова (сырой JSON данных, без обрезки)
    result: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    # Успешен ли вызов
    success: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    # Текст ошибки (если вызов провалился)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Длительность вызова в миллисекундах
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
