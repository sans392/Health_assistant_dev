"""Модель детального лога LLM-вызовов (Phase 2, Issue #23).

Каждый вызов LLM в пайплайне записывается сюда с привязкой к request_id.
Используется для observability в админке (Issue #34) и в planner loop.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class LLMCall(Base):
    """Лог одного LLM-вызова в рамках запроса пайплайна."""

    __tablename__ = "llm_calls"

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
    # Роль (intent_llm / safety_llm / response / planner)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    # Фактически использованная модель Ollama
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    # Тексты промпта и ответа
    prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    response: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Метрики
    prompt_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    response_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Номер итерации (для planner loop, где LLM вызывается N раз)
    iteration: Mapped[int | None] = mapped_column(Integer, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
