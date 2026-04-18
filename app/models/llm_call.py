"""Модель детального лога LLM-вызовов (Phase 2, Issue #23).

Каждый вызов LLM в пайплайне записывается сюда с привязкой к request_id.
Используется для observability в админке (Issue #34) и в planner loop.

В дополнение к prompt/response (полный текст, без обрезки) сохраняются
сырые JSON-payload запроса и ответа Ollama — чтобы админка показывала
полную картину каждого HTTP-вызова (endpoint, options, eval_count и т.д.).
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text
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
    # Роль (intent_llm / safety_llm / response / planner / embedding)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    # Фактически использованная модель Ollama
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    # Endpoint Ollama: /api/generate | /api/chat | /api/embeddings
    endpoint: Mapped[str | None] = mapped_column(String(50), nullable=True)
    # Флаг: использовался ли стриминг ответа
    stream: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    # HTTP status code фактического ответа Ollama (null при сетевой ошибке/timeout)
    http_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Тексты промпта и ответа (полный текст, без обрезки)
    prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    response: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Сырое тело HTTP-запроса (payload, отправленный в Ollama): JSON с model,
    # prompt/messages, options (temperature, num_predict, …), system и др.
    request_body: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    # Сырой JSON ответа от Ollama (done_reason, eval_count, prompt_eval_count,
    # eval_duration, load_duration и т.д.). Для стриминга — агрегированная сводка.
    response_body: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    # Текст ошибки (timeout, HTTP 5xx, JSON parse error и т.п.), если вызов провалился
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Метрики
    prompt_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    response_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Номер итерации (для planner loop, где LLM вызывается N раз)
    iteration: Mapped[int | None] = mapped_column(Integer, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
