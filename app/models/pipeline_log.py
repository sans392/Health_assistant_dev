"""Модель логов пайплайна (issue #12)."""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class PipelineLog(Base):
    """Лог одного запроса через пайплайн.

    Каждый запрос получает уникальный request_id и все этапы обработки
    логируются с привязкой к нему.
    """

    __tablename__ = "pipeline_logs"

    # Идентификатор запроса (request_id)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    session_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    raw_query: Mapped[str] = mapped_column(Text, nullable=False)

    # Intent и safety результаты
    intent: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    intent_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    route: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    fast_path: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    safety_level: Mapped[str | None] = mapped_column(String(20), nullable=True)

    # Вызванные tools и модули обработки данных
    tools_called: Mapped[list | None] = mapped_column(JSON, nullable=True)
    modules_used: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # LLM метрики
    llm_model_used: Mapped[str | None] = mapped_column(String(100), nullable=True)
    llm_calls_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Производительность
    total_duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    response_length: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Ошибки и текст ответа (для дебага в admin panel)
    errors: Mapped[list | None] = mapped_column(JSON, nullable=True)
    response_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Phase 2 (Issue #23): observability
    # Список id RAG-чанков, использованных в запросе
    rag_chunks_used: Mapped[list | None] = mapped_column(JSON, nullable=True)
    # Хронология стадий: [{stage, start_ms, duration_ms}, ...]
    stage_trace: Mapped[list | None] = mapped_column(JSON, nullable=True)
    # Количество LLM-вызовов по ролям: {role: count}
    llm_role_usage: Mapped[dict | None] = mapped_column(JSON, nullable=True)
