"""Модель логов пайплайна (для issue #12)."""

import uuid
from datetime import datetime

from sqlalchemy import String, DateTime, Text, Float, JSON, Integer
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class PipelineLog(Base):
    """Лог одного запроса через пайплайн."""

    __tablename__ = "pipeline_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    user_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    user_query: Mapped[str] = mapped_column(Text, nullable=False)
    # intent, safety, routing результаты
    intent: Mapped[str | None] = mapped_column(String(50), nullable=True)
    safety_passed: Mapped[bool | None] = mapped_column(nullable=True)
    route: Mapped[str | None] = mapped_column(String(50), nullable=True)
    # fast_path | standard | blocked
    pipeline_path: Mapped[str | None] = mapped_column(String(20), nullable=True)
    # Имя LLM модели
    llm_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    prompt_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    response_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    llm_duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Дополнительные данные (tool results, etc.)
    extra_data: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
