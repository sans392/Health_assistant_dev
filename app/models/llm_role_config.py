"""Модель конфигурации LLM-ролей (Phase 2, Issue #23).

Persistent хранение выбранной модели для каждой роли пайплайна.
Одна запись на роль. Меняется через админку (/admin/llm, Issue #35).
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class LLMRoleConfig(Base):
    """Конфиг модели для роли LLM (persistent override).

    Роли: intent_llm | safety_llm | response | planner
    """

    __tablename__ = "llm_role_config"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    # Уникальная роль (intent_llm / safety_llm / response / planner)
    role: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)
    # Имя модели Ollama (например qwen2.5:14b)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
