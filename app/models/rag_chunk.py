"""Модель метаданных RAG-чанков (Phase 2, Issue #23).

Текст чанка хранится и в этой таблице, и в ChromaDB (для embedding search).
embedding_id — ссылка на id в Chroma-коллекции knowledge_base.

Категории (из архитектуры v2):
  physiology_norms | training_principles | recovery_science |
  sport_specific    | nutrition_basics
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class RAGChunk(Base):
    """Метаданные RAG-чанка знаний."""

    __tablename__ = "rag_chunks"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    # Текст чанка (дублируется из ChromaDB для удобства просмотра в админке)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # Категория: physiology_norms / training_principles / recovery_science /
    #            sport_specific / nutrition_basics
    category: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    # Источник (название документа, URL, etc.)
    source: Mapped[str] = mapped_column(String(200), nullable=False)
    # Достоверность: high | medium
    confidence: Mapped[str] = mapped_column(
        String(10), nullable=False, default="medium"
    )
    # Опциональная фильтрация по виду спорта и уровню опыта
    sport_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    experience_level: Mapped[str | None] = mapped_column(String(20), nullable=True)
    # ID документа в ChromaDB коллекции knowledge_base
    embedding_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
