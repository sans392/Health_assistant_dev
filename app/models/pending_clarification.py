"""Модель незавершённого уточняющего диалога (Issue #57).

Одна запись на session_id. Оркестратор сохраняет pending, когда intent
требует обязательные слоты, а intent detection их не извлёк. При следующем
user message `resume_from_clarification` проверяет pending, пытается
дозаполнить слоты ответом пользователя и, если всё найдено, возвращает
полный IntentResult для нормального прохождения пайплайна.

TTL — 3 минуты: если пользователь вернулся позже, pending протухает и
новый запрос обрабатывается как обычный.
"""

from datetime import datetime

from sqlalchemy import JSON, DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class PendingClarification(Base):
    """Незавершённый уточняющий диалог в рамках сессии."""

    __tablename__ = "pending_clarifications"

    # session_id — PK: на сессию допускается только один pending
    session_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # Intent исходного запроса (plan_request / data_query / log_activity / ...)
    intent: Mapped[str] = mapped_column(String(50), nullable=False)
    # Исходный raw_query пользователя, ради которого попросили уточнение
    original_query: Mapped[str] = mapped_column(Text, nullable=False)
    # Уже извлечённые entities (dict) в момент запроса уточнения
    filled_slots: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    # Список имён недостающих слотов, которые ждём от пользователя
    missing_slots: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    # TTL: после expires_at pending игнорируется и удаляется
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
