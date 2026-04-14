"""Модуль формирования обогащённого контекста запроса."""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat import ChatMessage
from app.models.user_profile import UserProfile


@dataclass
class EnrichedQuery:
    """Обогащённый запрос с контекстом сессии и профилем пользователя."""

    raw_text: str
    normalized_text: str
    user_profile: dict | None
    conversation_history: list[dict]       # [{role, content, timestamp}]
    semantic_context: list = field(default_factory=list)   # пусто в MVP
    knowledge_context: list = field(default_factory=list)  # пусто в MVP
    metadata: dict = field(default_factory=dict)           # {timestamp, session_id, user_id}


def _normalize_text(text: str) -> str:
    """Приводит текст к нижнему регистру и нормализует пробелы."""
    lower = text.lower()
    normalized = re.sub(r"\s+", " ", lower).strip()
    return normalized


def _profile_to_dict(profile: UserProfile) -> dict:
    """Сериализует профиль пользователя в словарь."""
    return {
        "user_id": profile.user_id,
        "name": profile.name,
        "age": profile.age,
        "weight_kg": profile.weight_kg,
        "height_cm": profile.height_cm,
        "gender": profile.gender,
        "max_heart_rate": profile.max_heart_rate,
        "resting_heart_rate": profile.resting_heart_rate,
        "training_goals": profile.training_goals,
        "experience_level": profile.experience_level,
        "injuries": profile.injuries,
        "chronic_conditions": profile.chronic_conditions,
        "preferred_sports": profile.preferred_sports,
    }


class ContextBuilder:
    """Формирует EnrichedQuery из текста запроса, истории сессии и профиля пользователя."""

    MAX_MESSAGES = 10

    async def build(
        self,
        query: str,
        session_id: str,
        user_id: str,
        db: AsyncSession,
    ) -> EnrichedQuery:
        """Строит обогащённый запрос.

        Args:
            query: Текст запроса пользователя.
            session_id: Идентификатор текущей сессии чата.
            user_id: Идентификатор пользователя.
            db: Асинхронная сессия SQLAlchemy.

        Returns:
            EnrichedQuery со всеми полями.
        """
        normalized = _normalize_text(query)

        # Загружаем историю сессии (последние MAX_MESSAGES сообщений)
        history = await self._load_history(session_id, db)

        # Загружаем профиль пользователя
        profile_dict = await self._load_profile(user_id, db)

        return EnrichedQuery(
            raw_text=query,
            normalized_text=normalized,
            user_profile=profile_dict,
            conversation_history=history,
            semantic_context=[],
            knowledge_context=[],
            metadata={
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id,
                "user_id": user_id,
            },
        )

    async def _load_history(self, session_id: str, db: AsyncSession) -> list[dict]:
        """Загружает последние MAX_MESSAGES сообщений сессии из БД."""
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(desc(ChatMessage.order_index))
            .limit(self.MAX_MESSAGES)
        )
        result = await db.execute(stmt)
        messages = result.scalars().all()

        # Возвращаем в хронологическом порядке
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.created_at.isoformat(),
            }
            for msg in reversed(messages)
        ]

    async def _load_profile(self, user_id: str, db: AsyncSession) -> dict | None:
        """Загружает профиль пользователя из БД."""
        stmt = select(UserProfile).where(UserProfile.user_id == user_id)
        result = await db.execute(stmt)
        profile = result.scalar_one_or_none()
        if profile is None:
            return None
        return _profile_to_dict(profile)
