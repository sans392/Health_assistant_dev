"""Memory Update — асинхронное обновление памяти после обработки запроса.

Phase 2, Issue #32. Три уровня:
  1. Short-term (история сессии) — уже сохраняется в orchestrator._save_messages,
     здесь не дублируется.
  2. Long-term (факты профиля) — rule-based extraction из entities:
       - body_part → profile.injuries
       - goal      → profile.training_goals
       - sport_type → profile.preferred_sports
     TODO v3: LLM-based fact extraction
  3. Semantic — эмбеддинг пары (query, response) через semantic_memory.remember.

Запускается через asyncio.create_task() — не блокирует delivery ответа.
Ошибки логируются, но не влияют на ответ пользователю.
"""

from __future__ import annotations

import logging
from datetime import datetime

from app.db import AsyncSessionLocal
from app.services.semantic_memory import semantic_memory

logger = logging.getLogger(__name__)


class MemoryUpdater:
    """Асинхронное обновление памяти после генерации ответа.

    Создаёт собственную DB-сессию, чтобы не зависеть от сессии пайплайна
    (запускается как фоновая задача через asyncio.create_task).
    """

    async def update(
        self,
        user_id: str,
        session_id: str,
        request_id: str | None,
        query: str,
        response: str,
        intent: str = "",
        entities: dict | None = None,
    ) -> None:
        """Обновить все уровни памяти после ответа.

        Args:
            user_id: Идентификатор пользователя.
            session_id: Идентификатор сессии (для short-term).
            request_id: Идентификатор запроса (для semantic memory).
            query: Исходный запрос пользователя.
            response: Ответ ассистента.
            intent: Тип намерения из IntentResult.
            entities: Извлечённые сущности из IntentResult.
        """
        ents = entities or {}

        # 2. Long-term: rule-based обновление профиля
        await self._update_long_term(user_id=user_id, entities=ents)

        # 3. Semantic memory
        await self._update_semantic(
            user_id=user_id,
            request_id=request_id,
            query=query,
            response=response,
        )

    async def _update_long_term(
        self,
        user_id: str,
        entities: dict,
    ) -> None:
        """Rule-based extraction фактов из entities → обновление UserProfile.

        Обрабатывает:
        - body_part → profile.injuries (добавить, если нет)
        - goal      → profile.training_goals (добавить, если нет)
        - sport_type → profile.preferred_sports (добавить, если нет)

        TODO v3: LLM-based fact extraction
        """
        body_part = entities.get("body_part")
        goal = entities.get("goal")
        sport_type = entities.get("sport_type")

        if not (body_part or goal or sport_type):
            return

        try:
            from sqlalchemy import select
            from app.models.user_profile import UserProfile

            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(UserProfile).where(UserProfile.user_id == user_id)
                )
                profile = result.scalar_one_or_none()
                if profile is None:
                    return

                changed = False

                if body_part:
                    existing_parts = [
                        i.get("body_part") for i in (profile.injuries or [])
                        if isinstance(i, dict)
                    ]
                    if body_part not in existing_parts:
                        profile.injuries = list(profile.injuries or []) + [
                            {"body_part": body_part, "status": "active"}
                        ]
                        changed = True
                        logger.info(
                            "MemoryUpdater: добавлена травма '%s' для user=%s",
                            body_part, user_id,
                        )

                if goal:
                    existing_goals = list(profile.training_goals or [])
                    if goal not in existing_goals:
                        profile.training_goals = existing_goals + [goal]
                        changed = True
                        logger.info(
                            "MemoryUpdater: добавлена цель '%s' для user=%s",
                            goal, user_id,
                        )

                if sport_type:
                    existing_sports = list(profile.preferred_sports or [])
                    if sport_type not in existing_sports:
                        profile.preferred_sports = existing_sports + [sport_type]
                        changed = True
                        logger.info(
                            "MemoryUpdater: добавлен спорт '%s' для user=%s",
                            sport_type, user_id,
                        )

                if changed:
                    profile.updated_at = datetime.utcnow()
                    await db.commit()

        except Exception as exc:
            logger.warning(
                "MemoryUpdater._update_long_term: ошибка user=%s: %s", user_id, exc,
            )

    async def _update_semantic(
        self,
        user_id: str,
        request_id: str | None,
        query: str,
        response: str,
    ) -> None:
        """Сохранить пару Q/A в semantic memory."""
        try:
            await semantic_memory.remember(
                user_id=user_id,
                request_id=request_id,
                query=query,
                response=response,
            )
        except Exception as exc:
            logger.warning(
                "MemoryUpdater._update_semantic: ошибка user=%s: %s", user_id, exc,
            )


memory_updater = MemoryUpdater()
