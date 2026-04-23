"""Clarification loop — задать один уточняющий вопрос пользователю (Issue #57).

Когда пользователь пишет «составь план» без указания периода / спорта / уровня,
pipeline раньше либо угадывал generic-план, либо уходил в planner loop и тратил
итерации LLM впустую. Этот модуль добавляет один round-trip: определить
недостающие обязательные слоты, задать пользователю один вопрос, сохранить
состояние в `pending_clarifications`, а следующий user message дозаполнить и
прогнать как продолжение исходного запроса.

Публичный API:
- `needs_clarification(intent, slots) -> list[str]`
- `build_clarification_question(missing: list[str]) -> str`
- `save_pending(...)` / `resume_from_clarification(...)` / `clear_pending(...)`
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.pending_clarification import PendingClarification
from app.pipeline.intent_detection import IntentResult
from app.pipeline.slot_state import slot_state_from_entities
from app.tools.schemas import AnalysisType

logger = logging.getLogger(__name__)

# TTL pending: короче — запутаемся при возврате через час. Длиннее — забудется.
PENDING_TTL = timedelta(minutes=3)

# Required slots по intent'ам. Значения — имена атрибутов SlotState.
# Строго обязательны только те слоты, без которых ответ будет generic / бесполезным.
# Для plan_request: sport_types желателен, но не критичен — generic-план без
# указания спорта всё равно валиден. А вот time_range критичен — не поняв
# горизонта планирования, planner будет гадать.
# Для большинства data_query дефолт «за неделю» достаточен, уточнение не нужно.
_REQUIRED_SLOTS: dict[str, list[str]] = {
    "plan_request": ["time_range"],
    "log_activity": ["sport_types"],
}

# Приоритет слота при выборе одного вопроса: спрашиваем самый важный,
# чтобы не заваливать пользователя несколькими вопросами сразу.
_SLOT_PRIORITY: list[str] = ["time_range", "sport_types", "metrics", "body_parts"]

# Шаблоны вопросов для каждого слота.
_QUESTION_TEMPLATES: dict[str, str] = {
    "time_range": "За какой период тебя интересует? Например: за неделю, за месяц, за последние 14 дней.",
    "sport_types": "Для какого вида активности? Например: бег, велосипед, плавание, зал.",
    "metrics": "Какой показатель интересует? Например: пульс, шаги, HRV, сон.",
    "body_parts": "Какая часть тела беспокоит?",
}

_DEFAULT_QUESTION = "Уточни, пожалуйста, детали запроса — не хватает информации, чтобы ответить точно."


def needs_clarification(intent: str, slots) -> list[str]:
    """Вернуть список обязательных слотов, которых не хватает для intent.

    Args:
        intent: Имя intent'а (plan_request / data_query / log_activity / ...).
        slots: SlotState — типизированное состояние из IntentResult.slots.

    Returns:
        Список имён пустых обязательных слотов. Пустой список → уточнение
        не требуется.
    """
    required = list(_REQUIRED_SLOTS.get(intent, []))

    # data_query требует time_range только для COMPARE — сравнение без
    # периода бессмысленно. Для остальных analysis_type дефолт "за неделю"
    # достаточен в downstream модулях.
    if intent == "data_query" and slots.analysis_type == AnalysisType.COMPARE:
        required = ["time_range"]

    if not required:
        return []

    return slots.missing(required)


def build_clarification_question(missing: list[str]) -> str:
    """Сформировать один user-facing вопрос по списку недостающих слотов.

    Спрашиваем только самый приоритетный слот — остальные ждут следующего
    раунда. Это снижает нагрузку на пользователя и повышает шанс получить
    понятный ответ.
    """
    if not missing:
        return _DEFAULT_QUESTION

    for slot_name in _SLOT_PRIORITY:
        if slot_name in missing:
            return _QUESTION_TEMPLATES.get(slot_name, _DEFAULT_QUESTION)

    # На случай если missing содержит имя, которого нет в priority-списке —
    # просто берём первое, чтобы ответ был детерминированным.
    return _QUESTION_TEMPLATES.get(missing[0], _DEFAULT_QUESTION)


async def save_pending(
    db: AsyncSession,
    session_id: str,
    intent_result: IntentResult,
    missing: list[str],
    now: datetime | None = None,
) -> None:
    """Сохранить pending (replace-by-PK). На сессию — один активный pending."""
    now = now or datetime.utcnow()
    # Убираем предыдущий pending, если он был — новый запрос перекрывает старый.
    await db.execute(
        delete(PendingClarification).where(PendingClarification.session_id == session_id)
    )
    db.add(PendingClarification(
        session_id=session_id,
        intent=intent_result.intent,
        original_query=intent_result.raw_query,
        filled_slots=dict(intent_result.entities or {}),
        missing_slots=list(missing),
        created_at=now,
        expires_at=now + PENDING_TTL,
    ))
    await db.commit()


async def clear_pending(db: AsyncSession, session_id: str) -> None:
    """Удалить pending для сессии (идемпотентно)."""
    await db.execute(
        delete(PendingClarification).where(PendingClarification.session_id == session_id)
    )
    await db.commit()


async def resume_from_clarification(
    db: AsyncSession,
    session_id: str,
    new_message: str,
    now: datetime | None = None,
) -> IntentResult | None:
    """Попробовать продолжить прерванный запрос ответом пользователя.

    Если для сессии есть актуальный pending и в `new_message` удалось извлечь
    хотя бы один из недостающих слотов — мёржим entities, очищаем pending и
    возвращаем полный IntentResult с intent = pending.intent. raw_query
    склеивается из original_query + new_message, чтобы keyword-based routing
    ловил нужные маркеры (например `_WEEKLY_KEYWORDS` в плане).

    Возвращает None, если pending нет, он протух, либо ответ пользователя
    ничего из недостающего не покрыл — тогда вызывающий код должен
    обработать new_message как обычный запрос.
    """
    now = now or datetime.utcnow()

    stmt = select(PendingClarification).where(
        PendingClarification.session_id == session_id
    )
    result = await db.execute(stmt)
    pending = result.scalar_one_or_none()
    if pending is None:
        return None

    # TTL: pending протух → очищаем и отдаём None, чтобы запрос пошёл по
    # обычному пути.
    if pending.expires_at <= now:
        await db.execute(
            delete(PendingClarification).where(
                PendingClarification.session_id == session_id
            )
        )
        await db.commit()
        logger.info(
            "clarification: pending для session=%s протух, очищено", session_id
        )
        return None

    # Извлекаем слоты из нового сообщения и мёржим с уже заполненными.
    new_slots = slot_state_from_entities({}, raw_query=new_message)
    # Переиспользуем intent_detection._extract_entities — чтобы не дублировать
    # regex'ы. Импорт локальный во избежание циклов при загрузке модуля.
    from app.pipeline.intent_detection import _extract_entities

    new_entities = _extract_entities(new_message)

    filled: dict = dict(pending.filled_slots or {})
    missing: list[str] = list(pending.missing_slots or [])

    # Маппинг имя-слота (SlotState) → ключ(и) в legacy entities dict.
    # Первый ключ — канонический (single-value); второй — list-форма.
    slot_to_entity_keys: dict[str, tuple[str, ...]] = {
        "time_range": ("time_range",),
        "sport_types": ("sport_type", "sport_types"),
        "metrics": ("metric", "metrics"),
        "body_parts": ("body_part", "body_parts"),
        "intensity": ("intensity",),
    }

    # Мёржим ВСЕ извлечённые из new_message entities — пользователь мог
    # добавить лишнее помимо ответа («за неделю для бега» — и период, и
    # спорт): ловим это сейчас, чтобы потом не перепрашивать.
    for key, value in new_entities.items():
        if value and key not in filled:
            filled[key] = value

    # Проверяем, какие из изначально недостающих слотов покрыты новым ответом.
    covered: list[str] = []
    for slot_name in list(missing):
        keys = slot_to_entity_keys.get(slot_name, (slot_name,))
        for key in keys:
            if key in new_entities and new_entities[key]:
                covered.append(slot_name)
                break

    if not covered:
        logger.info(
            "clarification: new_message не покрыл недостающие слоты "
            "(missing=%s, session=%s) — обрабатываем как обычный запрос",
            missing, session_id,
        )
        return None

    # Склеиваем исходный запрос + ответ пользователя для raw_query —
    # keyword-based роутер (_WEEKLY_KEYWORDS и т.п.) смотрит именно сюда.
    combined_query = f"{pending.original_query}. {new_message}".strip()
    merged_slots = slot_state_from_entities(filled, raw_query=combined_query)

    resumed = IntentResult(
        intent=pending.intent,
        confidence=0.95,  # высокая уверенность: intent был зафиксирован в pending
        entities=filled,
        raw_query=combined_query,
        llm_used=False,
        slots=merged_slots,
    )

    # Pending исчерпан — удаляем.
    await db.execute(
        delete(PendingClarification).where(
            PendingClarification.session_id == session_id
        )
    )
    await db.commit()

    logger.info(
        "clarification: resumed session=%s intent=%s covered=%s",
        session_id, pending.intent, covered,
    )
    return resumed
