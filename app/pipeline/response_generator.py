"""Генератор ответов через LLM (Response Generator).

Формирует промпт из контекста и structured_result, вызывает Ollama,
возвращает текстовый ответ.
"""

import json
import logging
from dataclasses import dataclass

from app.pipeline.context_builder import EnrichedQuery
from app.services.llm_service import LLMResponse, ollama_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Системный промпт (из архитектуры)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """Ты — фитнес-ассистент. Отвечай на русском.

## Профиль пользователя
{user_profile}

## Контекст разговора
{conversation_history}

## Результаты анализа
{structured_result}

## Запрос пользователя
{normalized_text}

Правила:
- Используй ТОЛЬКО данные из "Результаты анализа"
- Не придумывай числа
- Если данных нет — скажи об этом
- Будь конкретен: цифры, даты, проценты
- Форматируй ответ в Markdown: summary → details → recommendations
- Максимум ~500 токенов"""

# Суффикс для medium_priority safety
_SAFETY_WARNING_SUFFIX = (
    "\n\n---\n> ⚠️ Если симптомы сохраняются, рекомендую проконсультироваться с врачом."
)

# Системный промпт для fast_path (без structured_result)
_FAST_PATH_SYSTEM = """Ты — фитнес-ассистент. Отвечай на русском.
Будь конкретен и краток. Максимум ~200 токенов."""


@dataclass
class GeneratorResult:
    """Результат генерации ответа."""

    content: str                # Текст ответа (Markdown)
    llm_response: LLMResponse   # Метрики вызова LLM
    route: str                  # Маршрут, для которого генерировался ответ


def _format_user_profile(profile: dict | None) -> str:
    """Форматировать профиль пользователя для промпта."""
    if not profile:
        return "Профиль не задан."
    lines = [
        f"Имя: {profile.get('name', '—')}",
        f"Возраст: {profile.get('age', '—')} лет",
        f"Вес: {profile.get('weight_kg', '—')} кг, рост: {profile.get('height_cm', '—')} см",
        f"Уровень: {profile.get('experience_level', '—')}",
    ]
    goals = profile.get("training_goals") or []
    if goals:
        lines.append(f"Цели: {', '.join(goals)}")
    sports = profile.get("preferred_sports") or []
    if sports:
        lines.append(f"Виды спорта: {', '.join(sports)}")
    return "\n".join(lines)


def _format_conversation_history(history: list[dict]) -> str:
    """Форматировать историю разговора для промпта."""
    if not history:
        return "История пуста."
    lines = []
    for msg in history[-5:]:  # только последние 5 сообщений
        role = "Пользователь" if msg.get("role") == "user" else "Ассистент"
        content = msg.get("content", "")[:200]  # ограничение длины
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _format_structured_result(structured_result: dict | None) -> str:
    """Форматировать структурированные результаты для промпта."""
    if not structured_result:
        return "Данных нет."
    try:
        return json.dumps(structured_result, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(structured_result)


class ResponseGenerator:
    """Генерирует человекочитаемый ответ через Ollama LLM.

    Не выполняет вычислений — только форматирует готовые данные и
    вызывает LLM для генерации ответа.
    """

    async def generate(
        self,
        enriched_query: EnrichedQuery,
        route: str,
        structured_result: dict | None = None,
        safety_level: str = "ok",
    ) -> GeneratorResult:
        """Сгенерировать ответ на запрос пользователя.

        Args:
            enriched_query: Обогащённый запрос с профилем и историей.
            route: Маршрут из RouteResult (fast_direct_answer, tool_simple и т.д.).
            structured_result: Агрегированные данные из Tool Executor / Data Processing.
            safety_level: Уровень безопасности из SafetyResult (ok / medium_priority).

        Returns:
            GeneratorResult с текстом ответа и метриками LLM.
        """
        if route == "fast_direct_answer":
            return await self._generate_fast_path(enriched_query, safety_level)

        return await self._generate_standard(
            enriched_query=enriched_query,
            route=route,
            structured_result=structured_result,
            safety_level=safety_level,
        )

    async def _generate_fast_path(
        self,
        enriched_query: EnrichedQuery,
        safety_level: str,
    ) -> GeneratorResult:
        """Fast path: ответ без structured_result (direct_question, general_chat)."""
        profile_str = _format_user_profile(enriched_query.user_profile)
        history_str = _format_conversation_history(enriched_query.conversation_history)

        prompt = (
            f"Профиль пользователя:\n{profile_str}\n\n"
            f"История разговора:\n{history_str}\n\n"
            f"Запрос: {enriched_query.normalized_text}"
        )

        llm_response = await ollama_client.generate(
            prompt=prompt,
            system_prompt=_FAST_PATH_SYSTEM,
            temperature=0.7,
            max_tokens=300,
        )

        content = llm_response.content
        if safety_level == "medium_priority":
            content += _SAFETY_WARNING_SUFFIX

        logger.info(
            "ResponseGenerator fast_path: route=fast_direct_answer "
            "prompt_len=%d response_len=%d duration_ms=%.1f",
            llm_response.prompt_length,
            llm_response.response_length,
            llm_response.duration_ms,
        )

        return GeneratorResult(
            content=content,
            llm_response=llm_response,
            route="fast_direct_answer",
        )

    async def _generate_standard(
        self,
        enriched_query: EnrichedQuery,
        route: str,
        structured_result: dict | None,
        safety_level: str,
    ) -> GeneratorResult:
        """Standard path: ответ с structured_result из БД/модулей."""
        profile_str = _format_user_profile(enriched_query.user_profile)
        history_str = _format_conversation_history(enriched_query.conversation_history)
        result_str = _format_structured_result(structured_result)

        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            user_profile=profile_str,
            conversation_history=history_str,
            structured_result=result_str,
            normalized_text=enriched_query.normalized_text,
        )

        # Для стандартного пути используем только запрос как prompt
        llm_response = await ollama_client.generate(
            prompt=enriched_query.normalized_text,
            system_prompt=system_prompt,
            temperature=0.5,   # менее творческий — данные точные
            max_tokens=600,
        )

        content = llm_response.content
        if safety_level == "medium_priority":
            content += _SAFETY_WARNING_SUFFIX

        logger.info(
            "ResponseGenerator standard: route=%s "
            "prompt_len=%d response_len=%d duration_ms=%.1f",
            route,
            llm_response.prompt_length,
            llm_response.response_length,
            llm_response.duration_ms,
        )

        return GeneratorResult(
            content=content,
            llm_response=llm_response,
            route=route,
        )
