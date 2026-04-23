"""Response Generator v2 (Phase 2, Issue #30).

Обновления:
- Выбор модели по intent через LLM Registry (response / planner роль)
- RAG-чанки из structured_result добавляются в промпт
- Semantic context из enriched_query.semantic_context
- Token streaming через Ollama API (on_token callback)
- Улучшенные правила промпта: данные с числами, anomaly_flags, честный fallback

Формат запроса к Ollama — `/api/chat` со структурированной историей:
  system  — базовые инструкции ассистента
  system  — профиль пользователя
  system  — (опционально) tool/RAG/semantic контекст
  user / assistant ... — история диалога
  user    — текущий запрос пользователя
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass

from app.pipeline.context_builder import EnrichedQuery
from app.services.llm_registry import llm_registry
from app.services.llm_service import LLMResponse
from app.tools.time_utils import current_datetime_str

logger = logging.getLogger(__name__)

_TIMEOUT_FALLBACK = "Извини, сейчас я не могу сгенерировать ответ. Попробуй ещё раз."

_SAFETY_WARNING_SUFFIX = (
    "\n\n---\n> ⚠️ Если симптомы сохраняются, рекомендую проконсультироваться с врачом."
)

_PLANNER_ROLE_INTENTS = {"plan_request"}
_RAG_INTENTS = {"plan_request", "health_concern", "data_query", "reference_question"}

_BASE_SYSTEM_PROMPT_TEMPLATE = """Ты — дружелюбный фитнес-ассистент. Отвечай на русском.

Текущая дата и время сервера: {current_datetime}

Правила ответа:
- Используй ТОЛЬКО данные из system-сообщений: «Информация о пользователе»,
  «Релевантные знания (RAG)», «Релевантные прошлые ответы», «Результаты анализа».
- Упоминай anomaly_flags если они есть (например: «твоё HRV сегодня ниже обычного»).
- Ссылайся на конкретные числа и даты.
- Если данных нет — скажи об этом честно: «недостаточно данных».
- Тон: дружелюбный тренер, не врач.
- Форматируй в Markdown: краткий вывод → подробности → рекомендации.
- Максимум ~500 токенов."""

_FAST_PATH_SYSTEM_TEMPLATE = """Ты — дружелюбный фитнес-ассистент. Отвечай на русском.

Текущая дата и время сервера: {current_datetime}

Будь конкретен и краток. Максимум ~200 токенов."""


def _base_system_prompt() -> str:
    return _BASE_SYSTEM_PROMPT_TEMPLATE.format(current_datetime=current_datetime_str())


def _fast_path_system_prompt() -> str:
    return _FAST_PATH_SYSTEM_TEMPLATE.format(current_datetime=current_datetime_str())


def _select_role(intent: str) -> str:
    return "planner" if intent in _PLANNER_ROLE_INTENTS else "response"


def _format_user_profile(profile: dict | None) -> str:
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
    if not history:
        return "История пуста."
    lines = []
    for msg in history[-5:]:
        role = "Пользователь" if msg.get("role") == "user" else "Ассистент"
        content = msg.get("content", "")[:200]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _build_chat_history(history: list[dict], max_messages: int = 5) -> list[dict[str, str]]:
    """Преобразовать сохранённую историю в chat-формат role=user/assistant.

    Фильтруем только user/assistant, пропускаем прочие служебные роли.
    Берём последние `max_messages` сообщений.
    """
    if not history:
        return []
    converted: list[dict[str, str]] = []
    for msg in history[-max_messages:]:
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content") or ""
        if not content:
            continue
        converted.append({"role": role, "content": content})
    return converted


def _build_user_profile_system(profile: dict | None) -> str:
    """Собрать отдельный system-блок «Информация о пользователе»."""
    return "## Информация о пользователе\n" + _format_user_profile(profile)


def _build_context_system(
    structured_result: dict | None,
    rag_chunks: list[dict],
    semantic_context: list,
) -> str | None:
    """Собрать дополнительный system-блок c данными tools / RAG / semantic memory.

    Возвращает None, если никаких дополнительных данных нет — тогда system
    для tool/RAG-контекста не добавляется вовсе. Из structured_result вырезаем
    ключи, уже отрендеренные в отдельных блоках, чтобы не дублировать их
    в «## Результаты анализа».
    """
    parts: list[str] = []
    if rag_chunks:
        parts.append(_format_rag_block(rag_chunks).rstrip())
    if semantic_context:
        parts.append(_format_semantic_block(semantic_context).rstrip())
    stripped = _strip_presented_keys(structured_result, strip_rag=bool(rag_chunks))
    if stripped:
        parts.append(
            "## Результаты анализа\n" + _format_structured_result(stripped)
        )
    if not parts:
        return None
    return "\n\n".join(parts)


def _format_structured_result(structured_result: dict | None) -> str:
    if not structured_result:
        return "Данных нет."
    try:
        return json.dumps(structured_result, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(structured_result)


def _is_rag_key(key: str) -> bool:
    """Ключ RAG-результата: rag_retrieve или rag_retrieve_<category>."""
    return "rag_retrieve" in key


def _extract_rag_chunks(structured_result: dict | None) -> list[dict]:
    """Извлечь RAG-чанки из structured_result.

    Поддерживает оба варианта структуры:
      - верхний уровень (template_plan / planner): {"rag_retrieve_*": [...]}
      - вложение в tool_data (tool_simple): {"tool_data": {"rag_retrieve": [...]}}
    """
    if not structured_result:
        return []
    chunks: list[dict] = []
    for key, val in structured_result.items():
        if _is_rag_key(key) and isinstance(val, list):
            chunks.extend(val)
        elif key == "tool_data" and isinstance(val, dict):
            for inner_key, inner_val in val.items():
                if _is_rag_key(inner_key) and isinstance(inner_val, list):
                    chunks.extend(inner_val)
    return chunks


def _strip_presented_keys(
    structured_result: dict | None,
    strip_rag: bool,
) -> dict | None:
    """Убрать из structured_result ключи, уже показанные в отдельных system-блоках.

    - rag_retrieve* — только если RAG-блок действительно сформирован (strip_rag=True)

    Обрабатывает и вложенный tool_data (структура маршрута tool_simple).
    Возвращает None, если после очистки словарь пуст — тогда блок
    «## Результаты анализа» можно не добавлять вовсе.
    """
    if not structured_result:
        return structured_result

    def is_presented(key: str) -> bool:
        if strip_rag and _is_rag_key(key):
            return True
        return False

    cleaned: dict = {}
    for key, val in structured_result.items():
        if is_presented(key):
            continue
        if key == "tool_data" and isinstance(val, dict):
            inner = {k: v for k, v in val.items() if not is_presented(k)}
            if inner:
                cleaned[key] = inner
            continue
        cleaned[key] = val

    return cleaned or None


def _format_rag_block(rag_chunks: list[dict]) -> str:
    if not rag_chunks:
        return ""
    lines = ["## Релевантные знания (RAG):"]
    for chunk in rag_chunks[:6]:
        if isinstance(chunk, dict):
            category = chunk.get("category", "")
            confidence = chunk.get("confidence", "")
            text = (chunk.get("text") or "")[:500]
            lines.append(f"- [{category}, {confidence}] {text}")
    return "\n".join(lines) + "\n\n"


def _format_semantic_block(semantic_context: list) -> str:
    if not semantic_context:
        return ""
    lines = ["## Релевантные прошлые ответы:"]
    for item in semantic_context[:3]:
        text = (item.get("text") or "")[:500]
        lines.append(f"- {text}")
    return "\n".join(lines) + "\n\n"


@dataclass
class GeneratorResult:
    """Результат генерации ответа."""

    content: str
    llm_response: LLMResponse
    route: str


class ResponseGenerator:
    """Генерирует человекочитаемый ответ через Ollama LLM.

    v2: role-based LLM, RAG-контекст, semantic context, token streaming.
    """

    async def generate(
        self,
        enriched_query: EnrichedQuery,
        route: str,
        structured_result: dict | None = None,
        safety_level: str = "ok",
        intent: str = "general_chat",
        on_token: Callable[[str], None] | None = None,
    ) -> GeneratorResult:
        """Сгенерировать ответ на запрос пользователя.

        Args:
            enriched_query: Обогащённый запрос с профилем и историей.
            route: Маршрут из RouteResult.
            structured_result: Данные из Tool/Template/Planner executor.
            safety_level: Уровень безопасности (ok / medium_priority).
            intent: Намерение пользователя (для выбора роли LLM).
            on_token: Callback для каждого токена стрима (опционально).

        Returns:
            GeneratorResult с текстом ответа и метриками LLM.
        """
        if route == "fast_direct_answer":
            return await self._generate_fast_path(enriched_query, safety_level, on_token)

        return await self._generate_standard(
            enriched_query=enriched_query,
            route=route,
            structured_result=structured_result,
            safety_level=safety_level,
            intent=intent,
            on_token=on_token,
        )

    async def _generate_fast_path(
        self,
        enriched_query: EnrichedQuery,
        safety_level: str,
        on_token: Callable[[str], None] | None,
    ) -> GeneratorResult:
        """Fast path: короткий ответ без данных (greeting, off_topic, general_question).

        Формирует два system-сообщения (базовая инструкция + профиль) и
        передаёт историю диалога + текущий запрос как role=user/assistant.
        """
        system_prompts = [
            _fast_path_system_prompt(),
            _build_user_profile_system(enriched_query.user_profile),
        ]
        messages = _build_chat_history(enriched_query.conversation_history)
        messages.append({"role": "user", "content": enriched_query.normalized_text})

        llm_client = llm_registry.get_client("response")
        llm_response = await self._call_llm(
            llm_client=llm_client,
            messages=messages,
            system_prompts=system_prompts,
            temperature=0.7,
            max_tokens=300,
            on_token=on_token,
        )

        content = llm_response.content
        if safety_level == "medium_priority":
            content += _SAFETY_WARNING_SUFFIX

        logger.info(
            "ResponseGenerator fast_path: prompt_len=%d response_len=%d duration_ms=%.1f",
            llm_response.prompt_length, llm_response.response_length, llm_response.duration_ms,
        )

        return GeneratorResult(content=content, llm_response=llm_response, route="fast_direct_answer")

    async def _generate_standard(
        self,
        enriched_query: EnrichedQuery,
        route: str,
        structured_result: dict | None,
        safety_level: str,
        intent: str,
        on_token: Callable[[str], None] | None,
    ) -> GeneratorResult:
        """Standard path: ответ с данными из tool/template/planner executor.

        Три system-блока:
          1. Базовый промпт (правила ответа).
          2. Информация о пользователе.
          3. (Опционально) RAG / semantic memory / результаты tools.
        История диалога передаётся через role=user/assistant.
        """
        rag_chunks: list[dict] = []
        if intent in _RAG_INTENTS:
            rag_chunks = _extract_rag_chunks(structured_result)
            if enriched_query.knowledge_context:
                rag_chunks = list(enriched_query.knowledge_context) + rag_chunks

        system_prompts: list[str] = [
            _base_system_prompt(),
            _build_user_profile_system(enriched_query.user_profile),
        ]
        context_system = _build_context_system(
            structured_result=structured_result,
            rag_chunks=rag_chunks,
            semantic_context=enriched_query.semantic_context,
        )
        if context_system:
            system_prompts.append(context_system)

        messages = _build_chat_history(enriched_query.conversation_history)
        messages.append({"role": "user", "content": enriched_query.normalized_text})

        role = _select_role(intent)
        llm_client = llm_registry.get_client(role)

        llm_response = await self._call_llm(
            llm_client=llm_client,
            messages=messages,
            system_prompts=system_prompts,
            temperature=0.5,
            max_tokens=600,
            on_token=on_token,
        )

        content = llm_response.content
        if safety_level == "medium_priority":
            content += _SAFETY_WARNING_SUFFIX

        logger.info(
            "ResponseGenerator standard: route=%s role=%s intent=%s "
            "prompt_len=%d response_len=%d duration_ms=%.1f",
            route, role, intent,
            llm_response.prompt_length, llm_response.response_length, llm_response.duration_ms,
        )

        return GeneratorResult(content=content, llm_response=llm_response, route=route)

    async def _call_llm(
        self,
        llm_client: object,
        messages: list[dict[str, str]],
        system_prompts: list[str],
        temperature: float,
        max_tokens: int,
        on_token: Callable[[str], None] | None,
    ) -> LLMResponse:
        """Вызвать LLM через /api/chat (streaming или non-streaming), с fallback при ошибке."""
        try:
            if on_token is not None:
                return await llm_client.chat_stream(
                    messages=messages,
                    system_prompts=system_prompts,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    on_token=on_token,
                )
            else:
                return await llm_client.chat(
                    messages=messages,
                    system_prompts=system_prompts,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
        except Exception as exc:
            logger.error("ResponseGenerator: LLM ошибка: %s", exc)
            return LLMResponse(
                content=_TIMEOUT_FALLBACK, model="fallback",
                prompt_length=0, response_length=len(_TIMEOUT_FALLBACK), duration_ms=0.0,
            )
