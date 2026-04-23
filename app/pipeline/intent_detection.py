"""Модуль определения намерения пользователя (Phase 2 — rule-based + LLM fallback).

Stage 1: rule-based классификатор (regex + keyword scoring).
Stage 2: LLM fallback через intent_llm роль при confidence < 0.85.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.pipeline.slot_state import SlotState, slot_state_from_entities
from app.tools.time_utils import current_datetime_str

if TYPE_CHECKING:
    from app.services.llm_registry import LLMRegistry

logger = logging.getLogger(__name__)

# Порог уверенности для перехода на LLM stage 2
_LLM_FALLBACK_THRESHOLD = 0.85

# Допустимые значения intent (для валидации ответа LLM)
_VALID_INTENTS = {
    "data_query",           # объединённый data_retrieval + data_analysis
    "plan_request",
    "health_concern",
    "direct_question",
    "general_chat",
    "emergency",
    "off_topic",
    "reference_question",   # FAQ: нормы, определения, рекомендации
    "capability_question",  # мета: «что ты умеешь»
}


@dataclass
class IntentResult:
    """Результат классификации намерения.

    entities — legacy dict (обратная совместимость с router/template_executor).
    slots — типизированный SlotState с нормализованным TimeRange и enum'ами.
    Оба поля согласованы и заполняются одновременно.
    """

    intent: str           # тип намерения
    confidence: float     # уверенность от 0.0 до 1.0
    entities: dict        # извлечённые сущности (legacy dict)
    raw_query: str        # исходный запрос
    llm_used: bool = False  # True если была выполнена LLM stage 2
    slots: SlotState = field(default_factory=SlotState)  # типизированное состояние


# Правила классификации: (паттерн, intent, confidence)
_RULES: list[tuple[str, str, float]] = [
    # capability_question — мета-вопросы о функциях ассистента
    (r"(что\s+ты\s+умеешь|что\s+умеешь|что\s+ты\s+можешь\s+делать)", "capability_question", 0.95),
    (r"(какие\s+у\s+тебя\s+(функции|возможности)|твои\s+(функции|возможности))", "capability_question", 0.9),
    (r"(как\s+тобой\s+пользоваться|как\s+с\s+тобой\s+работать)", "capability_question", 0.9),
    (r"^\s*(помощь|help)\s*[?!.]*\s*$", "capability_question", 0.9),

    # reference_question — FAQ, нормы, определения, рекомендации
    (r"(сколько\s+(нужно|надо|должно|должна|необходимо))", "reference_question", 0.9),
    (r"\bнорм[аыу]\s+(шаг|пульс|сн[ае]|калори|белк|воды|нагрузк|hrv|рест|чсс)", "reference_question", 0.9),
    (r"(что\s+так(ое|ая)|что\s+означает)\b", "reference_question", 0.88),
    (r"\bрекоменд(ац\w+|уй\w*|ов\w+)", "reference_question", 0.85),

    # data_query — запрос данных (с анализом или без); analysis_type — ортогональный слот
    (r"(покажи|дай|выведи).*(тренировк|занят|активност)", "data_query", 0.9),
    (r"(история|список|лог).*(тренировк|занят)", "data_query", 0.85),
    (r"(сколько|какие).*(тренировк|занят|активност).*(был|прошл|за)", "data_query", 0.85),
    (r"(проанализир|анализ|сравни|динамик|прогресс|тренд)", "data_query", 0.9),
    (r"(насколько|как.*(изменил|улучшил|вырос|снизил))", "data_query", 0.85),
    (r"(статистик|среднее|средн|за (неделю|месяц|период).*показател)", "data_query", 0.85),
    (r"это\s+норма(льно)?\b", "data_query", 0.85),
    (r"где\s+(я\s+)?(проседа|провал|слаб)", "data_query", 0.85),

    # plan_request — составление плана
    (r"(составь|создай|сделай|напиши|придумай).*(план|программ|расписани)", "plan_request", 0.9),
    (r"(план|программ).*(тренировок|занятий|на неделю|на месяц)", "plan_request", 0.85),

    # health_concern — жалоба на здоровье
    (r"(болит|боль|дискомфорт|немеет|ноет|жжёт|жжение|покалива)", "health_concern", 0.95),
    (r"(травм|ушиб|растяжен|надрыв|разрыв)", "health_concern", 0.95),
    (r"(плохо\s*(себя\s*)?чувству|самочувств)", "health_concern", 0.9),

    # direct_question — простой вопрос без вычислений
    (r"(сколько|какой|когда|где|что).*(пульс|вес|веш|калор|шаг|рост)", "direct_question", 0.85),
    (r"(какой у меня|покажи мой|мой).*(вес|пульс|рост|возраст)", "direct_question", 0.85),
    (r"объясни\b.+", "direct_question", 0.8),

    # general_chat — общий разговор
    (r"^(привет|здравствуй|добрый|хай|hello|hi)[\s!.]*$", "general_chat", 0.95),
    (r"^(спасибо|благодар|пожалуйста|окей|ок|хорошо|понял|ясно)[\s!.]*$", "general_chat", 0.95),
    (r"^(пока|до свидани|увидимся|всё)[\s!.]*$", "general_chat", 0.9),
    (r"(как дела|как ты|что новог)", "general_chat", 0.85),
]

# Паттерны для извлечения analysis_type — ортогональный слот для data_query
_ANALYSIS_TYPE_PATTERNS: list[tuple[str, str]] = [
    (r"это\s+норма(льно)?\b|в\s+норме\b|укладываюсь\s+в\s+норму", "norm_check"),
    (r"\bсравни(ть|м)?\b|по\s+сравнению|в\s+сравнении", "compare"),
    (r"\b(динамик|тренд|прогресс)\w*\b|как.*(изменил|улучшил|вырос|снизил)", "trend"),
    (r"где\s+(я\s+)?(проседа|провал|слаб)|в\s+чём\s+проблем", "breakdown"),
]

# Паттерны для извлечения сущностей time_range
_TIME_RANGE_PATTERNS: list[tuple[str, str | None]] = [
    (r"\bсегодня\b", "сегодня"),
    (r"\bвчера\b", "вчера"),
    (r"\bза неделю\b|\bна неделю\b|\bна прошл\w+ неделе\b|\bза последн\w+ неделю\b", "за неделю"),
    (r"\bза месяц\b|\bна месяц\b|\bна прошл\w+ месяц\b|\bза последн\w+ месяц\b", "за месяц"),
    (r"\bв январ\w+\b", "январь"),
    (r"\bв феврал\w+\b", "февраль"),
    (r"\bв март\w+\b", "март"),
    (r"\bв апрел\w+\b", "апрель"),
    (r"\bв ма[йе]\w*\b", "май"),
    (r"\bв июн\w+\b", "июнь"),
    (r"\bв июл\w+\b", "июль"),
    (r"\bв август\w*\b", "август"),
    (r"\bв сентябр\w+\b", "сентябрь"),
    (r"\bв октябр\w+\b", "октябрь"),
    (r"\bв ноябр\w+\b", "ноябрь"),
    (r"\bв декабр\w+\b", "декабрь"),
    (r"\bза последн\w+ (\d+) дн\w+\b", None),  # динамически
]

# Паттерны для извлечения сущностей sport_type
_SPORT_PATTERNS: list[tuple[str, str]] = [
    (r"\bбег\w*\b|\bпробежк\w*\b", "running"),
    (r"\bвелосипед\w*\b|\bвело\b|\bциклинг\w*\b", "cycling"),
    (r"\bплавани\w+\b|\bбассейн\w*\b|\bплавал\w*\b|\bплыл\w*\b", "swimming"),
    (r"\bзал\b|\bтренажёрн\w*\b|\bтренажерн\w*\b|\bсилов\w+\b", "gym"),
    (r"\bйог\w+\b", "yoga"),
    (r"\bфутбол\w*\b", "football"),
    (r"\bбаскетбол\w*\b", "basketball"),
    (r"\bтеннис\w*\b", "tennis"),
    (r"\bлыж\w+\b|\bлыжн\w+\b", "skiing"),
    (r"\bходьб\w+\b|\bпрогулк\w*\b", "walking"),
]

# Паттерны для извлечения сущностей metric (расширенный Phase 2)
_METRIC_PATTERNS: list[tuple[str, str]] = [
    (r"\bпульс\w*\b|\bчсс\b|\bсердцебиени\w+\b|\bheart.rate\b", "heart_rate"),
    (r"\bhrv\b|\bвариабельност\w+.*(ритм|пульс|серд)", "hrv"),
    (r"\bвес\w*\b|\bмасс\w+\b|\bвеш\w+\b", "вес"),
    (r"\bкалори\w+\b|\bккал\b", "калории"),
    (r"\bшаг\w*\b|\bшагомер\w*\b", "шаги"),
    (r"\bрост\w*\b", "рост"),
    (r"\bдистанци\w+\b|\bкилометр\w*\b|\bкм\b", "дистанция"),
    (r"\bвремя\b|\bдлительност\w+\b|\bпродолжительност\w+\b", "время"),
    (r"\bтемп\w*\b|\bскорост\w+\b|\bпейс\b|\bpace\b", "темп"),
    (r"\bкаденс\b|\bcadence\b|\bшагов.в.минут\w*\b", "cadence"),
    (r"\bсон\b|\bсна\b|\bсну\b|\bсном\b|\bсне\b|\bсплю\b|\bсплюсь\b|\bнасколько.*(сплю|сон)", "сон"),
    (r"\bвосстановлен\w+\b|\brecovery\b", "recovery"),
    (r"\bнагрузк\w+\b|\bstrain\b", "strain"),
    (r"\brpe\b|\bсубъективн\w*.*(нагрузк|усили)", "rpe"),
]

# Паттерны для извлечения body_part
_BODY_PART_PATTERNS: list[tuple[str, str]] = [
    (r"\bспин\w*\b|\bспиной\b", "спина"),
    (r"\bколен\w+\b|\bколено\b", "колено"),
    (r"\bпоясниц\w+\b|\bпояснич\w+\b", "поясница"),
    (r"\bплеч\w+\b|\bнаплечн\w+\b", "плечи"),
    (r"\bше[ея]\w*\b|\bшейн\w+\b", "шея"),
    (r"\bбедр\w+\b|\bбедренн\w+\b", "бедро"),
    (r"\bлодыжк\w+\b|\bлодыжечн\w+\b", "лодыжка"),
    (r"\bзапяст\w+\b", "запястье"),
    (r"\bлоктевой\b|\bлокт\w+\b", "локоть"),
    (r"\bикр\w+\b|\bикроножн\w+\b", "икра"),
]

# Паттерны для извлечения intensity
_INTENSITY_PATTERNS: list[tuple[str, str]] = [
    (r"\bлегк\w+\b|\bнебольш\w+\b|\bлёгк\w+\b", "легко"),
    (r"\bтяжел\w*\b|\bтяжёл\w*\b|\bтяжко\b|\bсложно\b|\bинтенсивн\w+\b", "тяжело"),
    (r"\bсильно\b|\bсильн\w+\b|\bочень\b", "сильно"),
    (r"\bумеренно\b|\bумеренн\w+\b|\bсредн\w+\b", "умеренно"),
]


def _extract_entities(text: str) -> dict:
    """Извлекает сущности из текста запроса."""
    entities: dict = {}
    lower = text.lower()

    # time_range
    for pattern, label in _TIME_RANGE_PATTERNS:
        if label is None:
            m = re.search(pattern, lower)
            if m:
                entities["time_range"] = f"за последние {m.group(1)} дней"
                break
        elif re.search(pattern, lower):
            entities["time_range"] = label
            break

    # sport_type
    for pattern, sport in _SPORT_PATTERNS:
        if re.search(pattern, lower):
            entities["sport_type"] = sport
            break

    # metric
    for pattern, metric in _METRIC_PATTERNS:
        if re.search(pattern, lower):
            entities["metric"] = metric
            break

    # body_part
    for pattern, part in _BODY_PART_PATTERNS:
        if re.search(pattern, lower):
            entities["body_part"] = part
            break

    # intensity
    for pattern, intensity in _INTENSITY_PATTERNS:
        if re.search(pattern, lower):
            entities["intensity"] = intensity
            break

    # analysis_type (ортогональный слот для data_query)
    for pattern, analysis in _ANALYSIS_TYPE_PATTERNS:
        if re.search(pattern, lower):
            entities["analysis_type"] = analysis
            break

    return entities


def _detect_rule_based(query: str) -> IntentResult:
    """Stage 1: rule-based классификация (синхронная)."""
    lower = query.lower().strip()
    entities = _extract_entities(query)

    best_intent = "general_chat"
    best_confidence = 0.0

    for pattern, intent, confidence in _RULES:
        if re.search(pattern, lower):
            if confidence > best_confidence:
                best_intent = intent
                best_confidence = confidence

    if best_confidence < 0.5:
        best_intent = "general_chat"
        best_confidence = 0.3

    return IntentResult(
        intent=best_intent,
        confidence=best_confidence,
        entities=entities,
        raw_query=query,
        llm_used=False,
        slots=slot_state_from_entities(entities, raw_query=query),
    )


def _parse_llm_json(text: str) -> dict | None:
    """Разобрать JSON-ответ LLM с несколькими fallback-стратегиями."""
    # 1. Прямой парсинг
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 2. Извлечь JSON из markdown-блока ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Найти первый {...} в тексте
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


_INTENT_SYSTEM_PROMPT_TEMPLATE = """\
Ты классифицируешь намерение пользователя фитнес-ассистента.
Ответь строго в JSON-формате (без текста вне JSON):
{{"intent": "...", "confidence": 0.0, "entities": {{}}}}

Текущая дата и время сервера: {current_datetime}

Доступные намерения:
- data_query: запрос данных пользователя (история тренировок, метрики,
  а также их анализ — сравнение, динамика, «это нормально», «где проседаю»).
  При необходимости заполни entities.analysis_type: "none" | "norm_check" |
  "trend" | "breakdown" | "compare".
- plan_request: просьба составить план тренировок или программу
- health_concern: жалоба на боль, дискомфорт, травму или плохое самочувствие
- direct_question: простой вопрос о конкретном показателе пользователя
- reference_question: вопрос-справка о нормах, определениях, рекомендациях
  («сколько белка нужно», «что такое HRV», «норма шагов»)
- capability_question: мета-вопрос о функциях ассистента
  («что ты умеешь», «какие у тебя возможности»)
- general_chat: приветствие, благодарность, общий разговор
- emergency: экстренная ситуация, острая боль, угроза жизни
- off_topic: вопрос не по теме здоровья и фитнеса"""


def _build_intent_system_prompt() -> str:
    return _INTENT_SYSTEM_PROMPT_TEMPLATE.format(current_datetime=current_datetime_str())


def _build_intent_messages(
    query: str, history: list[dict] | None
) -> list[dict[str, str]]:
    """Собрать messages для /api/chat: history (до 3 последних) + текущий запрос."""
    messages: list[dict[str, str]] = []
    if history:
        for msg in history[-3:]:
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue
            content = (msg.get("content") or "")[:200]
            if not content:
                continue
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": query})
    return messages


class IntentDetector:
    """Классификатор намерений пользователя (rule-based + LLM fallback).

    Stage 1: regex-правила (синхронно, быстро).
    Stage 2: LLM через intent_llm роль при confidence < 0.85.
    """

    async def detect(
        self,
        query: str,
        llm_registry: "LLMRegistry | None" = None,
        history: list[dict] | None = None,
    ) -> IntentResult:
        """Определить намерение пользователя.

        Args:
            query: Текст запроса пользователя.
            llm_registry: Реестр LLM-моделей (для Stage 2). None = только rule-based.
            history: История диалога [{"role": "user"/"assistant", "content": "..."}].

        Returns:
            IntentResult с определённым намерением и сущностями.
        """
        # Stage 1: rule-based
        result = _detect_rule_based(query)

        # Stage 2: LLM fallback при низкой уверенности
        if llm_registry is not None and result.confidence < _LLM_FALLBACK_THRESHOLD:
            result = await self._llm_stage(query, result, llm_registry, history)

        return result

    async def _llm_stage(
        self,
        query: str,
        stage1_result: IntentResult,
        llm_registry: "LLMRegistry",
        history: list[dict] | None,
    ) -> IntentResult:
        """Stage 2: LLM-уточнение намерения через /api/chat с format=json."""
        messages = _build_intent_messages(query, history)
        start_ms = time.monotonic() * 1000

        try:
            client = llm_registry.get_client("intent_llm")
            llm_response = await client.chat(
                messages=messages,
                system_prompt=_build_intent_system_prompt(),
                temperature=0.1,
                max_tokens=200,
                format="json",
            )
        except Exception as exc:
            logger.warning(
                "IntentDetector Stage 2: LLM вызов завершился ошибкой: %s. "
                "Используем rule-based результат.",
                exc,
            )
            return stage1_result

        duration_ms = time.monotonic() * 1000 - start_ms
        logger.info(
            "IntentDetector Stage 2: LLM вызов завершён за %.1fms (model=%s)",
            duration_ms,
            llm_response.model,
        )

        parsed = _parse_llm_json(llm_response.content)
        if parsed is None:
            logger.warning(
                "IntentDetector Stage 2: не удалось разобрать JSON ответ LLM: %r. "
                "Используем rule-based результат.",
                llm_response.content[:200],
            )
            return stage1_result

        llm_intent = parsed.get("intent", "")
        llm_confidence = float(parsed.get("confidence", 0.0))
        llm_entities = parsed.get("entities", {}) or {}

        # Валидация intent из LLM
        if llm_intent not in _VALID_INTENTS:
            logger.warning(
                "IntentDetector Stage 2: LLM вернул неизвестный intent %r. "
                "Используем rule-based результат.",
                llm_intent,
            )
            return stage1_result

        # Объединяем entities: rule-based + LLM (rule-based имеет приоритет)
        merged_entities = {**llm_entities, **stage1_result.entities}

        return IntentResult(
            intent=llm_intent,
            confidence=min(1.0, max(0.0, llm_confidence)),
            entities=merged_entities,
            raw_query=query,
            llm_used=True,
            slots=slot_state_from_entities(merged_entities, raw_query=query),
        )
