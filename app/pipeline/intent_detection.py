"""Модуль определения намерения пользователя (rule-based классификатор)."""

import re
from dataclasses import dataclass, field


@dataclass
class IntentResult:
    """Результат классификации намерения."""

    intent: str           # тип намерения
    confidence: float     # уверенность от 0.0 до 1.0
    entities: dict        # извлечённые сущности
    raw_query: str        # исходный запрос


# Правила классификации: (паттерн, intent, confidence)
_RULES: list[tuple[str, str, float]] = [
    # data_retrieval — запрос на получение данных
    (r"(покажи|дай|выведи|покажи).*(тренировк|занят|активност)", "data_retrieval", 0.9),
    (r"(история|список|лог).*(тренировк|занят)", "data_retrieval", 0.85),
    (r"(сколько|какие).*(тренировк|занят|активност).*(был|прошл|за)", "data_retrieval", 0.85),

    # plan_request — составление плана
    (r"(составь|создай|сделай|напиши|придумай).*(план|программ|расписани)", "plan_request", 0.9),
    (r"(план|программ).*(тренировок|занятий|на неделю|на месяц)", "plan_request", 0.85),

    # health_concern — жалоба на здоровье
    (r"(болит|боль|дискомфорт|немеет|ноет|жжёт|жжение|покалива)", "health_concern", 0.95),
    (r"(травм|ушиб|растяжен|надрыв|разрыв)", "health_concern", 0.95),
    (r"(плохо\s*(себя\s*)?чувству|самочувств)", "health_concern", 0.9),

    # data_analysis — запрос с вычислениями и анализом
    (r"(проанализир|анализ|сравни|динамик|прогресс|тренд)", "data_analysis", 0.9),
    (r"(насколько|как.*(изменил|улучшил|вырос|снизил))", "data_analysis", 0.85),
    (r"(статистик|среднее|средн|за (неделю|месяц|период).*показател)", "data_analysis", 0.85),

    # direct_question — простой вопрос без вычислений
    (r"(сколько|какой|когда|где|что).*(пульс|вес|веш|калор|шаг|рост)", "direct_question", 0.85),
    (r"(какой у меня|покажи мой|мой).*(вес|пульс|рост|возраст)", "direct_question", 0.85),
    (r"(что такое|что означает|объясни).+", "direct_question", 0.8),

    # general_chat — общий разговор
    (r"^(привет|здравствуй|добрый|хай|hello|hi)[\s!.]*$", "general_chat", 0.95),
    (r"^(спасибо|благодар|пожалуйста|окей|ок|хорошо|понял|ясно)[\s!.]*$", "general_chat", 0.95),
    (r"^(пока|до свидани|увидимся|всё)[\s!.]*$", "general_chat", 0.9),
    (r"(как дела|как ты|что новог)", "general_chat", 0.85),
]

# Паттерны для извлечения сущностей time_range
_TIME_RANGE_PATTERNS: list[tuple[str, str]] = [
    (r"\bсегодня\b", "сегодня"),
    (r"\bвчера\b", "вчера"),
    (r"\bза неделю\b|\bна прошл\w+ неделе\b|\bза последн\w+ неделю\b", "за неделю"),
    (r"\bза месяц\b|\bна прошл\w+ месяц\b|\bза последн\w+ месяц\b", "за месяц"),
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

# Паттерны для извлечения сущностей metric
_METRIC_PATTERNS: list[tuple[str, str]] = [
    (r"\bпульс\w*\b|\bчсс\b|\bсердцебиени\w+\b", "пульс"),
    (r"\bвес\w*\b|\bмасс\w+\b|\bвеш\w+\b", "вес"),
    (r"\bкалори\w+\b|\bккал\b", "калории"),
    (r"\bшаг\w*\b|\bшагомер\w*\b", "шаги"),
    (r"\bрост\w*\b", "рост"),
    (r"\bдистанци\w+\b|\bкилометр\w*\b|\bкм\b", "дистанция"),
    (r"\bвремя\b|\bдлительност\w+\b|\bпродолжительност\w+\b", "время"),
    (r"\bтемп\w*\b|\bскорост\w+\b", "темп"),
]


def _extract_entities(text: str) -> dict:
    """Извлекает сущности из текста запроса."""
    entities: dict = {}
    lower = text.lower()

    # Извлечение time_range
    for pattern, label in _TIME_RANGE_PATTERNS:
        if label is None:
            m = re.search(pattern, lower)
            if m:
                entities["time_range"] = f"за последние {m.group(1)} дней"
                break
        elif re.search(pattern, lower):
            entities["time_range"] = label
            break

    # Извлечение sport_type
    for pattern, sport in _SPORT_PATTERNS:
        if re.search(pattern, lower):
            entities["sport_type"] = sport
            break

    # Извлечение metric
    for pattern, metric in _METRIC_PATTERNS:
        if re.search(pattern, lower):
            entities["metric"] = metric
            break

    return entities


class IntentDetector:
    """Rule-based классификатор намерений пользователя."""

    def detect(self, query: str) -> IntentResult:
        """Определяет намерение пользователя по тексту запроса.

        Args:
            query: Текст запроса пользователя.

        Returns:
            IntentResult с определённым намерением и извлечёнными сущностями.
        """
        lower = query.lower().strip()
        entities = _extract_entities(query)

        best_intent = "general_chat"
        best_confidence = 0.0

        for pattern, intent, confidence in _RULES:
            if re.search(pattern, lower):
                if confidence > best_confidence:
                    best_intent = intent
                    best_confidence = confidence

        # Fallback: если уверенность слишком низкая — general_chat
        if best_confidence < 0.5:
            best_intent = "general_chat"
            best_confidence = 0.3

        return IntentResult(
            intent=best_intent,
            confidence=best_confidence,
            entities=entities,
            raw_query=query,
        )
