"""Модуль проверки безопасности запроса (pattern-based guard)."""

import re
from dataclasses import dataclass


@dataclass
class SafetyResult:
    """Результат проверки безопасности запроса."""

    is_safe: bool                    # True если можно продолжать пайплайн
    safety_level: str                # "ok" | "medium_priority" | "high_priority"
    redirect_message: str | None     # Сообщение при блокировке (high_priority)
    warning_suffix: str | None       # Добавка к ответу при medium_priority


# Паттерны высокой опасности — блокировка + редирект к врачу
_HIGH_PRIORITY_PATTERNS: list[str] = [
    r"боль в груди",
    r"боль в сердц",
    r"не могу дышать",
    r"потер\w* сознани",
    r"потеря сознани",
    r"онемени\w* конечност",
    r"онемел\w* конечност",
    r"сильное головокружени",
    r"кровь в моч",
    r"кровотечени\w* не останавлива",
    r"потерял сознани",
    r"упал в обморок",
    r"обморок",
    r"инфаркт",
    r"инсульт",
]

# Паттерны среднего уровня — ответ + предупреждение
_MEDIUM_PRIORITY_PATTERNS: list[str] = [
    r"болит.*уже.*недел",
    r"уже.*недел\w+.*болит",
    r"постоянн\w+ усталост",
    r"хроническ\w+ усталост",
    r"не могу набрать вес",
    r"головная боль после тренировки",
    r"голова болит после тренировки",
    r"боль.*несколько дней",
    r"несколько дней.*боль",
    r"давление.*повышен",
    r"высокое давление",
    r"тахикарди",
    r"аритми",
    r"одышка.*покое",
]

_REDIRECT_MESSAGE = (
    "⚠️ Я обнаружил тревожные симптомы в вашем сообщении. "
    "Пожалуйста, немедленно обратитесь к врачу или вызовите скорую помощь (103). "
    "Не откладывайте визит к специалисту — ваше здоровье важнее всего."
)

_WARNING_SUFFIX = (
    "\n\n⚠️ *Важно:* описанные симптомы требуют внимания. "
    "Рекомендую проконсультироваться с врачом, прежде чем продолжать тренировки."
)


class SafetyChecker:
    """Pattern-based проверка безопасности запроса пользователя."""

    def check(self, query: str) -> SafetyResult:
        """Проверяет запрос на признаки опасных симптомов.

        Args:
            query: Текст запроса пользователя.

        Returns:
            SafetyResult с уровнем безопасности и, при необходимости, сообщением.
        """
        lower = query.lower()

        # Проверка высокого приоритета (блокировка)
        for pattern in _HIGH_PRIORITY_PATTERNS:
            if re.search(pattern, lower):
                return SafetyResult(
                    is_safe=False,
                    safety_level="high_priority",
                    redirect_message=_REDIRECT_MESSAGE,
                    warning_suffix=None,
                )

        # Проверка среднего приоритета (предупреждение)
        for pattern in _MEDIUM_PRIORITY_PATTERNS:
            if re.search(pattern, lower):
                return SafetyResult(
                    is_safe=True,
                    safety_level="medium_priority",
                    redirect_message=None,
                    warning_suffix=_WARNING_SUFFIX,
                )

        # Запрос безопасен
        return SafetyResult(
            is_safe=True,
            safety_level="ok",
            redirect_message=None,
            warning_suffix=None,
        )
