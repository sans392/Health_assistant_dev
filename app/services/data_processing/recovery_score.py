"""Модуль recovery score (passthrough).

MVP: использует нативный recovery_score от Whoop из DailyFact.
Если данных нет — возвращает None (расчётный вариант требует HRV/resting HR,
которых пока нет в полном виде).
"""

from dataclasses import dataclass


@dataclass
class RecoveryScoreResult:
    """Результат получения recovery score."""

    score: int | None           # Значение 0–100 или None если нет данных
    source: str | None          # Источник данных (whoop, и т.д.)
    iso_date: str | None        # Дата записи
    available: bool             # True если данные есть


def get_recovery_score(daily_facts: list[dict]) -> RecoveryScoreResult:
    """Получить последний доступный recovery score из дневных фактов.

    Passthrough: возвращает нативный recovery_score от Whoop.
    Если нет данных — возвращает RecoveryScoreResult(available=False).

    Args:
        daily_facts: Список словарей DailyFact (из get_daily_facts),
                     отсортированных по дате по возрастанию.

    Returns:
        RecoveryScoreResult с последним доступным score.
    """
    if not daily_facts:
        return RecoveryScoreResult(
            score=None, source=None, iso_date=None, available=False
        )

    # Берём самую свежую запись с непустым recovery_score
    for fact in reversed(daily_facts):
        score = fact.get("recovery_score")
        if score is not None:
            return RecoveryScoreResult(
                score=int(score),
                source="whoop",
                iso_date=fact.get("iso_date"),
                available=True,
            )

    return RecoveryScoreResult(
        score=None, source=None, iso_date=None, available=False
    )
