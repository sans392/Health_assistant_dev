"""Модуль суммарной статистики тренировок (activity_summary).

Агрегирует список активностей за период и возвращает сводку.
"""

from dataclasses import dataclass, field
from datetime import date


@dataclass
class SportBreakdown:
    """Статистика по одному виду спорта."""

    count: int
    total_duration_seconds: int
    total_calories: int
    total_distance_meters: float


@dataclass
class ActivitySummary:
    """Суммарная статистика тренировок за период."""

    total_activities: int
    total_duration_seconds: int
    total_calories: int
    total_distance_meters: float
    by_sport: dict[str, SportBreakdown] = field(default_factory=dict)
    streak_days: int = 0    # Максимальная непрерывная серия дней с тренировкой
    rest_days: int = 0      # Дни без тренировок в периоде

    @property
    def total_duration_minutes(self) -> int:
        """Общая продолжительность в минутах."""
        return self.total_duration_seconds // 60

    @property
    def total_distance_km(self) -> float:
        """Общая дистанция в километрах."""
        return round(self.total_distance_meters / 1000, 2)


def compute_activity_summary(activities: list[dict]) -> ActivitySummary:
    """Вычислить суммарную статистику по списку активностей.

    Args:
        activities: Список словарей активностей (из get_activities).

    Returns:
        ActivitySummary с агрегированными данными.
    """
    if not activities:
        return ActivitySummary(
            total_activities=0,
            total_duration_seconds=0,
            total_calories=0,
            total_distance_meters=0.0,
        )

    total_duration = 0
    total_calories = 0
    total_distance = 0.0
    by_sport: dict[str, dict] = {}
    # Множество дат тренировок (YYYY-MM-DD)
    training_dates: set[str] = set()

    for act in activities:
        duration = act.get("duration_seconds", 0) or 0
        calories = act.get("calories", 0) or 0
        distance = act.get("distance_meters") or 0.0
        sport = act.get("sport_type", "other")

        total_duration += duration
        total_calories += calories
        total_distance += distance

        if sport not in by_sport:
            by_sport[sport] = {
                "count": 0,
                "total_duration_seconds": 0,
                "total_calories": 0,
                "total_distance_meters": 0.0,
            }
        by_sport[sport]["count"] += 1
        by_sport[sport]["total_duration_seconds"] += duration
        by_sport[sport]["total_calories"] += calories
        by_sport[sport]["total_distance_meters"] += distance

        # Извлекаем дату из start_time
        start_time = act.get("start_time", "")
        if start_time:
            training_dates.add(start_time[:10])  # "YYYY-MM-DD"

    sport_breakdown = {
        sport: SportBreakdown(**stats)
        for sport, stats in by_sport.items()
    }

    streak, rest = _compute_streak_and_rest(training_dates)

    return ActivitySummary(
        total_activities=len(activities),
        total_duration_seconds=total_duration,
        total_calories=total_calories,
        total_distance_meters=total_distance,
        by_sport=sport_breakdown,
        streak_days=streak,
        rest_days=rest,
    )


def _compute_streak_and_rest(training_dates: set[str]) -> tuple[int, int]:
    """Вычислить серию и дни отдыха по множеству дат.

    Args:
        training_dates: Множество строк формата 'YYYY-MM-DD'.

    Returns:
        Кортеж (streak_days, rest_days).
    """
    if not training_dates:
        return 0, 0

    sorted_dates = sorted(training_dates)
    date_objects = [date.fromisoformat(d) for d in sorted_dates]

    # Диапазон периода
    period_start = date_objects[0]
    period_end = date_objects[-1]
    total_days = (period_end - period_start).days + 1
    rest_days = total_days - len(date_objects)

    # Максимальная непрерывная серия
    max_streak = 1
    current_streak = 1
    for i in range(1, len(date_objects)):
        delta = (date_objects[i] - date_objects[i - 1]).days
        if delta == 1:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 1

    return max_streak, max(0, rest_days)
