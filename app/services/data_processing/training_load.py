"""Модуль расчёта тренировочной нагрузки (training_load).

MVP-формула: load = duration_seconds × sport_coefficient + calories × 0.1
Acute/chronic ratio: нагрузка последних 7 дней / нагрузка предыдущих 7 дней.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta


# Коэффициенты нагрузки по виду спорта
SPORT_COEFFICIENTS: dict[str, float] = {
    "running": 1.2,
    "cycling": 0.9,
    "gym": 1.0,
    "walking": 0.5,
    "swimming": 1.1,
    "yoga": 0.6,
    "football": 1.0,
    "basketball": 1.0,
    "tennis": 0.9,
    "skiing": 1.0,
}

_DEFAULT_COEFFICIENT = 1.0


@dataclass
class TrainingLoad:
    """Результат расчёта тренировочной нагрузки."""

    daily_load: dict[str, float] = field(default_factory=dict)   # дата → нагрузка
    weekly_load: float = 0.0                                      # сумма за последние 7 дней
    chronic_load: float = 0.0                                     # сумма предыдущих 7 дней
    acute_chronic_ratio: float = 0.0                              # weekly / chronic


def _activity_load(activity: dict) -> float:
    """Вычислить нагрузку одной тренировки по MVP-формуле."""
    duration = activity.get("duration_seconds", 0) or 0
    calories = activity.get("calories", 0) or 0
    sport = activity.get("sport_type", "other")
    coeff = SPORT_COEFFICIENTS.get(sport, _DEFAULT_COEFFICIENT)
    return duration * coeff + calories * 0.1


def compute_training_load(
    activities: list[dict],
    reference_date: date | None = None,
) -> TrainingLoad:
    """Вычислить тренировочную нагрузку и acute/chronic ratio.

    Использует MVP-формулу без TRIMP (нет HR-данных в activities).

    Args:
        activities: Список словарей активностей (из get_activities).
        reference_date: Дата «сегодня» (по умолчанию date.today()).

    Returns:
        TrainingLoad с дневной нагрузкой и соотношением acute/chronic.
    """
    if not activities:
        return TrainingLoad()

    today = reference_date or date.today()

    # Агрегируем нагрузку по датам
    daily_load: dict[str, float] = {}
    for act in activities:
        start_time = act.get("start_time", "")
        if not start_time:
            continue
        act_date = start_time[:10]  # "YYYY-MM-DD"
        load = _activity_load(act)
        daily_load[act_date] = daily_load.get(act_date, 0.0) + load

    # Acute: сумма нагрузки за последние 7 дней
    acute_start = today - timedelta(days=6)
    weekly_load = sum(
        load
        for d_str, load in daily_load.items()
        if date.fromisoformat(d_str) >= acute_start
    )

    # Chronic: сумма нагрузки за предыдущие 7 дней (8–14 дней назад)
    chronic_start = today - timedelta(days=13)
    chronic_end = today - timedelta(days=7)
    chronic_load = sum(
        load
        for d_str, load in daily_load.items()
        if chronic_start <= date.fromisoformat(d_str) <= chronic_end
    )

    # Acute/chronic ratio (избегаем деления на 0)
    ratio = round(weekly_load / chronic_load, 2) if chronic_load > 0 else 0.0

    return TrainingLoad(
        daily_load={k: round(v, 1) for k, v in daily_load.items()},
        weekly_load=round(weekly_load, 1),
        chronic_load=round(chronic_load, 1),
        acute_chronic_ratio=ratio,
    )
