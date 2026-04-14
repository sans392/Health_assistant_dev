"""Модуль анализа трендов метрик (trend_analyzer).

Метод: сравнение среднего за последние 7 дней с предыдущими 7 днями.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal


@dataclass
class TrendResult:
    """Результат анализа тренда метрики."""

    direction: Literal["up", "down", "stable"]  # направление изменения
    change_percent: float                         # изменение в процентах
    recent_avg: float                             # среднее за последние 7 дней
    baseline_avg: float                           # среднее за предыдущие 7 дней
    data_points_recent: int                       # кол-во точек в recent периоде
    data_points_baseline: int                     # кол-во точек в baseline периоде


def analyze_trend(
    time_series: dict[str, float],
    reference_date: date | None = None,
    stable_threshold_pct: float = 5.0,
) -> TrendResult:
    """Определить направление тренда по временному ряду.

    Сравнивает среднее за последние 7 дней с предыдущими 7 днями.
    Если изменение меньше stable_threshold_pct — направление «stable».

    Args:
        time_series: Словарь {iso_date: значение метрики} — любая метрика.
        reference_date: Дата «сегодня» (по умолчанию date.today()).
        stable_threshold_pct: Порог для признания тренда стабильным (%).

    Returns:
        TrendResult с direction, change_percent и средними значениями.
    """
    today = reference_date or date.today()

    recent_start = today - timedelta(days=6)
    baseline_start = today - timedelta(days=13)
    baseline_end = today - timedelta(days=7)

    recent_values: list[float] = []
    baseline_values: list[float] = []

    for d_str, value in time_series.items():
        try:
            d = date.fromisoformat(d_str)
        except ValueError:
            continue
        if d >= recent_start:
            recent_values.append(value)
        elif baseline_start <= d <= baseline_end:
            baseline_values.append(value)

    # Если нет данных — стабильный тренд
    if not recent_values and not baseline_values:
        return TrendResult(
            direction="stable",
            change_percent=0.0,
            recent_avg=0.0,
            baseline_avg=0.0,
            data_points_recent=0,
            data_points_baseline=0,
        )

    recent_avg = sum(recent_values) / len(recent_values) if recent_values else 0.0
    baseline_avg = sum(baseline_values) / len(baseline_values) if baseline_values else 0.0

    # Вычисляем изменение в процентах относительно baseline
    if baseline_avg != 0.0:
        change_pct = ((recent_avg - baseline_avg) / abs(baseline_avg)) * 100
    elif recent_avg != 0.0:
        change_pct = 100.0
    else:
        change_pct = 0.0

    # Определяем направление
    if abs(change_pct) < stable_threshold_pct:
        direction: Literal["up", "down", "stable"] = "stable"
    elif change_pct > 0:
        direction = "up"
    else:
        direction = "down"

    return TrendResult(
        direction=direction,
        change_percent=round(change_pct, 1),
        recent_avg=round(recent_avg, 2),
        baseline_avg=round(baseline_avg, 2),
        data_points_recent=len(recent_values),
        data_points_baseline=len(baseline_values),
    )


def build_time_series_from_facts(
    daily_facts: list[dict],
    metric: str,
) -> dict[str, float]:
    """Построить временной ряд из списка дневных фактов для указанной метрики.

    Args:
        daily_facts: Список словарей DailyFact (из get_daily_facts).
        metric: Имя поля (например 'steps', 'recovery_score', 'hrv_rmssd_milli').

    Returns:
        Словарь {iso_date: значение} с непустыми значениями.
    """
    series: dict[str, float] = {}
    for fact in daily_facts:
        iso_date = fact.get("iso_date", "")
        value = fact.get(metric)
        if iso_date and value is not None:
            try:
                series[iso_date] = float(value)
            except (TypeError, ValueError):
                continue
    return series


def build_time_series_from_activities(
    activities: list[dict],
    metric: str,
) -> dict[str, float]:
    """Построить временной ряд из списка активностей для указанной метрики.

    Если несколько активностей в один день — суммируются (duration, calories, distance)
    или усредняются (avg_heart_rate, avg_speed).

    Args:
        activities: Список словарей активностей (из get_activities).
        metric: Имя поля активности.

    Returns:
        Словарь {iso_date: значение}.
    """
    _SUM_METRICS = {"duration_seconds", "calories", "distance_meters"}

    accumulator: dict[str, list[float]] = {}
    for act in activities:
        start_time = act.get("start_time", "")
        if not start_time:
            continue
        act_date = start_time[:10]
        value = act.get(metric)
        if value is not None:
            try:
                accumulator.setdefault(act_date, []).append(float(value))
            except (TypeError, ValueError):
                continue

    series: dict[str, float] = {}
    for d_str, values in accumulator.items():
        if metric in _SUM_METRICS:
            series[d_str] = sum(values)
        else:
            series[d_str] = sum(values) / len(values)

    return series
