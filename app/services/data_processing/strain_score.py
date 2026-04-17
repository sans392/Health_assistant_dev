"""Модуль расчёта тренировочной нагрузки Strain Score — аналог Whoop Strain.

Формула: TRIMP-подобная схема (duration × zone_weight) нормализованная в 0–21.
"""

from dataclasses import dataclass
from datetime import date

from app.services.data_processing.heart_rate_zones import HRZones

# Веса для каждой HR-зоны (TRIMP zone multipliers)
_ZONE_WEIGHTS: dict[int, float] = {
    1: 1.0,
    2: 2.0,
    3: 3.5,
    4: 5.0,
    5: 7.0,
}

# Делитель для нормализации TRIMP → 0–21 (соответствует ~8ч интенсивной Z3 тренировке)
_NORMALIZATION_DIVISOR = 100_800.0

_STRENGTH_SPORTS = {"gym", "strength"}
_CARDIO_SPORTS = {"running", "cycling", "swimming", "walking", "skiing"}


@dataclass
class StrainScoreResult:
    """Результат расчёта Strain Score."""

    strain: float          # шкала 0–21
    primary_driver: str    # "cardio" | "strength" | "mixed"
    activities_count: int  # количество тренировок в расчёте


def _hr_to_zone(avg_hr: int, hr_zones: HRZones | None) -> int:
    """Определить зону по среднему пульсу."""
    if hr_zones:
        zone = hr_zones.zone_for_hr(avg_hr)
        if zone is not None:
            return zone
        # Выше max_hr → зона 5
        if avg_hr >= hr_zones.max_hr:
            return 5
        return 1

    # Без hr_zones — приблизительно по % от условного max=180
    pct = avg_hr / 180.0
    if pct < 0.60:
        return 1
    if pct < 0.70:
        return 2
    if pct < 0.80:
        return 3
    if pct < 0.90:
        return 4
    return 5


def compute_strain_score(
    activities: list[dict],
    hr_zones: HRZones | None = None,
    reference_date: date | None = None,
) -> StrainScoreResult:
    """Рассчитать Strain Score по тренировочным данным за день.

    Если activities содержит тренировки за несколько дней — берёт только
    тренировки за reference_date (сегодня). Если ни одной не нашлось —
    использует все переданные активности (удобно при явной фильтрации снаружи).

    Args:
        activities: Список тренировок из get_activities.
        hr_zones: Зоны пульса (опционально; без них — приближённый расчёт).
        reference_date: «Сегодня» (по умолчанию date.today()).

    Returns:
        StrainScoreResult со strain в шкале 0–21 и primary_driver.
    """
    if not activities:
        return StrainScoreResult(strain=0.0, primary_driver="cardio", activities_count=0)

    today = reference_date or date.today()
    day_str = today.isoformat()

    day_acts = [
        a for a in activities
        if (a.get("start_time") or "")[:10] == day_str
    ]
    target_acts = day_acts if day_acts else activities

    total_trimp = 0.0
    cardio_count = 0
    strength_count = 0

    for act in target_acts:
        duration = act.get("duration_seconds") or 0
        avg_hr = act.get("avg_heart_rate")
        sport = (act.get("sport_type") or "other").lower()

        if avg_hr and avg_hr > 0:
            zone = _hr_to_zone(int(avg_hr), hr_zones)
        else:
            zone = 2  # зона по умолчанию если нет HR-данных

        total_trimp += duration * _ZONE_WEIGHTS[zone]

        if sport in _STRENGTH_SPORTS:
            strength_count += 1
        elif sport in _CARDIO_SPORTS:
            cardio_count += 1

    strain = min(21.0, round(total_trimp / _NORMALIZATION_DIVISOR * 21.0, 1))

    if strength_count > 0 and cardio_count > 0:
        primary_driver = "mixed"
    elif strength_count > cardio_count:
        primary_driver = "strength"
    else:
        primary_driver = "cardio"

    return StrainScoreResult(
        strain=strain,
        primary_driver=primary_driver,
        activities_count=len(target_acts),
    )
