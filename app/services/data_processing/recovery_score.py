"""Модуль recovery score (Phase 2 — расчётная версия, Issue #27).

Приоритет:
1. Если в daily_facts есть recovery_score от Whoop — возвращает его напрямую.
2. Иначе — рассчитывает на основе HRV-тренда, RHR, качества сна, нагрузки.

get_recovery_score() — обратная совместимость с MVP (passthrough Whoop).
compute_recovery_score() — расчётная версия Phase 2.
"""

import statistics
from dataclasses import dataclass, field


@dataclass
class RecoveryScoreResult:
    """Результат получения recovery score."""

    score: int | None                        # Значение 0–100 или None если нет данных
    source: str | None                       # "whoop" | "calculated"
    iso_date: str | None                     # Дата записи (YYYY-MM-DD)
    available: bool                          # True если данные есть
    factors: dict = field(default_factory=dict)  # Разбивка по факторам


def get_recovery_score(daily_facts: list[dict]) -> RecoveryScoreResult:
    """Получить последний доступный recovery score из дневных фактов (passthrough Whoop).

    Args:
        daily_facts: Список словарей DailyFact (из get_daily_facts),
                     отсортированных по дате по возрастанию.

    Returns:
        RecoveryScoreResult с последним доступным score или available=False.
    """
    if not daily_facts:
        return RecoveryScoreResult(
            score=None, source=None, iso_date=None, available=False
        )

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


def compute_recovery_score(
    daily_facts: list[dict],
    activities: list[dict] | None = None,
) -> RecoveryScoreResult:
    """Рассчитать recovery score.

    Если в последней записи daily_facts есть recovery_score (Whoop) — возвращает его.
    Иначе рассчитывает по факторам: HRV-тренд, RHR, сон, тренировочная нагрузка.

    Args:
        daily_facts: Список DailyFact за последние 14+ дней (по возрастанию даты).
        activities: Список активностей для компонента нагрузки (опционально).

    Returns:
        RecoveryScoreResult со score и factors-разбивкой.
    """
    if not daily_facts:
        return RecoveryScoreResult(
            score=None, source=None, iso_date=None, available=False
        )

    # Приоритет: Whoop-данные
    whoop = get_recovery_score(daily_facts)
    if whoop.available:
        return whoop

    latest = daily_facts[-1]
    iso_date = latest.get("iso_date")
    factors: dict = {}
    component_scores: list[float] = []

    hrv_score = _hrv_component(daily_facts, factors)
    if hrv_score is not None:
        component_scores.append(hrv_score)

    rhr_score = _rhr_component(daily_facts, factors)
    if rhr_score is not None:
        component_scores.append(rhr_score)

    sleep_score = _sleep_component(daily_facts, factors)
    if sleep_score is not None:
        component_scores.append(sleep_score)

    if activities:
        load_score = _load_component(activities, factors)
        if load_score is not None:
            component_scores.append(load_score)

    if not component_scores:
        return RecoveryScoreResult(
            score=None, source=None, iso_date=iso_date, available=False
        )

    total = max(0, min(100, int(round(statistics.mean(component_scores)))))
    return RecoveryScoreResult(
        score=total,
        source="calculated",
        iso_date=iso_date,
        available=True,
        factors=factors,
    )


def _hrv_component(daily_facts: list[dict], factors: dict) -> float | None:
    """HRV-компонент: сравнение последнего HRV с 7d-базой. Возвращает 0–100."""
    hrv_values = [
        f["hrv_rmssd_milli"]
        for f in daily_facts
        if f.get("hrv_rmssd_milli") is not None
    ]
    if len(hrv_values) < 2:
        return None

    baseline = statistics.mean(hrv_values[:-1][-7:])
    current = hrv_values[-1]

    if baseline == 0:
        return None

    # +20% от базы → 100, -20% → 0
    deviation = (current - baseline) / baseline
    score = max(0.0, min(100.0, 50.0 + deviation * 250.0))

    factors["hrv"] = {
        "current": round(current, 1),
        "baseline_7d": round(baseline, 1),
        "deviation_pct": round(deviation * 100, 1),
        "score": round(score, 1),
    }
    return score


def _rhr_component(daily_facts: list[dict], factors: dict) -> float | None:
    """RHR-компонент: отклонение пульса покоя от базы. Возвращает 0–100."""
    rhr_values = [
        f["resting_heart_rate"]
        for f in daily_facts
        if f.get("resting_heart_rate") is not None
    ]
    if len(rhr_values) < 2:
        return None

    baseline = statistics.mean(rhr_values[:-1][-7:])
    current = rhr_values[-1]

    if baseline == 0:
        return None

    # Каждый +1bpm от базы снижает score на 4 очка
    deviation_bpm = current - baseline
    score = max(0.0, min(100.0, 50.0 - deviation_bpm * 4.0))

    factors["rhr"] = {
        "current": round(current, 1),
        "baseline_7d": round(baseline, 1),
        "deviation_bpm": round(deviation_bpm, 1),
        "score": round(score, 1),
    }
    return score


def _sleep_component(daily_facts: list[dict], factors: dict) -> float | None:
    """Sleep-компонент: соотношение сна к целевым 8 часам. Возвращает 0–100."""
    sleep_values = [
        f["sleep_total_in_bed_milli"]
        for f in daily_facts
        if f.get("sleep_total_in_bed_milli") is not None
    ]
    if not sleep_values:
        return None

    current_ms = sleep_values[-1]
    target_ms = 8 * 3_600_000  # 8 часов

    score = min(100.0, (current_ms / target_ms) * 100.0)

    factors["sleep"] = {
        "sleep_hours": round(current_ms / 3_600_000, 2),
        "target_hours": 8.0,
        "score": round(score, 1),
    }
    return score


def _load_component(activities: list[dict], factors: dict) -> float | None:
    """Load-компонент: высокая нагрузка снижает recovery score."""
    from app.services.data_processing.training_load import compute_training_load

    load = compute_training_load(activities)
    if load.weekly_load == 0 and load.acute_chronic_ratio == 0:
        return None

    ratio = load.acute_chronic_ratio
    if ratio == 0:
        score = 80.0
    elif ratio <= 0.8:
        score = 85.0
    elif ratio <= 1.3:
        score = 70.0
    elif ratio <= 1.5:
        score = 45.0
    else:
        score = 25.0

    factors["training_load"] = {
        "acute_chronic_ratio": round(ratio, 2),
        "weekly_load": round(load.weekly_load, 1),
        "score": round(score, 1),
    }
    return score
