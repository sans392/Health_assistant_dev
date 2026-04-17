"""Модуль определения рисков перетренированности (Phase 2, Issue #27).

Маркеры согласно архитектуре v2:
- HRV drop > 10% vs 7d baseline
- RHR elevation > 5 bpm vs baseline
- Снижение качества сна > 15%
- Высокий acute/chronic ratio > 1.5
"""

import statistics
from dataclasses import dataclass, field


@dataclass
class OvertrainingResult:
    """Результат анализа рисков перетренированности."""

    risk_level: str                               # "low" | "medium" | "high"
    markers_triggered: list[str] = field(default_factory=list)
    recommendation: str = ""


def detect_overtraining(
    daily_facts: list[dict],
    activities: list[dict] | None = None,
) -> OvertrainingResult:
    """Определить риски перетренированности по маркерам.

    Args:
        daily_facts: Данные daily_facts за последние 14+ дней (по возрастанию даты).
        activities: Список тренировок для проверки нагрузки (опционально).

    Returns:
        OvertrainingResult с уровнем риска, маркерами и рекомендацией.
    """
    if not daily_facts:
        return OvertrainingResult(
            risk_level="low",
            recommendation="Недостаточно данных для анализа перетренированности.",
        )

    markers: list[str] = []

    _check_hrv(daily_facts, markers)
    _check_rhr(daily_facts, markers)
    _check_sleep(daily_facts, markers)
    if activities:
        _check_load(activities, markers)

    count = len(markers)
    if count == 0:
        risk_level = "low"
        recommendation = (
            "Признаков перетренированности не обнаружено. Продолжайте тренировки в том же режиме."
        )
    elif count == 1:
        risk_level = "low"
        recommendation = (
            "Один маркер усталости. Следите за самочувствием, обеспечьте достаточный сон и восстановление."
        )
    elif count == 2:
        risk_level = "medium"
        recommendation = (
            "Несколько маркеров перетренированности. Рекомендуется снизить интенсивность тренировок "
            "и добавить 1–2 дня активного восстановления."
        )
    else:
        risk_level = "high"
        recommendation = (
            "Высокий риск перетренированности. Рекомендуется взять 2–3 дня полного отдыха, "
            "увеличить продолжительность сна и проконсультироваться с тренером."
        )

    return OvertrainingResult(
        risk_level=risk_level,
        markers_triggered=markers,
        recommendation=recommendation,
    )


def _check_hrv(daily_facts: list[dict], markers: list[str]) -> None:
    """HRV drop > 10% vs 7d baseline."""
    hrv_values = [
        f["hrv_rmssd_milli"]
        for f in daily_facts
        if f.get("hrv_rmssd_milli") is not None
    ]
    if len(hrv_values) < 3:
        return

    baseline = statistics.mean(hrv_values[:-1][-7:])
    current = hrv_values[-1]

    if baseline > 0 and (baseline - current) / baseline > 0.10:
        drop_pct = round((baseline - current) / baseline * 100, 1)
        markers.append(f"HRV снижен на {drop_pct}% относительно 7d-baseline")


def _check_rhr(daily_facts: list[dict], markers: list[str]) -> None:
    """RHR elevation > 5 bpm vs baseline."""
    rhr_values = [
        f["resting_heart_rate"]
        for f in daily_facts
        if f.get("resting_heart_rate") is not None
    ]
    if len(rhr_values) < 3:
        return

    baseline = statistics.mean(rhr_values[:-1][-7:])
    current = rhr_values[-1]

    if current - baseline > 5:
        markers.append(
            f"ЧСС покоя повышен на {round(current - baseline, 1)} уд/мин относительно baseline"
        )


def _check_sleep(daily_facts: list[dict], markers: list[str]) -> None:
    """Снижение сна > 15% vs 7d baseline."""
    sleep_values = [
        f["sleep_total_in_bed_milli"]
        for f in daily_facts
        if f.get("sleep_total_in_bed_milli") is not None
    ]
    if len(sleep_values) < 3:
        return

    baseline = statistics.mean(sleep_values[:-1][-7:])
    current = sleep_values[-1]

    if baseline > 0 and (baseline - current) / baseline > 0.15:
        hours_drop = round((baseline - current) / 3_600_000, 2)
        markers.append(f"Сон сократился на {hours_drop}ч относительно 7d-baseline")


def _check_load(activities: list[dict], markers: list[str]) -> None:
    """Acute/chronic ratio > 1.5."""
    from app.services.data_processing.training_load import compute_training_load

    load = compute_training_load(activities)
    if load.acute_chronic_ratio > 1.5:
        markers.append(
            f"Острая нагрузка превышает хроническую в {round(load.acute_chronic_ratio, 2)}x"
        )
