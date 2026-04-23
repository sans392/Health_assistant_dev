"""Summary Builder — структурированные сводки для Response Generator (Issue #56).

Задача модуля: вместо голого JSON-дампа `structured_result` формировать
предварительно агрегированные сводки с baseline-сравнением и аномалиями.
LLM перестаёт самостоятельно считать отклонения от нормы — эта работа
выполнена детерминированным кодом.

Основные функции:
    - build_metric_summary(facts, metric, baseline_facts=None) -> MetricSummary
    - build_activity_summary(activities) -> ActivityPromptSummary
    - annotate_anomalies(value, baseline_mean, baseline_std) -> list[str]

Форматирование в human-readable блок:
    - format_metric_summary(summary) — однострочная сводка по метрике
    - format_activity_summary(summary) — краткая сводка по активностям
    - format_structured_block(structured_result, baseline_facts=None) —
      главный entry-point, рендерит весь словарь результатов с fallback
      на truncated JSON для неизвестных структур.

Класс `ActivityPromptSummary` отличается от `ActivitySummary` из
`activity_summary.py` — тот предназначен для data processing (полная
статистика, streak/rest), а этот — только для подачи в промпт.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

# Подпись дефолтных метрик для автоматического рендеринга в промпт.
# Значения — (человекочитаемое имя, единица измерения, precision для среднего).
_DEFAULT_METRIC_LABELS: dict[str, tuple[str, str, int]] = {
    "steps": ("Шаги", "шагов", 0),
    "calories_kcal": ("Калории", "ккал", 0),
    "hrv_rmssd_milli": ("HRV (RMSSD)", "мс", 1),
    "resting_heart_rate": ("Пульс покоя", "уд/мин", 0),
    "recovery_score": ("Recovery (native)", "", 0),
    "recovery_score_calculated": ("Recovery (calc)", "", 0),
    "strain_score": ("Strain", "", 1),
    "spo2_percentage": ("SpO₂", "%", 1),
    "skin_temp_celsius": ("Темп. кожи", "°C", 1),
    "sleep_total_in_bed_milli": ("Сон", "мс", 0),
    "water_liters": ("Вода", "л", 1),
}

# Ключи, которые в tool_data / planner tool_results содержат daily-facts.
_DAILY_FACTS_KEYS = {"get_daily_facts", "daily_facts"}
# Ключи с активностями.
_ACTIVITIES_KEYS = {
    "get_activities",
    "get_activities_by_sport",
    "activities",
}
# Ключи-дубли RAG (их рендерит сам response_generator в отдельном блоке).
_RAG_KEYS = ("rag_retrieve",)

_FALLBACK_JSON_LIMIT = 500

_ANOMALY_SIGMA = 2.0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MetricSummary:
    """Сводка по одной числовой метрике за период.

    Поля `baseline_*` и `latest_*` заполняются только при достаточном
    количестве точек: baseline — не менее 2 точек в окне baseline.
    """

    metric: str
    count: int
    total: float
    mean: float
    min_value: float
    max_value: float
    latest_date: str | None = None
    latest_value: float | None = None
    baseline_mean: float | None = None
    baseline_std: float | None = None
    baseline_points: int = 0
    # Процентное отклонение среднего периода от baseline_mean.
    delta_pct: float | None = None
    # Флаги аномалий latest-значения (например ["ANOMALY"] при |Δ| > 2σ).
    anomaly_flags: list[str] = field(default_factory=list)
    unit: str = ""
    label: str = ""

    @property
    def has_baseline(self) -> bool:
        return self.baseline_mean is not None and self.baseline_points >= 2


@dataclass
class ActivityPromptSummary:
    """Сводка по активностям за период (для подачи в промпт).

    В отличие от `ActivitySummary` из `activity_summary.py`, содержит только
    поля, нужные response_generator'у: количество, суммарные величины и
    указатель на самую интенсивную тренировку.
    """

    total_activities: int
    total_duration_seconds: int
    total_calories: int
    total_distance_meters: float
    by_sport_counts: dict[str, int] = field(default_factory=dict)
    # Самая интенсивная тренировка (по calories, при равенстве — по длительности).
    most_intense: dict[str, Any] | None = None

    @property
    def total_duration_minutes(self) -> int:
        return self.total_duration_seconds // 60

    @property
    def total_distance_km(self) -> float:
        return round(self.total_distance_meters / 1000, 2)


# ---------------------------------------------------------------------------
# Core: anomalies
# ---------------------------------------------------------------------------


def annotate_anomalies(
    value: float,
    baseline_mean: float,
    baseline_std: float,
    sigma: float = _ANOMALY_SIGMA,
) -> list[str]:
    """Вернуть флаги аномалии для одиночного значения.

    Флаг выставляется, когда value выходит за `baseline_mean ± sigma * baseline_std`.
    Если baseline_std равен нулю — аномалия фиксируется только при строгом
    неравенстве с baseline_mean (защита от деления на ноль в δ %).

    Args:
        value: Измеренное значение.
        baseline_mean: Средняя baseline.
        baseline_std: Стандартное отклонение baseline.
        sigma: Количество σ для порога (по умолчанию 2).

    Returns:
        Пустой список, если отклонение в норме; иначе ["ANOMALY"] + направление.
    """
    flags: list[str] = []
    if baseline_std <= 0:
        # Вырожденный случай — baseline константный.
        if value != baseline_mean:
            flags.append("ANOMALY")
            flags.append("above_baseline" if value > baseline_mean else "below_baseline")
        return flags

    delta = value - baseline_mean
    if abs(delta) > sigma * baseline_std:
        flags.append("ANOMALY")
        flags.append("above_baseline" if delta > 0 else "below_baseline")
    return flags


# ---------------------------------------------------------------------------
# Core: metric summary
# ---------------------------------------------------------------------------


def _extract_series(facts: list[dict], metric: str) -> list[tuple[str, float]]:
    """Вытащить отсортированный по дате ряд (iso_date, value) для метрики.

    Игнорирует записи без iso_date или с нечисловым значением.
    """
    series: list[tuple[str, float]] = []
    for fact in facts or []:
        iso_date = fact.get("iso_date")
        value = fact.get(metric)
        if not iso_date or value is None:
            continue
        try:
            series.append((iso_date, float(value)))
        except (TypeError, ValueError):
            continue
    series.sort(key=lambda p: p[0])
    return series


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float], mean_val: float | None = None) -> float:
    """Стандартное отклонение (sample, ddof=0) — для baseline-аномалий."""
    if len(values) < 2:
        return 0.0
    m = _mean(values) if mean_val is None else mean_val
    variance = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def build_metric_summary(
    facts: list[dict],
    metric: str,
    baseline_facts: list[dict] | None = None,
) -> MetricSummary:
    """Построить сводку по числовой метрике.

    Если `baseline_facts` переданы, они используются как источник baseline;
    иначе baseline считается из самой же серии `facts`, исключая самую
    свежую точку (нужно минимум 3 точки, чтобы получить осмысленное
    сравнение latest vs baseline).

    Args:
        facts: Список dict'ов из get_daily_facts (должны содержать iso_date).
        metric: Имя поля метрики (например 'hrv_rmssd_milli').
        baseline_facts: Опциональный расширенный window для baseline.

    Returns:
        MetricSummary с агрегатами, baseline-сравнением и anomaly_flags.
    """
    label, unit, _ = _DEFAULT_METRIC_LABELS.get(metric, (metric, "", 2))
    series = _extract_series(facts, metric)

    if not series:
        return MetricSummary(
            metric=metric, count=0, total=0.0, mean=0.0,
            min_value=0.0, max_value=0.0, label=label, unit=unit,
        )

    values = [v for _, v in series]
    total = sum(values)
    mean_val = _mean(values)
    latest_date, latest_value = series[-1]

    # Baseline: либо явно переданный набор, либо окно внутри самой серии.
    if baseline_facts is not None:
        baseline_series = _extract_series(baseline_facts, metric)
        # Исключаем точки, которые точно входят в «текущее» окно (по latest_date).
        baseline_values = [
            v for d, v in baseline_series if d != latest_date
        ]
    else:
        baseline_values = [v for d, v in series if d != latest_date]

    baseline_mean: float | None = None
    baseline_std: float | None = None
    baseline_points = len(baseline_values)
    delta_pct: float | None = None
    anomaly_flags: list[str] = []

    if baseline_points >= 2:
        baseline_mean = _mean(baseline_values)
        baseline_std = _std(baseline_values, baseline_mean)
        if baseline_mean != 0:
            delta_pct = round(
                (mean_val - baseline_mean) / abs(baseline_mean) * 100, 1
            )
        anomaly_flags = annotate_anomalies(latest_value, baseline_mean, baseline_std)

    return MetricSummary(
        metric=metric,
        count=len(values),
        total=round(total, 2),
        mean=round(mean_val, 2),
        min_value=round(min(values), 2),
        max_value=round(max(values), 2),
        latest_date=latest_date,
        latest_value=round(latest_value, 2),
        baseline_mean=round(baseline_mean, 2) if baseline_mean is not None else None,
        baseline_std=round(baseline_std, 2) if baseline_std is not None else None,
        baseline_points=baseline_points,
        delta_pct=delta_pct,
        anomaly_flags=anomaly_flags,
        unit=unit,
        label=label,
    )


# ---------------------------------------------------------------------------
# Core: activity summary
# ---------------------------------------------------------------------------


def _activity_intensity(act: dict) -> float:
    """Скор «интенсивности» тренировки для выбора «самой интенсивной».

    Приоритет: calories > duration_seconds > distance_meters. Чтобы
    сравнение работало при частичных данных — складываем нормированные
    компоненты в один скор.
    """
    calories = float(act.get("calories") or 0)
    duration = float(act.get("duration_seconds") or 0)
    distance = float(act.get("distance_meters") or 0)
    # Калории — самый надёжный показатель. Продолжительность/дистанция —
    # вспомогательные, с меньшими весами.
    return calories + duration * 0.05 + distance * 0.001


def build_activity_summary(activities: list[dict]) -> ActivityPromptSummary:
    """Построить краткую сводку по списку активностей.

    Args:
        activities: Список dict'ов (из get_activities).

    Returns:
        ActivityPromptSummary с агрегатами и самой интенсивной тренировкой.
    """
    if not activities:
        return ActivityPromptSummary(
            total_activities=0,
            total_duration_seconds=0,
            total_calories=0,
            total_distance_meters=0.0,
        )

    total_duration = 0
    total_calories = 0
    total_distance = 0.0
    by_sport: dict[str, int] = {}

    most_intense: dict[str, Any] | None = None
    best_score = -1.0

    for act in activities:
        total_duration += int(act.get("duration_seconds") or 0)
        total_calories += int(act.get("calories") or 0)
        total_distance += float(act.get("distance_meters") or 0.0)
        sport = act.get("sport_type", "other") or "other"
        by_sport[sport] = by_sport.get(sport, 0) + 1

        score = _activity_intensity(act)
        if score > best_score:
            best_score = score
            most_intense = {
                "title": act.get("title"),
                "sport_type": sport,
                "start_time": act.get("start_time"),
                "duration_seconds": int(act.get("duration_seconds") or 0),
                "calories": int(act.get("calories") or 0),
                "distance_meters": float(act.get("distance_meters") or 0.0),
                "avg_heart_rate": act.get("avg_heart_rate"),
            }

    return ActivityPromptSummary(
        total_activities=len(activities),
        total_duration_seconds=total_duration,
        total_calories=total_calories,
        total_distance_meters=total_distance,
        by_sport_counts=by_sport,
        most_intense=most_intense,
    )


# ---------------------------------------------------------------------------
# Formatting (human-readable)
# ---------------------------------------------------------------------------


def _format_number(value: float, precision: int) -> str:
    """Форматирует число с заданной точностью и разделителями разрядов."""
    if precision == 0:
        return f"{int(round(value)):,}".replace(",", " ")
    return f"{value:,.{precision}f}".replace(",", " ")


def format_metric_summary(summary: MetricSummary) -> str:
    """Отрендерить MetricSummary в одну-две строки для промпта."""
    if summary.count == 0:
        return f"- {summary.label or summary.metric}: данных нет."

    precision = _DEFAULT_METRIC_LABELS.get(summary.metric, (summary.label, summary.unit, 2))[2]
    unit = f" {summary.unit}" if summary.unit else ""
    label = summary.label or summary.metric

    parts: list[str] = []
    # Главная строка: агрегаты за период.
    total_str = _format_number(summary.total, precision)
    mean_str = _format_number(summary.mean, precision)
    line = (
        f"- {label} за {summary.count} дн.: сумма {total_str}{unit}, "
        f"среднее {mean_str}{unit}/день "
        f"(min {_format_number(summary.min_value, precision)}, "
        f"max {_format_number(summary.max_value, precision)})"
    )

    if summary.has_baseline and summary.baseline_mean is not None:
        base_str = _format_number(summary.baseline_mean, precision)
        delta_str = ""
        if summary.delta_pct is not None:
            sign = "+" if summary.delta_pct >= 0 else ""
            delta_str = f" → {sign}{summary.delta_pct}% от базы"
        line += f"; базовое {base_str}{unit}{delta_str}"

    parts.append(line)

    # Вторая строка: latest + аномалия.
    if summary.latest_value is not None and summary.latest_date:
        latest_str = _format_number(summary.latest_value, precision)
        latest_line = (
            f"  · Последнее ({summary.latest_date}): {latest_str}{unit}"
        )
        if summary.has_baseline and summary.baseline_mean is not None and summary.baseline_mean != 0:
            delta_latest = (
                (summary.latest_value - summary.baseline_mean) / abs(summary.baseline_mean) * 100
            )
            sign = "+" if delta_latest >= 0 else ""
            latest_line += f" ({sign}{round(delta_latest, 1)}% от базы)"
        if summary.anomaly_flags:
            latest_line += f" [{', '.join(summary.anomaly_flags)}]"
        parts.append(latest_line)

    return "\n".join(parts)


def format_activity_summary(summary: ActivityPromptSummary) -> str:
    """Отрендерить ActivityPromptSummary в блок текста."""
    if summary.total_activities == 0:
        return "- Тренировок за период: 0."

    parts: list[str] = []
    duration_min = summary.total_duration_minutes
    distance_km = summary.total_distance_km
    by_sport_str = ", ".join(
        f"{sport}×{count}" for sport, count in sorted(summary.by_sport_counts.items())
    )
    parts.append(
        f"- Тренировок: {summary.total_activities} "
        f"(суммарно {duration_min} мин, {summary.total_calories} ккал, {distance_km} км)"
    )
    if by_sport_str:
        parts.append(f"  · По видам: {by_sport_str}")

    mi = summary.most_intense
    if mi:
        title = mi.get("title") or mi.get("sport_type") or "тренировка"
        start = (mi.get("start_time") or "")[:10]
        dur_min = int(mi.get("duration_seconds", 0)) // 60
        cal = mi.get("calories") or 0
        hr = mi.get("avg_heart_rate")
        hr_str = f", ЧСС {hr}" if hr else ""
        date_str = f" ({start})" if start else ""
        parts.append(
            f"  · Самая интенсивная{date_str}: {title} — {dur_min} мин, {cal} ккал{hr_str}"
        )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Entry point: render structured_result as a human-readable block
# ---------------------------------------------------------------------------


def _looks_like_daily_facts(value: Any) -> bool:
    """Эвристика: это список дневных фактов?"""
    if not isinstance(value, list) or not value:
        return False
    first = value[0]
    return isinstance(first, dict) and "iso_date" in first


def _looks_like_activities(value: Any) -> bool:
    """Эвристика: это список активностей?"""
    if not isinstance(value, list) or not value:
        return False
    first = value[0]
    return isinstance(first, dict) and (
        "sport_type" in first or "start_time" in first and "duration_seconds" in first
    )


def _format_daily_facts_block(
    facts: list[dict],
    baseline_facts: list[dict] | None,
    metrics: list[str] | None = None,
) -> str:
    """Отрендерить блок метрик по списку фактов.

    Если metrics не указан — берутся все метрики из _DEFAULT_METRIC_LABELS,
    у которых в facts есть хотя бы одно ненулевое значение.
    """
    chosen = metrics or [
        m for m in _DEFAULT_METRIC_LABELS if any(f.get(m) is not None for f in facts)
    ]
    if not chosen:
        return "- Метрик нет."

    lines: list[str] = []
    for metric in chosen:
        summary = build_metric_summary(facts, metric, baseline_facts=baseline_facts)
        if summary.count == 0:
            continue
        lines.append(format_metric_summary(summary))
    return "\n".join(lines) if lines else "- Метрик нет."


def _truncated_json(value: Any, limit: int = _FALLBACK_JSON_LIMIT) -> str:
    """Fallback: сериализация произвольного значения в усечённый JSON."""
    try:
        text = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(value)
    if len(text) > limit:
        return text[:limit] + "…"
    return text


def _format_value(
    key: str,
    value: Any,
    baseline_facts: list[dict] | None,
) -> str:
    """Отрендерить одно значение из structured_result."""
    # Специальные случаи по ключу.
    if any(k in key for k in _RAG_KEYS):
        # RAG-чанки рендерит сам response_generator, здесь пропускаем.
        return ""

    if key in _DAILY_FACTS_KEYS or _looks_like_daily_facts(value):
        return _format_daily_facts_block(value, baseline_facts)

    if key in _ACTIVITIES_KEYS or _looks_like_activities(value):
        summary = build_activity_summary(value)
        return format_activity_summary(summary)

    # Вложенные dict (например tool_data / processed) — рекурсивно.
    if isinstance(value, dict):
        return _format_dict_block(value, baseline_facts, depth=1)

    # Fallback — truncated JSON.
    return "- " + _truncated_json(value)


def _format_dict_block(
    data: dict,
    baseline_facts: list[dict] | None,
    depth: int,
) -> str:
    """Отрендерить словарь верхнего уровня (или вложенный tool_data/processed)."""
    lines: list[str] = []
    indent = "  " * depth
    for key, value in data.items():
        # Пропускаем RAG-ключи — их рендерит отдельный блок response_generator.
        if any(k in key for k in _RAG_KEYS):
            continue
        rendered = _format_value(key, value, baseline_facts)
        if not rendered:
            continue
        header = f"{indent}### {key}"
        lines.append(header)
        # Добавляем отступ к каждой строке рендеринга, сохраняя переносы.
        for sub in rendered.splitlines():
            lines.append(f"{indent}{sub}" if sub else sub)
    return "\n".join(lines)


def format_structured_block(
    structured_result: dict | None,
    baseline_facts: list[dict] | None = None,
) -> str:
    """Главный рендер `structured_result` → human-readable блок.

    Обходит верхний уровень dict, для каждого ключа выбирает подходящий
    форматтер (daily_facts, activities, вложенный dict) или падает на
    truncated JSON.

    Args:
        structured_result: Собранные результаты tool/template/planner.
        baseline_facts: (Опционально) 30-дневное окно daily_facts для baseline.

    Returns:
        Готовый текст для подстановки в system prompt. Пустая строка,
        если структура не содержит полезного.
    """
    if not structured_result:
        return ""

    lines: list[str] = []
    for key, value in structured_result.items():
        if any(k in key for k in _RAG_KEYS):
            continue
        rendered = _format_value(key, value, baseline_facts)
        if not rendered:
            continue
        lines.append(f"### {key}")
        lines.append(rendered)

    return "\n".join(lines).strip()
