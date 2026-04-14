"""Тесты модуля trend_analyzer (app/services/data_processing/trend_analyzer.py)."""

from datetime import date, timedelta

import pytest

from app.services.data_processing.trend_analyzer import (
    TrendResult,
    analyze_trend,
    build_time_series_from_activities,
    build_time_series_from_facts,
)


def _make_series(today: date, recent_value: float, baseline_value: float) -> dict[str, float]:
    """Создать тестовый временной ряд с заданными средними значениями."""
    series = {}
    # Заполняем recent (последние 7 дней)
    for i in range(7):
        d = today - timedelta(days=i)
        series[d.isoformat()] = recent_value
    # Заполняем baseline (7–13 дней назад)
    for i in range(7, 14):
        d = today - timedelta(days=i)
        series[d.isoformat()] = baseline_value
    return series


class TestAnalyzeTrend:
    """Тесты analyze_trend."""

    def test_empty_series_returns_stable(self) -> None:
        result = analyze_trend({})
        assert result.direction == "stable"
        assert result.change_percent == 0.0

    def test_upward_trend(self) -> None:
        today = date.today()
        series = _make_series(today, recent_value=100.0, baseline_value=80.0)
        result = analyze_trend(series, reference_date=today)
        assert result.direction == "up"
        assert result.change_percent > 0

    def test_downward_trend(self) -> None:
        today = date.today()
        series = _make_series(today, recent_value=60.0, baseline_value=100.0)
        result = analyze_trend(series, reference_date=today)
        assert result.direction == "down"
        assert result.change_percent < 0

    def test_stable_within_threshold(self) -> None:
        today = date.today()
        # Изменение 3% — ниже порога 5%
        series = _make_series(today, recent_value=103.0, baseline_value=100.0)
        result = analyze_trend(series, reference_date=today, stable_threshold_pct=5.0)
        assert result.direction == "stable"

    def test_stable_threshold_custom(self) -> None:
        today = date.today()
        # Изменение 8% — выше порога 5%, ниже 10%
        series = _make_series(today, recent_value=108.0, baseline_value=100.0)
        result_5 = analyze_trend(series, reference_date=today, stable_threshold_pct=5.0)
        result_10 = analyze_trend(series, reference_date=today, stable_threshold_pct=10.0)
        assert result_5.direction == "up"
        assert result_10.direction == "stable"

    def test_data_points_counted(self) -> None:
        today = date.today()
        series = _make_series(today, recent_value=100.0, baseline_value=90.0)
        result = analyze_trend(series, reference_date=today)
        assert result.data_points_recent == 7
        assert result.data_points_baseline == 7

    def test_no_baseline_data(self) -> None:
        """Только recent данные — baseline_avg = 0, изменение = 100%."""
        today = date.today()
        series = {today.isoformat(): 100.0}
        result = analyze_trend(series, reference_date=today)
        assert result.direction == "up"
        assert result.change_percent == 100.0
        assert result.data_points_baseline == 0

    def test_change_percent_calculation(self) -> None:
        today = date.today()
        series = _make_series(today, recent_value=120.0, baseline_value=100.0)
        result = analyze_trend(series, reference_date=today)
        assert abs(result.change_percent - 20.0) < 0.1

    def test_averages_calculated(self) -> None:
        today = date.today()
        series = _make_series(today, recent_value=50.0, baseline_value=30.0)
        result = analyze_trend(series, reference_date=today)
        assert result.recent_avg == 50.0
        assert result.baseline_avg == 30.0


class TestBuildTimeSeries:
    """Тесты построения временных рядов."""

    def test_build_from_facts(self) -> None:
        facts = [
            {"iso_date": "2026-04-10", "steps": 8000, "recovery_score": 75},
            {"iso_date": "2026-04-11", "steps": 10000, "recovery_score": None},
        ]
        series = build_time_series_from_facts(facts, "steps")
        assert series == {"2026-04-10": 8000.0, "2026-04-11": 10000.0}

    def test_build_from_facts_skips_none(self) -> None:
        facts = [
            {"iso_date": "2026-04-10", "recovery_score": None},
            {"iso_date": "2026-04-11", "recovery_score": 80},
        ]
        series = build_time_series_from_facts(facts, "recovery_score")
        assert "2026-04-10" not in series
        assert series["2026-04-11"] == 80.0

    def test_build_from_activities_sum_metric(self) -> None:
        activities = [
            {"start_time": "2026-04-10T10:00:00", "duration_seconds": 3600},
            {"start_time": "2026-04-10T16:00:00", "duration_seconds": 1800},  # тот же день
        ]
        series = build_time_series_from_activities(activities, "duration_seconds")
        # Суммируем за день
        assert series["2026-04-10"] == 5400.0

    def test_build_from_activities_avg_metric(self) -> None:
        activities = [
            {"start_time": "2026-04-10T10:00:00", "avg_heart_rate": 140},
            {"start_time": "2026-04-10T16:00:00", "avg_heart_rate": 160},
        ]
        series = build_time_series_from_activities(activities, "avg_heart_rate")
        # Усредняем avg_heart_rate
        assert series["2026-04-10"] == 150.0
