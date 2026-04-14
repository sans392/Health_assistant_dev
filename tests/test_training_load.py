"""Тесты модуля training_load (app/services/data_processing/training_load.py)."""

from datetime import date, timedelta

import pytest

from app.services.data_processing.training_load import (
    SPORT_COEFFICIENTS,
    TrainingLoad,
    compute_training_load,
)


def _make_activity(
    sport_type: str = "running",
    duration_seconds: int = 3600,
    calories: int = 500,
    start_date: str = "2026-04-10",
) -> dict:
    return {
        "sport_type": sport_type,
        "duration_seconds": duration_seconds,
        "calories": calories,
        "start_time": f"{start_date}T10:00:00",
    }


class TestComputeTrainingLoad:
    """Тесты compute_training_load."""

    def test_empty_list_returns_zeros(self) -> None:
        load = compute_training_load([])
        assert load.weekly_load == 0.0
        assert load.chronic_load == 0.0
        assert load.acute_chronic_ratio == 0.0
        assert load.daily_load == {}

    def test_single_running_activity(self) -> None:
        today = date.today()
        activities = [_make_activity(
            sport_type="running",
            duration_seconds=3600,
            calories=500,
            start_date=today.isoformat(),
        )]
        load = compute_training_load(activities, reference_date=today)

        # Ожидаемая нагрузка: 3600 * 1.2 + 500 * 0.1 = 4320 + 50 = 4370
        expected = 3600 * 1.2 + 500 * 0.1
        assert abs(load.weekly_load - expected) < 0.1
        assert today.isoformat() in load.daily_load

    def test_sport_coefficients_applied(self) -> None:
        today = date.today()
        for sport, coeff in SPORT_COEFFICIENTS.items():
            activities = [_make_activity(
                sport_type=sport,
                duration_seconds=1000,
                calories=0,
                start_date=today.isoformat(),
            )]
            load = compute_training_load(activities, reference_date=today)
            expected = 1000 * coeff
            assert abs(load.weekly_load - expected) < 0.1, f"Ошибка для спорта {sport}"

    def test_acute_chronic_ratio(self) -> None:
        today = date.today()

        # Acute: тренировки за последние 7 дней
        acute_activities = [
            _make_activity(start_date=(today - timedelta(days=i)).isoformat(),
                           duration_seconds=3600, calories=0, sport_type="gym")
            for i in range(7)
        ]
        # Chronic: тренировки 7–14 дней назад
        chronic_activities = [
            _make_activity(start_date=(today - timedelta(days=7 + i)).isoformat(),
                           duration_seconds=1800, calories=0, sport_type="gym")
            for i in range(7)
        ]

        all_activities = acute_activities + chronic_activities
        load = compute_training_load(all_activities, reference_date=today)

        # Acute нагрузка > Chronic нагрузки (больше длительность)
        assert load.weekly_load > load.chronic_load
        assert load.acute_chronic_ratio > 1.0

    def test_zero_chronic_gives_zero_ratio(self) -> None:
        today = date.today()
        activities = [_make_activity(
            start_date=today.isoformat(),
            duration_seconds=3600,
            calories=0,
        )]
        load = compute_training_load(activities, reference_date=today)
        # Chronic = 0 → ratio = 0.0
        assert load.acute_chronic_ratio == 0.0

    def test_activities_outside_range_not_counted(self) -> None:
        today = date.today()
        # Активность 30 дней назад — не входит в acute (7 дней) и chronic (7–14 дней)
        activities = [_make_activity(
            start_date=(today - timedelta(days=30)).isoformat(),
            duration_seconds=9999,
            calories=9999,
        )]
        load = compute_training_load(activities, reference_date=today)
        assert load.weekly_load == 0.0
        assert load.chronic_load == 0.0
