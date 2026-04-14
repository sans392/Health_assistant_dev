"""Тесты модуля activity_summary (app/services/data_processing/activity_summary.py)."""

import pytest

from app.services.data_processing.activity_summary import (
    ActivitySummary,
    compute_activity_summary,
)


def _make_activity(
    sport_type: str = "running",
    duration_seconds: int = 3600,
    calories: int = 500,
    distance_meters: float = 10000.0,
    start_time: str = "2026-04-10T10:00:00",
) -> dict:
    return {
        "id": "test-id",
        "user_id": "user-1",
        "title": "Test activity",
        "sport_type": sport_type,
        "duration_seconds": duration_seconds,
        "calories": calories,
        "distance_meters": distance_meters,
        "start_time": start_time,
        "end_time": "2026-04-10T11:00:00",
        "avg_speed": None,
        "avg_heart_rate": None,
        "source": "manual",
    }


class TestComputeActivitySummary:
    """Тесты compute_activity_summary."""

    def test_empty_list_returns_zeros(self) -> None:
        summary = compute_activity_summary([])
        assert summary.total_activities == 0
        assert summary.total_duration_seconds == 0
        assert summary.total_calories == 0
        assert summary.total_distance_meters == 0.0
        assert summary.by_sport == {}

    def test_single_activity_counts(self) -> None:
        activities = [_make_activity(
            sport_type="running",
            duration_seconds=1800,
            calories=300,
            distance_meters=5000.0,
        )]
        summary = compute_activity_summary(activities)
        assert summary.total_activities == 1
        assert summary.total_duration_seconds == 1800
        assert summary.total_calories == 300
        assert summary.total_distance_meters == 5000.0

    def test_multiple_sports_breakdown(self) -> None:
        activities = [
            _make_activity(sport_type="running", duration_seconds=3600, calories=500,
                           distance_meters=10000.0, start_time="2026-04-10T10:00:00"),
            _make_activity(sport_type="running", duration_seconds=1800, calories=250,
                           distance_meters=5000.0, start_time="2026-04-11T10:00:00"),
            _make_activity(sport_type="gym", duration_seconds=3600, calories=400,
                           distance_meters=0.0, start_time="2026-04-12T10:00:00"),
        ]
        summary = compute_activity_summary(activities)

        assert summary.total_activities == 3
        assert summary.total_duration_seconds == 9000
        assert summary.total_calories == 1150

        assert "running" in summary.by_sport
        assert summary.by_sport["running"].count == 2
        assert summary.by_sport["running"].total_duration_seconds == 5400

        assert "gym" in summary.by_sport
        assert summary.by_sport["gym"].count == 1

    def test_total_duration_minutes(self) -> None:
        activities = [_make_activity(duration_seconds=3600)]
        summary = compute_activity_summary(activities)
        assert summary.total_duration_minutes == 60

    def test_total_distance_km(self) -> None:
        activities = [_make_activity(distance_meters=10000.0)]
        summary = compute_activity_summary(activities)
        assert summary.total_distance_km == 10.0

    def test_streak_consecutive_days(self) -> None:
        activities = [
            _make_activity(start_time="2026-04-08T10:00:00"),
            _make_activity(start_time="2026-04-09T10:00:00"),
            _make_activity(start_time="2026-04-10T10:00:00"),
        ]
        summary = compute_activity_summary(activities)
        assert summary.streak_days == 3

    def test_streak_non_consecutive(self) -> None:
        activities = [
            _make_activity(start_time="2026-04-08T10:00:00"),
            _make_activity(start_time="2026-04-10T10:00:00"),  # пропуск 9го
            _make_activity(start_time="2026-04-11T10:00:00"),
        ]
        summary = compute_activity_summary(activities)
        assert summary.streak_days == 2  # 10–11

    def test_rest_days_counted(self) -> None:
        activities = [
            _make_activity(start_time="2026-04-08T10:00:00"),
            _make_activity(start_time="2026-04-10T10:00:00"),  # 9е — отдых
        ]
        summary = compute_activity_summary(activities)
        # period: 8–10, total_days=3, training=2, rest=1
        assert summary.rest_days == 1

    def test_none_values_handled(self) -> None:
        """Поля None не ломают агрегацию."""
        activity = _make_activity()
        activity["calories"] = None
        activity["distance_meters"] = None
        summary = compute_activity_summary([activity])
        assert summary.total_calories == 0
        assert summary.total_distance_meters == 0.0
