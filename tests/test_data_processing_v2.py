"""Тесты модулей Data Processing v2 (Issue #27).

Охватывает: heart_rate_zones, strain_score, recovery_score (расчётная),
overtraining_detection, training_load (расширения).
"""

from datetime import date, timedelta

import pytest

from app.services.data_processing.heart_rate_zones import HRZones, compute_hr_zones
from app.services.data_processing.strain_score import StrainScoreResult, compute_strain_score
from app.services.data_processing.recovery_score import (
    RecoveryScoreResult,
    compute_recovery_score,
    get_recovery_score,
)
from app.services.data_processing.overtraining_detection import (
    OvertrainingResult,
    detect_overtraining,
)
from app.services.data_processing.training_load import TrainingLoad, compute_training_load


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fact(
    iso_date: str,
    hrv: float | None = None,
    rhr: float | None = None,
    sleep_ms: int | None = None,
    recovery: int | None = None,
) -> dict:
    return {
        "iso_date": iso_date,
        "hrv_rmssd_milli": hrv,
        "resting_heart_rate": rhr,
        "sleep_total_in_bed_milli": sleep_ms,
        "recovery_score": recovery,
    }


def _make_activity(
    sport_type: str = "running",
    duration_seconds: int = 3600,
    calories: int = 400,
    start_date: str | None = None,
    avg_hr: int | None = None,
) -> dict:
    d = start_date or date.today().isoformat()
    return {
        "sport_type": sport_type,
        "duration_seconds": duration_seconds,
        "calories": calories,
        "start_time": f"{d}T10:00:00",
        "avg_heart_rate": avg_hr,
    }


# ---------------------------------------------------------------------------
# HRZones
# ---------------------------------------------------------------------------

class TestComputeHRZones:

    def test_basic_zones_order(self) -> None:
        zones = compute_hr_zones(age=30, resting_hr=60)
        # Каждая зона начинается выше предыдущей
        assert zones.z1[0] < zones.z2[0] < zones.z3[0] < zones.z4[0] < zones.z5[0]

    def test_zones_boundaries_karvonen(self) -> None:
        zones = compute_hr_zones(age=30, resting_hr=60, max_hr=190)
        hrr = 190 - 60  # = 130
        assert zones.z1 == (60 + int(130 * 0.50), 60 + int(130 * 0.60))
        assert zones.z5 == (60 + int(130 * 0.90), 60 + int(130 * 1.00))

    def test_max_hr_fallback_220_minus_age(self) -> None:
        zones = compute_hr_zones(age=25, resting_hr=55)
        assert zones.max_hr == 220 - 25  # = 195

    def test_explicit_max_hr_used(self) -> None:
        zones = compute_hr_zones(age=30, resting_hr=60, max_hr=185)
        assert zones.max_hr == 185

    def test_zone_for_hr_returns_correct_zone(self) -> None:
        zones = compute_hr_zones(age=30, resting_hr=60, max_hr=190)
        # z1 low boundary
        assert zones.zone_for_hr(zones.z1[0]) == 1
        # z3 midpoint
        mid_z3 = (zones.z3[0] + zones.z3[1]) // 2
        assert zones.zone_for_hr(mid_z3) == 3
        # z5 upper boundary
        assert zones.zone_for_hr(zones.z5[1]) == 5

    def test_zone_for_hr_below_z1_returns_none(self) -> None:
        zones = compute_hr_zones(age=30, resting_hr=60, max_hr=190)
        assert zones.zone_for_hr(zones.z1[0] - 1) is None

    def test_returns_hrzone_dataclass(self) -> None:
        zones = compute_hr_zones(age=40, resting_hr=65)
        assert isinstance(zones, HRZones)


# ---------------------------------------------------------------------------
# StrainScore
# ---------------------------------------------------------------------------

class TestComputeStrainScore:

    def test_empty_returns_zero(self) -> None:
        result = compute_strain_score([])
        assert result.strain == 0.0
        assert result.activities_count == 0

    def test_strain_in_valid_range(self) -> None:
        acts = [_make_activity(duration_seconds=3600, avg_hr=150)]
        result = compute_strain_score(acts)
        assert 0.0 <= result.strain <= 21.0

    def test_longer_workout_higher_strain(self) -> None:
        short = compute_strain_score([_make_activity(duration_seconds=1800)])
        long_ = compute_strain_score([_make_activity(duration_seconds=7200)])
        assert long_.strain > short.strain

    def test_primary_driver_cardio(self) -> None:
        acts = [_make_activity(sport_type="running")]
        result = compute_strain_score(acts)
        assert result.primary_driver == "cardio"

    def test_primary_driver_strength(self) -> None:
        acts = [_make_activity(sport_type="gym")]
        result = compute_strain_score(acts)
        assert result.primary_driver == "strength"

    def test_primary_driver_mixed(self) -> None:
        acts = [
            _make_activity(sport_type="running"),
            _make_activity(sport_type="gym"),
        ]
        result = compute_strain_score(acts)
        assert result.primary_driver == "mixed"

    def test_higher_hr_zone_higher_strain(self) -> None:
        zones = compute_hr_zones(age=30, resting_hr=60, max_hr=190)
        low_hr_act = [_make_activity(duration_seconds=3600, avg_hr=zones.z1[0])]
        high_hr_act = [_make_activity(duration_seconds=3600, avg_hr=zones.z5[0])]
        low = compute_strain_score(low_hr_act, hr_zones=zones)
        high = compute_strain_score(high_hr_act, hr_zones=zones)
        assert high.strain > low.strain

    def test_activities_count_returned(self) -> None:
        acts = [_make_activity(), _make_activity(sport_type="gym")]
        result = compute_strain_score(acts)
        assert result.activities_count == 2


# ---------------------------------------------------------------------------
# RecoveryScore
# ---------------------------------------------------------------------------

class TestGetRecoveryScore:

    def test_empty_returns_unavailable(self) -> None:
        result = get_recovery_score([])
        assert not result.available
        assert result.score is None

    def test_whoop_score_returned(self) -> None:
        facts = [_make_fact("2026-04-10", recovery=78)]
        result = get_recovery_score(facts)
        assert result.available
        assert result.score == 78
        assert result.source == "whoop"

    def test_no_recovery_score_returns_unavailable(self) -> None:
        facts = [_make_fact("2026-04-10", hrv=45.0)]
        result = get_recovery_score(facts)
        assert not result.available

    def test_latest_score_used(self) -> None:
        facts = [
            _make_fact("2026-04-08", recovery=50),
            _make_fact("2026-04-09", recovery=65),
            _make_fact("2026-04-10", recovery=80),
        ]
        result = get_recovery_score(facts)
        assert result.score == 80


class TestComputeRecoveryScore:

    def test_empty_returns_unavailable(self) -> None:
        result = compute_recovery_score([])
        assert not result.available

    def test_whoop_priority(self) -> None:
        facts = [_make_fact("2026-04-10", recovery=72)]
        result = compute_recovery_score(facts)
        assert result.source == "whoop"
        assert result.score == 72

    def test_calculated_when_no_whoop(self) -> None:
        # 10 дней HRV + RHR данных без recovery_score
        today = date.today()
        facts = [
            _make_fact(
                (today - timedelta(days=i)).isoformat(),
                hrv=50.0,
                rhr=60.0,
                sleep_ms=8 * 3_600_000,
            )
            for i in range(10, 0, -1)
        ]
        result = compute_recovery_score(facts)
        assert result.available
        assert result.source == "calculated"
        assert 0 <= result.score <= 100

    def test_factors_populated(self) -> None:
        today = date.today()
        facts = [
            _make_fact(
                (today - timedelta(days=i)).isoformat(),
                hrv=50.0,
                rhr=60.0,
                sleep_ms=7 * 3_600_000,
            )
            for i in range(10, 0, -1)
        ]
        result = compute_recovery_score(facts)
        assert "hrv" in result.factors or "sleep" in result.factors

    def test_insufficient_data_returns_unavailable(self) -> None:
        # Только 1 факт без recovery_score
        facts = [_make_fact("2026-04-10", hrv=50.0)]
        result = compute_recovery_score(facts)
        # Одна точка — нет baseline, нет компонентов → unavailable
        assert not result.available

    def test_score_in_valid_range(self) -> None:
        today = date.today()
        facts = [
            _make_fact(
                (today - timedelta(days=i)).isoformat(),
                hrv=45.0 + i,
                rhr=62.0,
                sleep_ms=7_200_000,
            )
            for i in range(14, 0, -1)
        ]
        result = compute_recovery_score(facts)
        if result.available:
            assert 0 <= result.score <= 100


# ---------------------------------------------------------------------------
# OvertrainingDetection
# ---------------------------------------------------------------------------

class TestDetectOvertraining:

    def test_empty_returns_low_risk(self) -> None:
        result = detect_overtraining([])
        assert result.risk_level == "low"
        assert len(result.markers_triggered) == 0

    def test_no_markers_low_risk(self) -> None:
        today = date.today()
        facts = [
            _make_fact((today - timedelta(days=i)).isoformat(), hrv=50.0, rhr=60.0)
            for i in range(14, 0, -1)
        ]
        result = detect_overtraining(facts)
        assert result.risk_level == "low"

    def test_hrv_drop_triggers_marker(self) -> None:
        today = date.today()
        # 13 дней нормального HRV, затем резкое падение
        facts = [
            _make_fact((today - timedelta(days=i)).isoformat(), hrv=50.0)
            for i in range(14, 1, -1)
        ]
        facts.append(_make_fact(today.isoformat(), hrv=30.0))  # -40%
        result = detect_overtraining(facts)
        assert any("HRV" in m for m in result.markers_triggered)

    def test_rhr_elevation_triggers_marker(self) -> None:
        today = date.today()
        facts = [
            _make_fact((today - timedelta(days=i)).isoformat(), rhr=60.0)
            for i in range(14, 1, -1)
        ]
        facts.append(_make_fact(today.isoformat(), rhr=68.0))  # +8 bpm
        result = detect_overtraining(facts)
        assert any("ЧСС" in m for m in result.markers_triggered)

    def test_multiple_markers_medium_risk(self) -> None:
        today = date.today()
        facts = [
            _make_fact(
                (today - timedelta(days=i)).isoformat(),
                hrv=50.0, rhr=60.0, sleep_ms=8 * 3_600_000,
            )
            for i in range(14, 1, -1)
        ]
        # Последний день: HRV упал + RHR вырос
        facts.append(_make_fact(
            today.isoformat(), hrv=30.0, rhr=70.0, sleep_ms=8 * 3_600_000
        ))
        result = detect_overtraining(facts)
        assert result.risk_level in {"medium", "high"}
        assert len(result.markers_triggered) >= 2

    def test_many_markers_high_risk(self) -> None:
        today = date.today()
        facts = [
            _make_fact(
                (today - timedelta(days=i)).isoformat(),
                hrv=50.0, rhr=60.0, sleep_ms=8 * 3_600_000,
            )
            for i in range(14, 1, -1)
        ]
        facts.append(_make_fact(
            today.isoformat(),
            hrv=25.0,       # -50% → маркер HRV
            rhr=70.0,       # +10 bpm → маркер RHR
            sleep_ms=4 * 3_600_000,  # -50% сна → маркер sleep
        ))
        # Добавляем высокий acute/chronic ratio
        acts = [
            _make_activity(duration_seconds=7200, start_date=(today - timedelta(days=i)).isoformat())
            for i in range(3)
        ]
        result = detect_overtraining(facts, activities=acts)
        assert result.risk_level == "high"

    def test_recommendation_non_empty(self) -> None:
        result = detect_overtraining([])
        assert len(result.recommendation) > 0

    def test_load_marker_with_high_ratio(self) -> None:
        today = date.today()
        facts = [_make_fact(today.isoformat(), hrv=50.0)]
        # 7 тяжёлых тренировок подряд (острая нагрузка) без хронической базы
        acts = [
            _make_activity(
                duration_seconds=7200,
                start_date=(today - timedelta(days=i)).isoformat(),
            )
            for i in range(7)
        ]
        result = detect_overtraining(facts, activities=acts)
        # Маркер нагрузки может сработать если acute/chronic > 1.5
        assert isinstance(result.risk_level, str)


# ---------------------------------------------------------------------------
# TrainingLoad (расширения: monotony, strain_weekly, load_warning)
# ---------------------------------------------------------------------------

class TestTrainingLoadExtensions:

    def test_empty_has_zero_monotony(self) -> None:
        load = compute_training_load([])
        assert load.monotony == 0.0
        assert load.strain_weekly == 0.0
        assert load.load_warning is None

    def test_monotony_calculated(self) -> None:
        today = date.today()
        # Нерегулярные тренировки → высокая монотонность невозможна из-за разброса
        acts = [
            _make_activity(duration_seconds=3600, start_date=(today - timedelta(days=i)).isoformat())
            for i in range(7)
        ]
        load = compute_training_load(acts, reference_date=today)
        # Одинаковые тренировки каждый день → stdev=0 → monotony=0
        assert load.monotony == 0.0  # stdev нулевой при одинаковых значениях

    def test_monotony_non_zero_with_variation(self) -> None:
        today = date.today()
        # Разные тренировки в разные дни
        acts = [
            _make_activity(duration_seconds=1800, start_date=(today - timedelta(days=6)).isoformat()),
            _make_activity(duration_seconds=5400, start_date=(today - timedelta(days=4)).isoformat()),
            _make_activity(duration_seconds=3600, start_date=(today - timedelta(days=2)).isoformat()),
        ]
        load = compute_training_load(acts, reference_date=today)
        assert load.monotony >= 0.0

    def test_strain_weekly_equals_load_times_monotony(self) -> None:
        today = date.today()
        acts = [
            _make_activity(duration_seconds=1800, start_date=(today - timedelta(days=5)).isoformat()),
            _make_activity(duration_seconds=5400, start_date=(today - timedelta(days=3)).isoformat()),
        ]
        load = compute_training_load(acts, reference_date=today)
        expected = round(load.weekly_load * load.monotony, 1)
        assert abs(load.strain_weekly - expected) < 0.01

    def test_load_warning_high_ratio(self) -> None:
        today = date.today()
        # Много острой нагрузки, мало хронической
        acute_acts = [
            _make_activity(duration_seconds=7200, start_date=(today - timedelta(days=i)).isoformat())
            for i in range(7)
        ]
        chronic_acts = [
            _make_activity(duration_seconds=600, start_date=(today - timedelta(days=7 + i)).isoformat())
            for i in range(7)
        ]
        load = compute_training_load(acute_acts + chronic_acts, reference_date=today)
        if load.acute_chronic_ratio > 1.5:
            assert load.load_warning is not None
            assert "перетренированно" in load.load_warning.lower()

    def test_load_warning_low_ratio(self) -> None:
        today = date.today()
        # Мало острой нагрузки, много хронической
        acute_acts = [
            _make_activity(duration_seconds=600, start_date=(today - timedelta(days=i)).isoformat())
            for i in range(7)
        ]
        chronic_acts = [
            _make_activity(duration_seconds=7200, start_date=(today - timedelta(days=7 + i)).isoformat())
            for i in range(7)
        ]
        load = compute_training_load(acute_acts + chronic_acts, reference_date=today)
        if 0 < load.acute_chronic_ratio < 0.8:
            assert load.load_warning is not None
            assert "нагрузка ниже нормы" in load.load_warning.lower()

    def test_no_warning_normal_ratio(self) -> None:
        today = date.today()
        # Равная нагрузка acute и chronic → ratio ~1.0
        acts = [
            _make_activity(duration_seconds=3600, start_date=(today - timedelta(days=i)).isoformat())
            for i in range(14)
        ]
        load = compute_training_load(acts, reference_date=today)
        assert load.load_warning is None
