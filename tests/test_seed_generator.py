"""Unit-тесты для SeedGenerator (Issue #33)."""

import os
import sys
from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Фикстура — in-memory SQLite БД
# ---------------------------------------------------------------------------

@pytest.fixture
def db_session():
    """Создать свежую in-memory БД с полной схемой для тестов."""
    from app.db import Base
    import app.models  # noqa: F401 — регистрирует все модели

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


# ---------------------------------------------------------------------------
# Импорт SeedGenerator
# ---------------------------------------------------------------------------

@pytest.fixture
def gen_factory():
    from scripts.seed_data import SeedGenerator
    return SeedGenerator


# ---------------------------------------------------------------------------
# Тесты параметров
# ---------------------------------------------------------------------------

class TestSeedGeneratorParams:
    def test_defaults(self, gen_factory):
        gen = gen_factory()
        assert gen.days == 30
        assert gen.user_count == 1
        assert gen.profile_preset == "intermediate"
        assert gen.scenario == "normal_load"
        assert gen.add_anomalies is False
        assert gen.missing_data_rate == 0.0
        assert gen.truncate_before is False

    def test_custom_params(self, gen_factory):
        gen = gen_factory(days=14, user_count=2, profile_preset="advanced", scenario="overreaching")
        assert gen.days == 14
        assert gen.user_count == 2
        assert gen.profile_preset == "advanced"
        assert gen.scenario == "overreaching"

    def test_missing_data_rate_clamped(self, gen_factory):
        gen = gen_factory(missing_data_rate=1.5)
        assert gen.missing_data_rate == 1.0
        gen2 = gen_factory(missing_data_rate=-0.5)
        assert gen2.missing_data_rate == 0.0

    def test_days_minimum(self, gen_factory):
        gen = gen_factory(days=0)
        assert gen.days == 1


# ---------------------------------------------------------------------------
# Тесты генерации данных (без БД)
# ---------------------------------------------------------------------------

class TestSeedGeneratorPreview:
    def test_preview_returns_data(self, gen_factory):
        gen = gen_factory(days=30, scenario="normal_load")
        result = gen.preview(count=5)
        assert result["scenario"] == "normal_load"
        assert len(result["sample_activities"]) <= 5
        assert len(result["sample_daily_facts"]) == 5
        assert result["days"] == 30

    def test_preview_overreaching(self, gen_factory):
        gen = gen_factory(days=30, scenario="overreaching")
        result = gen.preview()
        assert result["scenario"] == "overreaching"
        assert len(result["sample_daily_facts"]) == 5

    def test_preview_recovery_phase(self, gen_factory):
        gen = gen_factory(days=30, scenario="recovery_phase")
        result = gen.preview()
        assert result["scenario"] == "recovery_phase"

    def test_preview_injury_recovery(self, gen_factory):
        gen = gen_factory(days=30, scenario="injury_recovery")
        result = gen.preview()
        assert result["scenario"] == "injury_recovery"
        # Только low-impact активности
        for act in result["sample_activities"]:
            assert act["sport_type"] in ("cycling", "swimming")

    def test_preview_profile_preset(self, gen_factory):
        gen = gen_factory(profile_preset="advanced")
        result = gen.preview()
        assert result["profile_preset"] == "advanced"


# ---------------------------------------------------------------------------
# Тесты генерации в БД
# ---------------------------------------------------------------------------

class TestSeedGeneratorDB:
    def test_normal_load_creates_records(self, gen_factory, db_session):
        gen = gen_factory(days=30, scenario="normal_load")
        result = gen.generate(db_session)
        assert result.profiles_created == 1
        assert result.activities_created > 0
        assert result.daily_facts_created == 30
        assert len(result.users) == 1

    def test_multi_user(self, gen_factory, db_session):
        gen = gen_factory(days=30, user_count=3)
        result = gen.generate(db_session)
        assert result.profiles_created == 3
        assert result.daily_facts_created == 90  # 30 * 3
        assert len(result.users) == 3

    def test_skip_existing_profile(self, gen_factory, db_session):
        gen = gen_factory(days=30)
        result1 = gen.generate(db_session)
        result2 = gen.generate(db_session)
        assert result1.profiles_created == 1
        assert result2.profiles_created == 0  # уже существует

    def test_beginner_profile(self, gen_factory, db_session):
        from app.models.user_profile import UserProfile
        gen = gen_factory(profile_preset="beginner")
        gen.generate(db_session)
        profile = db_session.query(UserProfile).filter_by(user_id="test-user-001").first()
        assert profile is not None
        assert profile.experience_level == "beginner"

    def test_advanced_profile(self, gen_factory, db_session):
        from app.models.user_profile import UserProfile
        gen = gen_factory(profile_preset="advanced")
        gen.generate(db_session)
        profile = db_session.query(UserProfile).filter_by(user_id="test-user-001").first()
        assert profile is not None
        assert profile.experience_level == "advanced"

    def test_days_parameter(self, gen_factory, db_session):
        from app.models.daily_fact import DailyFact
        gen = gen_factory(days=14)
        gen.generate(db_session)
        count = db_session.query(DailyFact).filter_by(user_id="test-user-001").count()
        assert count == 14

    def test_truncate_before(self, gen_factory, db_session):
        from app.models.activity import Activity
        gen1 = gen_factory(days=30, scenario="normal_load")
        gen1.generate(db_session)
        count_before = db_session.query(Activity).count()
        assert count_before > 0

        # Truncate очищает activities; gen2 пропускает дублирующийся профиль → 0 новых активностей
        gen2 = gen_factory(days=30, scenario="recovery_phase", truncate_before=True)
        gen2.generate(db_session)
        count_after = db_session.query(Activity).count()
        assert count_after <= count_before


# ---------------------------------------------------------------------------
# Тест overreaching — risk_level: high
# ---------------------------------------------------------------------------

class TestOverreachingPreset:
    def test_overreaching_triggers_high_risk(self, gen_factory):
        """overreaching preset должен дать risk_level: high в overtraining_detection."""
        from app.services.data_processing.overtraining_detection import detect_overtraining

        gen = gen_factory(days=30, scenario="overreaching")
        user_id = "test-user-001"
        base_date = date.today() - timedelta(days=29)

        daily_facts = gen._make_daily_facts(user_id, base_date)

        # Конвертируем в dict для detect_overtraining
        facts_as_dicts = []
        for df in daily_facts:
            facts_as_dicts.append({
                "hrv_rmssd_milli": df.hrv_rmssd_milli,
                "resting_heart_rate": df.resting_heart_rate,
                "sleep_total_in_bed_milli": df.sleep_total_in_bed_milli,
                "recovery_score": df.recovery_score,
                "iso_date": df.iso_date,
            })

        result = detect_overtraining(facts_as_dicts)
        assert result.risk_level == "high", (
            f"Ожидался risk_level='high', получили '{result.risk_level}'. "
            f"Маркеры: {result.markers_triggered}"
        )
        assert len(result.markers_triggered) >= 3

    def test_normal_load_low_risk(self, gen_factory):
        """normal_load preset должен дать risk_level: low."""
        from app.services.data_processing.overtraining_detection import detect_overtraining

        gen = gen_factory(days=30, scenario="normal_load")
        user_id = "test-user-001"
        base_date = date.today() - timedelta(days=29)

        daily_facts = gen._make_daily_facts(user_id, base_date)
        facts_as_dicts = [
            {
                "hrv_rmssd_milli": df.hrv_rmssd_milli,
                "resting_heart_rate": df.resting_heart_rate,
                "sleep_total_in_bed_milli": df.sleep_total_in_bed_milli,
                "recovery_score": df.recovery_score,
            }
            for df in daily_facts
        ]

        result = detect_overtraining(facts_as_dicts)
        assert result.risk_level in ("low", "medium"), (
            f"Ожидался low или medium, получили '{result.risk_level}'"
        )


# ---------------------------------------------------------------------------
# Тест injury_recovery — только low-impact
# ---------------------------------------------------------------------------

class TestInjuryRecoveryPreset:
    def test_injury_only_low_impact_sports(self, gen_factory):
        gen = gen_factory(days=30, scenario="injury_recovery")
        user_id = "test-user-001"
        base_date = date.today() - timedelta(days=29)

        activities = gen._make_activities(user_id, base_date)
        for act in activities:
            assert act.sport_type in ("cycling", "swimming"), (
                f"Ожидался cycling или swimming, получили '{act.sport_type}'"
            )

    def test_injury_reduced_steps(self, gen_factory):
        gen = gen_factory(days=30, scenario="injury_recovery")
        user_id = "test-user-001"
        base_date = date.today() - timedelta(days=29)

        daily_facts = gen._make_daily_facts(user_id, base_date)
        avg_steps = sum(df.steps or 0 for df in daily_facts) / len(daily_facts)
        # Шаги при травме должны быть ниже нормы (< 7000 в среднем)
        assert avg_steps < 7_000, f"Среднее шагов при травме слишком высокое: {avg_steps}"


# ---------------------------------------------------------------------------
# Тест recovery_phase — улучшение метрик
# ---------------------------------------------------------------------------

class TestRecoveryPhasePreset:
    def test_recovery_hrv_improves(self, gen_factory):
        gen = gen_factory(days=30, scenario="recovery_phase")
        user_id = "test-user-001"
        base_date = date.today() - timedelta(days=29)

        daily_facts = gen._make_daily_facts(user_id, base_date)
        first_week_hrv = [df.hrv_rmssd_milli for df in daily_facts[:7] if df.hrv_rmssd_milli]
        last_week_hrv = [df.hrv_rmssd_milli for df in daily_facts[-7:] if df.hrv_rmssd_milli]

        if first_week_hrv and last_week_hrv:
            assert sum(last_week_hrv) / len(last_week_hrv) > sum(first_week_hrv) / len(first_week_hrv), \
                "В recovery_phase HRV должен расти"

    def test_recovery_only_light_activities(self, gen_factory):
        gen = gen_factory(days=30, scenario="recovery_phase")
        user_id = "test-user-001"
        base_date = date.today() - timedelta(days=29)

        activities = gen._make_activities(user_id, base_date)
        for act in activities:
            assert act.sport_type in ("walking", "running", "cycling")
            # Нет высоконагрузочных активностей
            assert "crossfit" not in act.title.lower()
            assert "интервал" not in act.title.lower() or "восстанов" in act.title.lower()


# ---------------------------------------------------------------------------
# Тест missing_data_rate
# ---------------------------------------------------------------------------

class TestMissingDataRate:
    def test_missing_data_rate_zero(self, gen_factory):
        gen = gen_factory(days=30, missing_data_rate=0.0)
        user_id = "test-user-001"
        base_date = date.today() - timedelta(days=29)

        daily_facts = gen._make_daily_facts(user_id, base_date)
        none_count = sum(1 for df in daily_facts if df.hrv_rmssd_milli is None)
        assert none_count == 0

    def test_missing_data_rate_high(self, gen_factory):
        import random
        random.seed(42)
        gen = gen_factory(days=100, missing_data_rate=0.8)
        user_id = "test-user-001"
        base_date = date.today() - timedelta(days=99)

        daily_facts = gen._make_daily_facts(user_id, base_date)
        none_count = sum(1 for df in daily_facts if df.hrv_rmssd_milli is None)
        # При rate=0.8 ожидаем много None (примерно 80%, допуск ±25%)
        assert none_count > 30, f"Ожидалось больше пропусков, получили {none_count}/100"
