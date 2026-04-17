"""Генератор тестовых данных v2 — параметризуемый SeedGenerator с preset-сценариями.

Запуск:
    docker compose exec app python scripts/seed_data.py
    docker compose exec app python scripts/seed_data.py --days 30 --preset overreaching --users 2
    docker compose exec app python scripts/seed_data.py --preset recovery_phase --truncate
"""

import argparse
import os
import random
import sys
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Literal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.config import settings
from app.models.activity import Activity
from app.models.daily_fact import DailyFact
from app.models.user_profile import UserProfile


# ---------------------------------------------------------------------------
# Типы и константы
# ---------------------------------------------------------------------------

ProfilePreset = Literal["beginner", "intermediate", "advanced"]
Scenario = Literal["normal_load", "overreaching", "recovery_phase", "injury_recovery"]

_PROFILE_CONFIGS: dict[str, dict[str, Any]] = {
    "beginner": {
        "name": "Алексей Новиков",
        "age": 28,
        "weight_kg": 82.0,
        "height_cm": 178.0,
        "gender": "male",
        "max_heart_rate": 182,
        "resting_heart_rate": 65,
        "experience_level": "beginner",
        "training_goals": ["снижение веса", "улучшение общей формы"],
        "preferred_sports": ["running", "walking", "gym"],
        "injuries": [],
    },
    "intermediate": {
        "name": "Алексей Тестов",
        "age": 30,
        "weight_kg": 78.5,
        "height_cm": 180.0,
        "gender": "male",
        "max_heart_rate": 190,
        "resting_heart_rate": 58,
        "experience_level": "intermediate",
        "training_goals": ["похудение", "повышение выносливости", "подготовка к полумарафону"],
        "preferred_sports": ["running", "cycling", "gym"],
        "injuries": [
            {
                "body_part": "правое колено",
                "description": "Синдром IT-тяжа",
                "date_occurred": "2025-08-15",
                "status": "recovered",
                "restrictions": ["избегать резких спусков"],
            }
        ],
    },
    "advanced": {
        "name": "Марина Быстрова",
        "age": 27,
        "weight_kg": 62.0,
        "height_cm": 168.0,
        "gender": "female",
        "max_heart_rate": 195,
        "resting_heart_rate": 48,
        "experience_level": "advanced",
        "training_goals": ["подготовка к марафону", "улучшение результата на 10км"],
        "preferred_sports": ["running", "cycling", "triathlon"],
        "injuries": [],
    },
}

# ---------------------------------------------------------------------------
# SeedGenerator
# ---------------------------------------------------------------------------


@dataclass
class SeedResult:
    """Итог генерации."""
    profiles_created: int = 0
    activities_created: int = 0
    daily_facts_created: int = 0
    users: list[str] = field(default_factory=list)
    scenario: str = "normal_load"
    days: int = 30


class SeedGenerator:
    """Параметризуемый генератор тестовых данных.

    Параметры:
        days: Количество дней для генерации (default: 30).
        user_count: Количество пользователей (default: 1).
        profile_preset: Уровень опытности профиля (beginner/intermediate/advanced).
        scenario: Сценарий данных (normal_load/overreaching/recovery_phase/injury_recovery).
        add_anomalies: Добавлять случайные флаги аномалий в активности и метрики.
        missing_data_rate: Доля пропусков в ежедневных метриках (0–1).
        truncate_before: Очистить таблицы (кроме профилей) перед генерацией.
    """

    def __init__(
        self,
        days: int = 30,
        user_count: int = 1,
        profile_preset: ProfilePreset = "intermediate",
        scenario: Scenario = "normal_load",
        add_anomalies: bool = False,
        missing_data_rate: float = 0.0,
        truncate_before: bool = False,
    ) -> None:
        self.days = max(1, days)
        self.user_count = max(1, user_count)
        self.profile_preset = profile_preset
        self.scenario = scenario
        self.add_anomalies = add_anomalies
        self.missing_data_rate = max(0.0, min(1.0, missing_data_rate))
        self.truncate_before = truncate_before

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def generate(self, session: Session) -> SeedResult:
        """Сгенерировать и сохранить данные в БД."""
        if self.truncate_before:
            self._truncate(session)

        result = SeedResult(scenario=self.scenario, days=self.days)

        for i in range(self.user_count):
            suffix = f"-{i + 1:03d}" if self.user_count > 1 else "-001"
            user_id = f"test-user{suffix}"

            existing = session.query(UserProfile).filter_by(user_id=user_id).first()
            if existing:
                print(f"  Профиль {user_id} уже существует — пропускаем.")
                continue

            profile = self._make_profile(user_id)
            session.add(profile)

            base_date = date.today() - timedelta(days=self.days - 1)

            activities = self._make_activities(user_id, base_date)
            for act in activities:
                session.add(act)

            daily_facts = self._make_daily_facts(user_id, base_date)
            for df in daily_facts:
                session.add(df)

            result.profiles_created += 1
            result.activities_created += len(activities)
            result.daily_facts_created += len(daily_facts)
            result.users.append(user_id)

        session.commit()
        return result

    def preview(self, count: int = 5) -> dict[str, Any]:
        """Вернуть несколько сгенерированных записей без записи в БД."""
        user_id = "preview-user-001"
        base_date = date.today() - timedelta(days=self.days - 1)

        activities = self._make_activities(user_id, base_date)
        daily_facts = self._make_daily_facts(user_id, base_date)

        def act_dict(a: Activity) -> dict:
            return {
                "title": a.title,
                "sport_type": a.sport_type,
                "duration_seconds": a.duration_seconds,
                "distance_meters": a.distance_meters,
                "calories": a.calories,
                "avg_heart_rate": a.avg_heart_rate,
                "start_time": a.start_time.isoformat() if a.start_time else None,
                "anomaly_flags": a.anomaly_flags,
            }

        def fact_dict(f: DailyFact) -> dict:
            return {
                "iso_date": f.iso_date,
                "steps": f.steps,
                "recovery_score": f.recovery_score,
                "hrv_rmssd_milli": f.hrv_rmssd_milli,
                "resting_heart_rate": f.resting_heart_rate,
                "sleep_hours": round(f.sleep_total_in_bed_milli / 3_600_000, 2) if f.sleep_total_in_bed_milli else None,
                "anomaly_flags": f.anomaly_flags,
            }

        return {
            "scenario": self.scenario,
            "profile_preset": self.profile_preset,
            "days": self.days,
            "sample_activities": [act_dict(a) for a in activities[:count]],
            "sample_daily_facts": [fact_dict(f) for f in daily_facts[:count]],
        }

    # ------------------------------------------------------------------
    # Внутренние методы — профиль
    # ------------------------------------------------------------------

    def _make_profile(self, user_id: str) -> UserProfile:
        cfg = _PROFILE_CONFIGS[self.profile_preset]
        return UserProfile(
            id=str(uuid.uuid4()),
            user_id=user_id,
            name=cfg["name"],
            age=cfg["age"],
            weight_kg=cfg["weight_kg"],
            height_cm=cfg["height_cm"],
            gender=cfg["gender"],
            max_heart_rate=cfg["max_heart_rate"],
            resting_heart_rate=cfg["resting_heart_rate"],
            training_goals=cfg["training_goals"],
            experience_level=cfg["experience_level"],
            injuries=cfg["injuries"],
            chronic_conditions=[],
            preferred_sports=cfg["preferred_sports"],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    # ------------------------------------------------------------------
    # Внутренние методы — активности
    # ------------------------------------------------------------------

    def _make_activities(self, user_id: str, base_date: date) -> list[Activity]:
        if self.scenario == "injury_recovery":
            return self._activities_injury(user_id, base_date)
        elif self.scenario == "recovery_phase":
            return self._activities_recovery(user_id, base_date)
        elif self.scenario == "overreaching":
            return self._activities_overreaching(user_id, base_date)
        else:
            return self._activities_normal(user_id, base_date)

    def _build_activity(
        self,
        user_id: str,
        act_date: date,
        sport_type: str,
        title: str,
        distance_m: int | None,
        duration_s: int,
        anomaly_flags: list[str] | None = None,
    ) -> Activity:
        start_time = datetime(act_date.year, act_date.month, act_date.day, 7, 0)
        start_time += timedelta(minutes=random.randint(-30, 30))
        end_time = start_time + timedelta(seconds=duration_s + random.randint(-120, 120))

        avg_speed = None
        max_speed = None
        if distance_m and duration_s > 0:
            avg_speed = round((distance_m / duration_s) * 3.6, 2)
            max_speed = round(avg_speed * random.uniform(1.15, 1.35), 2)

        calories = int(duration_s / 60 * random.uniform(7, 12))
        avg_hr = random.randint(125, 165)
        max_hr = random.randint(168, 188)

        flags = anomaly_flags or []
        if self.add_anomalies and random.random() < 0.15:
            flags = flags + ["calorie_outlier"]

        return Activity(
            id=str(uuid.uuid4()),
            user_id=user_id,
            title=title,
            sport_type=sport_type,
            distance_meters=float(distance_m) if distance_m else None,
            duration_seconds=duration_s + random.randint(-120, 120),
            start_time=start_time,
            end_time=end_time,
            avg_speed=avg_speed,
            max_speed=max_speed,
            elevation_meters=round(random.uniform(10, 80), 1) if sport_type in ("running", "cycling") else None,
            calories=calories,
            avg_heart_rate=avg_hr,
            max_heart_rate=max_hr,
            source="garmin",
            is_primary=True,
            anomaly_flags=flags,
            raw_title=title,
        )

    def _activities_normal(self, user_id: str, base_date: date) -> list[Activity]:
        """Стандартная прогрессия нагрузки."""
        schedule = [
            (0, "running", "Утренняя пробежка", 8_000, 2_800),
            (1, "gym", "Силовая тренировка — верх", None, 3_600),
            (3, "running", "Интервальная тренировка", 6_000, 2_400),
            (4, "cycling", "Велопоездка по городу", 25_000, 5_400),
            (6, "running", "Длинная пробежка", 15_000, 5_400),
            (7, "gym", "Силовая тренировка — ноги", None, 4_200),
            (9, "walking", "Прогулка", 4_000, 3_600),
            (10, "running", "Темповый бег", 7_500, 2_700),
            (11, "gym", "Функциональная тренировка", None, 3_300),
            (13, "cycling", "Велопоездка за город", 40_000, 7_200),
            (14, "running", "Восстановительная пробежка", 5_000, 2_100),
            (16, "gym", "Силовая тренировка — верх", None, 3_600),
            (17, "running", "Интервальная тренировка", 8_000, 3_000),
            (19, "walking", "Активный отдых", 6_000, 5_400),
            (20, "running", "Длинная пробежка", 18_000, 6_300),
            (21, "gym", "Силовая тренировка — ноги", None, 4_200),
            (23, "cycling", "Велопоездка", 30_000, 6_000),
            (24, "running", "Лёгкий бег", 6_000, 2_400),
            (26, "gym", "Кроссфит", None, 3_600),
            (28, "running", "Воскресная длинная пробежка", 20_000, 6_900),
        ]
        activities = []
        for day_offset, sport, title, dist, dur in schedule:
            if day_offset >= self.days:
                continue
            activities.append(
                self._build_activity(user_id, base_date + timedelta(days=day_offset), sport, title, dist, dur)
            )
        return activities

    def _activities_overreaching(self, user_id: str, base_date: date) -> list[Activity]:
        """Высокая нагрузка + заключительные 7 дней с усиленными тренировками."""
        schedule = [
            (0, "running", "Утренняя пробежка", 8_000, 2_800),
            (1, "gym", "Силовая — верх", None, 3_600),
            (2, "running", "Интервалы", 7_000, 2_700),
            (3, "cycling", "Велопоездка", 30_000, 5_400),
            (5, "running", "Длинная пробежка", 16_000, 5_700),
            (6, "gym", "Силовая — ноги", None, 4_200),
            (7, "running", "Темповый бег", 8_000, 2_850),
            (8, "cycling", "Интенсивные интервалы", 20_000, 3_600),
            (9, "gym", "Тяжёлая силовая", None, 4_500),
            (10, "running", "Средняя пробежка", 12_000, 4_200),
            (11, "cycling", "Длинная велопоездка", 50_000, 9_000),
            (12, "running", "Легкий бег", 5_000, 2_100),
            (13, "gym", "Кроссфит", None, 3_600),
            # Последние 7 дней: повышенная интенсивность
            (14, "running", "Быстрый темп (перегрузка)", 10_000, 3_000),
            (15, "gym", "Сверхнагрузка — верх", None, 5_400),
            (16, "running", "Двойная тренировка — утро", 12_000, 4_200),
            (17, "cycling", "Горные интервалы", 35_000, 5_400),
            (18, "gym", "Сверхнагрузка — ноги", None, 5_400),
            (19, "running", "Длинный забег в темпе", 22_000, 7_200),
            (20, "gym", "Кроссфит финал", None, 4_200),
        ]
        activities = []
        for day_offset, sport, title, dist, dur in schedule:
            if day_offset >= self.days:
                continue
            activities.append(
                self._build_activity(user_id, base_date + timedelta(days=day_offset), sport, title, dist, dur)
            )
        return activities

    def _activities_recovery(self, user_id: str, base_date: date) -> list[Activity]:
        """Период восстановления — лёгкие тренировки."""
        schedule = [
            (1, "walking", "Прогулка", 4_000, 3_600),
            (3, "running", "Лёгкий восстановительный бег", 5_000, 2_400),
            (5, "walking", "Долгая прогулка", 7_000, 5_400),
            (7, "running", "Лёгкий бег", 6_000, 2_700),
            (9, "walking", "Прогулка", 4_000, 3_600),
            (11, "cycling", "Спокойная велопоездка", 15_000, 3_600),
            (13, "running", "Восстановительный бег", 5_500, 2_500),
            (15, "walking", "Активный отдых", 6_000, 4_800),
            (17, "running", "Лёгкий бег", 6_000, 2_700),
            (19, "cycling", "Спокойная велопоездка", 18_000, 4_200),
            (21, "walking", "Прогулка", 5_000, 4_200),
            (23, "running", "Лёгкий темп", 7_000, 3_000),
            (25, "cycling", "Восстановительная езда", 20_000, 4_500),
            (27, "walking", "Долгая прогулка", 8_000, 6_000),
            (29, "running", "Лёгкий бег", 6_500, 2_850),
        ]
        activities = []
        for day_offset, sport, title, dist, dur in schedule:
            if day_offset >= self.days:
                continue
            activities.append(
                self._build_activity(user_id, base_date + timedelta(days=day_offset), sport, title, dist, dur)
            )
        return activities

    def _activities_injury(self, user_id: str, base_date: date) -> list[Activity]:
        """Реабилитация — только low-impact (велосипед, плавание)."""
        schedule = [
            (1, "cycling", "Спокойная велопоездка", 10_000, 2_400),
            (3, "swimming", "Плавание — восстановление", None, 1_800),
            (5, "cycling", "Велопоездка — низкая интенсивность", 12_000, 2_700),
            (7, "swimming", "Плавание", None, 2_100),
            (9, "cycling", "Лёгкая велопоездка", 14_000, 3_000),
            (11, "swimming", "Техническое плавание", None, 2_400),
            (13, "cycling", "Велопоездка", 16_000, 3_300),
            (15, "swimming", "Плавание", None, 2_400),
            (17, "cycling", "Велопрогулка", 15_000, 3_200),
            (19, "swimming", "Плавание восстановление", None, 2_100),
            (21, "cycling", "Велосипед — умеренный темп", 18_000, 3_600),
            (23, "swimming", "Плавание", None, 2_700),
            (25, "cycling", "Длинная велопоездка", 22_000, 4_200),
            (27, "swimming", "Плавание — интенсивное", None, 2_700),
            (29, "cycling", "Велопоездка — прогрессия", 20_000, 3_900),
        ]
        activities = []
        for day_offset, sport, title, dist, dur in schedule:
            if day_offset >= self.days:
                continue
            activities.append(
                self._build_activity(user_id, base_date + timedelta(days=day_offset), sport, title, dist, dur)
            )
        return activities

    # ------------------------------------------------------------------
    # Внутренние методы — дневные метрики
    # ------------------------------------------------------------------

    def _make_daily_facts(self, user_id: str, base_date: date) -> list[DailyFact]:
        if self.scenario == "overreaching":
            return self._daily_facts_overreaching(user_id, base_date)
        elif self.scenario == "recovery_phase":
            return self._daily_facts_recovery(user_id, base_date)
        elif self.scenario == "injury_recovery":
            return self._daily_facts_injury(user_id, base_date)
        else:
            return self._daily_facts_normal(user_id, base_date)

    def _maybe_none(self, value: Any) -> Any:
        """Вернуть None с вероятностью missing_data_rate."""
        if self.missing_data_rate > 0 and random.random() < self.missing_data_rate:
            return None
        return value

    def _build_daily_fact(
        self,
        user_id: str,
        fact_date: date,
        hrv: float,
        rhr: int,
        recovery_score: int,
        sleep_hours: float,
        steps: int | None = None,
        anomaly_flags: list[str] | None = None,
    ) -> DailyFact:
        sleep_ms = int(sleep_hours * 3_600_000)
        flags = anomaly_flags or []
        if self.add_anomalies and random.random() < 0.1:
            flags = flags + [random.choice(["hrv_drop", "rhr_spike", "low_sleep"])]

        return DailyFact(
            id=str(uuid.uuid4()),
            user_id=user_id,
            iso_date=fact_date.isoformat(),
            steps=self._maybe_none(steps or random.randint(6_000, 14_000)),
            calories_kcal=self._maybe_none(random.randint(1_800, 2_800)),
            recovery_score=self._maybe_none(recovery_score + random.randint(-5, 5)),
            hrv_rmssd_milli=self._maybe_none(round(hrv + random.uniform(-3.0, 3.0), 1)),
            resting_heart_rate=self._maybe_none(rhr + random.randint(-2, 2)),
            spo2_percentage=self._maybe_none(round(random.uniform(96.5, 99.5), 1)),
            skin_temp_celsius=self._maybe_none(round(random.uniform(33.5, 35.5), 2)),
            sleep_total_in_bed_milli=self._maybe_none(
                sleep_ms + random.randint(-1_800_000, 1_800_000)
            ),
            water_liters=self._maybe_none(round(random.uniform(1.5, 3.0), 1)),
            sources_json={
                "steps": "garmin",
                "calories": "garmin",
                "recovery_score": "whoop",
                "hrv": "whoop",
                "sleep": "whoop",
                "heart_rate": "garmin",
            },
            anomaly_flags=flags,
        )

    def _daily_facts_normal(self, user_id: str, base_date: date) -> list[DailyFact]:
        """Нормальная нагрузка — стабильные метрики."""
        facts = []
        for day in range(self.days):
            fact_date = base_date + timedelta(days=day)
            facts.append(self._build_daily_fact(
                user_id, fact_date,
                hrv=round(random.uniform(55.0, 75.0), 1),
                rhr=random.randint(54, 64),
                recovery_score=random.randint(60, 90),
                sleep_hours=random.uniform(7.0, 8.5),
            ))
        return facts

    def _daily_facts_overreaching(self, user_id: str, base_date: date) -> list[DailyFact]:
        """Перетренированность: нормально до дня (days-7), затем резкое ухудшение.

        Гарантирует risk_level: high в overtraining_detection (≥3 маркера):
        - HRV drop ≥ 40% от baseline → обнаружение: drop > 10%
        - RHR +20 bpm в последний день → обнаружение: elevation > 5 bpm
        - Sleep drop ≥ 50% → обнаружение: drop > 15%

        Параметры рассчитаны с учётом случайного шума _build_daily_fact (±3 HRV,
        ±2 RHR, ±0.5h sleep), чтобы пороги срабатывали даже при максимальном шуме.
        """
        facts = []
        # Высокие базовые значения (хорошая форма перед перегрузкой)
        baseline_hrv = 72.0
        baseline_rhr = 57
        baseline_sleep_h = 7.8

        overreach_start = max(0, self.days - 7)

        for day in range(self.days):
            fact_date = base_date + timedelta(days=day)

            if day < overreach_start:
                # Нормальная зона — реалистичные вариации
                hrv = baseline_hrv + random.uniform(-4.0, 4.0)
                rhr = baseline_rhr + random.randint(-2, 2)
                recovery = random.randint(65, 90)
                sleep_h = baseline_sleep_h + random.uniform(-0.5, 0.5)
                flags: list[str] = []
            else:
                # Зона перетренированности: сильное ухудшение без лишнего шума
                # (шум добавляется в _build_daily_fact, здесь только детерминированный тренд)
                drop_factor = (day - overreach_start + 1) / 7.0  # 1/7 .. 7/7
                hrv = baseline_hrv * (1.0 - 0.40 * drop_factor)   # до 40% падение
                rhr = baseline_rhr + int(20 * drop_factor)          # до +20 bpm
                recovery = max(15, int(80 * (1.0 - 0.60 * drop_factor)))
                sleep_h = baseline_sleep_h * (1.0 - 0.50 * drop_factor)  # до 50% падение
                flags = ["hrv_drop", "rhr_spike"]
                if sleep_h < baseline_sleep_h * 0.85:
                    flags.append("low_sleep")

            facts.append(self._build_daily_fact(
                user_id, fact_date,
                hrv=round(max(20.0, hrv), 1),
                rhr=max(45, min(110, rhr)),
                recovery_score=max(10, min(100, recovery)),
                sleep_hours=max(3.5, sleep_h),
                anomaly_flags=flags,
            ))
        return facts

    def _daily_facts_recovery(self, user_id: str, base_date: date) -> list[DailyFact]:
        """Восстановление — высокий сон, стабильный HRV, низкая нагрузка."""
        facts = []
        for day in range(self.days):
            fact_date = base_date + timedelta(days=day)
            # Улучшение к концу периода
            progress = day / max(1, self.days - 1)
            facts.append(self._build_daily_fact(
                user_id, fact_date,
                hrv=round(55.0 + 20.0 * progress + random.uniform(-3.0, 3.0), 1),
                rhr=int(65 - 8 * progress) + random.randint(-1, 2),
                recovery_score=int(50 + 40 * progress) + random.randint(-5, 5),
                sleep_hours=8.0 + 0.5 * progress + random.uniform(-0.3, 0.3),
                steps=random.randint(4_000, 8_000),
            ))
        return facts

    def _daily_facts_injury(self, user_id: str, base_date: date) -> list[DailyFact]:
        """Травма: низкая активность, нестабильный HRV."""
        facts = []
        base_hrv = 60.0
        for day in range(self.days):
            fact_date = base_date + timedelta(days=day)
            # Нестабильный HRV (скачет ±20%)
            hrv = base_hrv + random.uniform(-12.0, 12.0)
            facts.append(self._build_daily_fact(
                user_id, fact_date,
                hrv=round(max(30.0, hrv), 1),
                rhr=random.randint(56, 70),
                recovery_score=random.randint(35, 75),
                sleep_hours=random.uniform(6.5, 8.5),
                steps=random.randint(2_000, 6_000),
            ))
        return facts

    # ------------------------------------------------------------------
    # Truncate
    # ------------------------------------------------------------------

    def _truncate(self, session: Session) -> None:
        """Очистить таблицы данных (кроме user_profiles)."""
        for table in ("activities", "daily_facts", "pipeline_logs", "chat_messages", "chat_sessions"):
            try:
                session.execute(text(f"DELETE FROM {table}"))
            except Exception:
                pass
        session.commit()
        print("  Таблицы очищены.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed Generator v2")
    parser.add_argument("--days", type=int, default=30, help="Количество дней (default: 30)")
    parser.add_argument("--users", type=int, default=1, help="Количество пользователей (default: 1)")
    parser.add_argument(
        "--preset",
        choices=["normal_load", "overreaching", "recovery_phase", "injury_recovery"],
        default="normal_load",
        help="Сценарий данных (default: normal_load)",
    )
    parser.add_argument(
        "--profile",
        choices=["beginner", "intermediate", "advanced"],
        default="intermediate",
        help="Профиль пользователя (default: intermediate)",
    )
    parser.add_argument("--anomalies", action="store_true", help="Добавить случайные аномалии")
    parser.add_argument(
        "--missing-rate",
        type=float,
        default=0.0,
        help="Доля пропущенных метрик 0–1 (default: 0.0)",
    )
    parser.add_argument("--truncate", action="store_true", help="Очистить таблицы перед генерацией")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    url = settings.database_url_sync
    engine = create_engine(url, connect_args={"check_same_thread": False})

    db_path = settings.db_path
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    print(f"Подключение к БД: {url}")

    gen = SeedGenerator(
        days=args.days,
        user_count=args.users,
        profile_preset=args.profile,
        scenario=args.preset,
        add_anomalies=args.anomalies,
        missing_data_rate=args.missing_rate,
        truncate_before=args.truncate,
    )

    with Session(engine) as session:
        print(f"Запуск SeedGenerator: scenario={gen.scenario}, days={gen.days}, users={gen.user_count}")
        result = gen.generate(session)

    print("\nSeed завершён!")
    print(f"  Профилей:        {result.profiles_created}")
    print(f"  Тренировок:      {result.activities_created}")
    print(f"  Дневных записей: {result.daily_facts_created}")
    print(f"  Пользователи:    {result.users}")


if __name__ == "__main__":
    main()
