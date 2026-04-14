"""Скрипт наполнения базы данных тестовыми данными за 30 дней.

Запуск:
    docker compose exec app python scripts/seed_data.py
    # или локально:
    python scripts/seed_data.py
"""

import sys
import os
import uuid
import random
from datetime import datetime, timedelta, date

# Добавляем корень проекта в sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import settings
from app.db import Base
from app.models.user_profile import UserProfile
from app.models.activity import Activity
from app.models.daily_fact import DailyFact


def create_user_profile(user_id: str) -> UserProfile:
    """Создать тестовый профиль пользователя."""
    return UserProfile(
        id=str(uuid.uuid4()),
        user_id=user_id,
        name="Алексей Тестов",
        age=30,
        weight_kg=78.5,
        height_cm=180.0,
        gender="male",
        max_heart_rate=190,
        resting_heart_rate=58,
        training_goals=["похудение", "повышение выносливости", "подготовка к полумарафону"],
        experience_level="intermediate",
        injuries=[
            {
                "body_part": "правое колено",
                "description": "Синдром IT-тяжа",
                "date_occurred": "2025-08-15",
                "status": "recovered",
                "restrictions": ["избегать резких спусков"],
            }
        ],
        chronic_conditions=[],
        preferred_sports=["running", "cycling", "gym"],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


def generate_activities(user_id: str, base_date: date) -> list[Activity]:
    """Генерация ~20 тренировок за 30 дней с реалистичными данными."""
    activities = []

    # Расписание тренировок (номера дней и типы)
    schedule = [
        (0, "running", "Утренняя пробежка", 8_000, 2_800),
        (1, "gym", "Силовая тренировка — верх", None, 3_600),
        (3, "running", "Интервальная тренировка", 6_000, 2_400),
        (4, "cycling", "Велопоездка по городу", 25_000, 5_400),
        (6, "running", "Длинная пробежка", 15_000, 5_400),
        (7, "gym", "Силовая тренировка — ноги", None, 4_200),
        (9, "walking", "Прогулка с собакой", 4_000, 3_600),
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

    for day_offset, sport_type, title, distance, duration in schedule:
        act_date = base_date + timedelta(days=day_offset)
        start_time = datetime(act_date.year, act_date.month, act_date.day, 7, 0, 0)
        # Небольшая случайная вариация времени старта (±30 мин)
        start_time += timedelta(minutes=random.randint(-30, 30))
        end_time = start_time + timedelta(seconds=duration + random.randint(-300, 300))

        # Вычислить скорость для беговых/велосипедных тренировок
        avg_speed = None
        max_speed = None
        if distance and duration > 0:
            avg_speed = round((distance / duration) * 3.6, 2)  # м/с → км/ч
            max_speed = round(avg_speed * random.uniform(1.15, 1.35), 2)

        calories = int(duration / 60 * random.uniform(7, 12))
        avg_hr = random.randint(130, 165)
        max_hr = random.randint(170, 188)

        activities.append(
            Activity(
                id=str(uuid.uuid4()),
                user_id=user_id,
                title=title,
                sport_type=sport_type,
                distance_meters=float(distance) if distance else None,
                duration_seconds=duration + random.randint(-120, 120),
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
                anomaly_flags=[],
                raw_title=title,
            )
        )

    return activities


def generate_daily_facts(user_id: str, base_date: date) -> list[DailyFact]:
    """Генерация 30 записей дневных метрик с реалистичными вариациями."""
    daily_facts = []

    for day in range(30):
        fact_date = base_date + timedelta(days=day)

        # Вариативные метрики
        steps = random.randint(6_000, 14_000)
        calories = random.randint(1_800, 2_800)
        recovery_score = random.randint(40, 95)
        hrv = round(random.uniform(35.0, 90.0), 1)
        rhr = random.randint(52, 68)
        spo2 = round(random.uniform(96.5, 99.5), 1)
        skin_temp = round(random.uniform(33.5, 35.5), 2)
        # Сон: 6–9 часов в миллисекундах
        sleep_hours = random.uniform(6.0, 9.0)
        sleep_ms = int(sleep_hours * 3_600_000)
        water = round(random.uniform(1.5, 3.0), 1)

        daily_facts.append(
            DailyFact(
                id=str(uuid.uuid4()),
                user_id=user_id,
                iso_date=fact_date.isoformat(),
                steps=steps,
                calories_kcal=calories,
                recovery_score=recovery_score,
                hrv_rmssd_milli=hrv,
                resting_heart_rate=rhr,
                spo2_percentage=spo2,
                skin_temp_celsius=skin_temp,
                sleep_total_in_bed_milli=sleep_ms,
                water_liters=water,
                sources_json={
                    "steps": "garmin",
                    "calories": "garmin",
                    "recovery_score": "whoop",
                    "hrv": "whoop",
                    "sleep": "whoop",
                    "heart_rate": "garmin",
                },
            )
        )

    return daily_facts


def seed(db_url: str | None = None) -> None:
    """Основная функция наполнения БД тестовыми данными."""
    url = db_url or settings.database_url_sync
    engine = create_engine(url, connect_args={"check_same_thread": False})

    # Создать директорию для БД если нет
    db_path = settings.db_path
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    print(f"Подключение к БД: {url}")

    with Session(engine) as session:
        # Проверить, не заполнена ли уже БД
        existing = session.query(UserProfile).count()
        if existing > 0:
            print(f"БД уже содержит {existing} профилей. Пропускаем seed.")
            return

        user_id = "test-user-001"
        base_date = date(2026, 3, 15)  # 30 дней назад от ~14 апреля 2026

        print("Создание профиля пользователя...")
        profile = create_user_profile(user_id)
        session.add(profile)

        print("Генерация тренировок...")
        activities = generate_activities(user_id, base_date)
        for act in activities:
            session.add(act)
        print(f"  Добавлено {len(activities)} тренировок")

        print("Генерация дневных метрик...")
        daily_facts = generate_daily_facts(user_id, base_date)
        for df in daily_facts:
            session.add(df)
        print(f"  Добавлено {len(daily_facts)} дневных записей")

        session.commit()
        print("Seed данные успешно загружены!")

        # Статистика
        print("\n--- Статистика ---")
        print(f"Профилей пользователей: {session.query(UserProfile).count()}")
        print(f"Тренировок: {session.query(Activity).count()}")
        print(f"Дневных записей: {session.query(DailyFact).count()}")

        from sqlalchemy import func
        sport_stats = session.query(Activity.sport_type, func.count()).group_by(Activity.sport_type).all()
        print("\nТренировки по типам:")
        for sport, count in sport_stats:
            print(f"  {sport}: {count}")


if __name__ == "__main__":
    seed()
