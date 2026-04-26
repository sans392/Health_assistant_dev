"""DB tools — функции для запроса и записи данных в БД.

Каждая функция возвращает ToolResult с success/error и данными.
Все вызовы логируются.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.activity import Activity
from app.models.daily_fact import DailyFact
from app.models.user_profile import UserProfile


def _apply_range_filters(stmt, model, ranges: dict[str, tuple[Any, Any]]):
    """Прокинуть min/max-фильтры на column'ы модели в SQLAlchemy stmt.

    ranges: column_name -> (min_value | None, max_value | None).
    None-границы пропускаются, чтобы не плодить тривиальные WHERE 1=1.
    """
    for column_name, (lo, hi) in ranges.items():
        if lo is None and hi is None:
            continue
        column = getattr(model, column_name)
        if lo is not None:
            stmt = stmt.where(column >= lo)
        if hi is not None:
            stmt = stmt.where(column <= hi)
    return stmt

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Результат вызова одного tool."""

    tool_name: str
    success: bool
    data: Any  # list[dict] | dict | None
    error: str | None


# ---------------------------------------------------------------------------
# Вспомогательные сериализаторы
# ---------------------------------------------------------------------------


def _activity_to_dict(activity: Activity) -> dict:
    """Сериализует Activity в словарь для передачи дальше по пайплайну."""
    return {
        "id": activity.id,
        "user_id": activity.user_id,
        "title": activity.title,
        "sport_type": activity.sport_type,
        "distance_meters": activity.distance_meters,
        "duration_seconds": activity.duration_seconds,
        "start_time": activity.start_time.isoformat(),
        "end_time": activity.end_time.isoformat(),
        "avg_speed": activity.avg_speed,
        "calories": activity.calories,
        "avg_heart_rate": activity.avg_heart_rate,
        "source": activity.source,
    }


def _daily_fact_to_dict(fact: DailyFact) -> dict:
    """Сериализует DailyFact в словарь."""
    return {
        "id": fact.id,
        "user_id": fact.user_id,
        "iso_date": fact.iso_date,
        "steps": fact.steps,
        "calories_kcal": fact.calories_kcal,
        "recovery_score": fact.recovery_score,
        "hrv_rmssd_milli": fact.hrv_rmssd_milli,
        "resting_heart_rate": fact.resting_heart_rate,
        "spo2_percentage": fact.spo2_percentage,
        "skin_temp_celsius": fact.skin_temp_celsius,
        "sleep_total_in_bed_milli": fact.sleep_total_in_bed_milli,
        "water_liters": fact.water_liters,
    }


# Маппинг логических метрик (значения MetricEnum + английские алиасы) на
# реальные столбцы DailyFact. Нужен для фильтра в get_daily_facts: intent
# detection/planner передают метрики в терминах MetricEnum («шаги», «калории»,
# «сон», «recovery», ...), а DailyFact хранит их под техническими именами.
# Метрики, которых нет в DailyFact (вес, рост, strain, rpe, дистанция, темп,
# время, cadence), в словарь не попадают и при фильтре игнорируются.
_METRIC_TO_DAILY_FACT_FIELDS: dict[str, tuple[str, ...]] = {
    "шаги": ("steps",),
    "steps": ("steps",),
    "калории": ("calories_kcal",),
    "calories": ("calories_kcal",),
    "calories_kcal": ("calories_kcal",),
    "heart_rate": ("resting_heart_rate",),
    "resting_heart_rate": ("resting_heart_rate",),
    "hrv": ("hrv_rmssd_milli",),
    "hrv_rmssd_milli": ("hrv_rmssd_milli",),
    "сон": ("sleep_total_in_bed_milli",),
    "sleep": ("sleep_total_in_bed_milli",),
    "sleep_total_in_bed_milli": ("sleep_total_in_bed_milli",),
    "recovery": ("recovery_score",),
    "recovery_score": ("recovery_score",),
    "spo2": ("spo2_percentage",),
    "spo2_percentage": ("spo2_percentage",),
    "skin_temp": ("skin_temp_celsius",),
    "skin_temp_celsius": ("skin_temp_celsius",),
    "water": ("water_liters",),
    "water_liters": ("water_liters",),
}


def _profile_to_dict(profile: UserProfile) -> dict:
    """Сериализует UserProfile в словарь."""
    return {
        "user_id": profile.user_id,
        "name": profile.name,
        "age": profile.age,
        "weight_kg": profile.weight_kg,
        "height_cm": profile.height_cm,
        "gender": profile.gender,
        "max_heart_rate": profile.max_heart_rate,
        "resting_heart_rate": profile.resting_heart_rate,
        "training_goals": profile.training_goals,
        "experience_level": profile.experience_level,
        "injuries": profile.injuries,
        "chronic_conditions": profile.chronic_conditions,
        "preferred_sports": profile.preferred_sports,
    }


# ---------------------------------------------------------------------------
# Tool-функции (чтение)
# ---------------------------------------------------------------------------


async def get_activities(
    db: AsyncSession,
    user_id: str,
    date_from: date,
    date_to: date,
    sport_type: str | None = None,
    sport_types: list[str] | None = None,
    min_distance_meters: float | None = None,
    max_distance_meters: float | None = None,
    min_duration_seconds: int | None = None,
    max_duration_seconds: int | None = None,
    min_calories: int | None = None,
    max_calories: int | None = None,
    min_avg_heart_rate: int | None = None,
    max_avg_heart_rate: int | None = None,
    min_avg_speed: float | None = None,
    max_avg_speed: float | None = None,
    min_elevation_meters: float | None = None,
    max_elevation_meters: float | None = None,
    title_contains: str | None = None,
) -> ToolResult:
    """Получить тренировки пользователя за период с whitelisted-фильтрами.

    Возвращает только первичные записи (is_primary=True). Все min/max-фильтры
    опциональны и применяются только если значение не None — None-границы не
    добавляют WHERE.

    Args:
        db: Асинхронная сессия SQLAlchemy.
        user_id: Идентификатор пользователя.
        date_from: Начало периода (включительно).
        date_to: Конец периода (включительно).
        sport_type: Фильтр по виду спорта (одно значение).
        sport_types: Несколько видов спорта одновременно (логическое OR).
        min_distance_meters / max_distance_meters: диапазон дистанции.
        min_duration_seconds / max_duration_seconds: диапазон длительности.
        min_calories / max_calories: диапазон калорий.
        min_avg_heart_rate / max_avg_heart_rate: диапазон средней ЧСС.
        min_avg_speed / max_avg_speed: диапазон средней скорости (м/с).
        min_elevation_meters / max_elevation_meters: диапазон набора высоты.
        title_contains: подстрока в title (case-insensitive).

    Returns:
        ToolResult с list[dict] активностей.
    """
    try:
        start_dt = datetime.combine(date_from, datetime.min.time())
        end_dt = datetime.combine(date_to, datetime.max.time())

        stmt = (
            select(Activity)
            .where(
                Activity.user_id == user_id,
                Activity.is_primary == True,  # noqa: E712
                Activity.start_time >= start_dt,
                Activity.start_time <= end_dt,
            )
            .order_by(Activity.start_time.desc())
        )
        if sport_type:
            stmt = stmt.where(Activity.sport_type == sport_type)
        if sport_types:
            stmt = stmt.where(Activity.sport_type.in_(list(sport_types)))

        stmt = _apply_range_filters(stmt, Activity, {
            "distance_meters": (min_distance_meters, max_distance_meters),
            "duration_seconds": (min_duration_seconds, max_duration_seconds),
            "calories": (min_calories, max_calories),
            "avg_heart_rate": (min_avg_heart_rate, max_avg_heart_rate),
            "avg_speed": (min_avg_speed, max_avg_speed),
            "elevation_meters": (min_elevation_meters, max_elevation_meters),
        })

        if title_contains:
            stmt = stmt.where(Activity.title.ilike(f"%{title_contains}%"))

        result = await db.execute(stmt)
        activities = result.scalars().all()
        data = [_activity_to_dict(a) for a in activities]

        logger.info(
            "Tool get_activities: user=%s date_from=%s date_to=%s sport=%s "
            "filters=%d → %d записей",
            user_id, date_from, date_to, sport_type,
            sum(
                1 for v in (
                    sport_types, min_distance_meters, max_distance_meters,
                    min_duration_seconds, max_duration_seconds,
                    min_calories, max_calories,
                    min_avg_heart_rate, max_avg_heart_rate,
                    min_avg_speed, max_avg_speed,
                    min_elevation_meters, max_elevation_meters,
                    title_contains,
                ) if v is not None
            ),
            len(data),
        )
        return ToolResult(tool_name="get_activities", success=True, data=data, error=None)
    except Exception as exc:
        logger.error("Tool get_activities ошибка: %s", exc)
        return ToolResult(tool_name="get_activities", success=False, data=None, error=str(exc))


async def get_activities_by_sport(
    db: AsyncSession,
    user_id: str,
    sport_type: str,
    days: int = 30,
    limit: int | None = None,
) -> ToolResult:
    """Получить тренировки пользователя по виду спорта.

    Args:
        db: Асинхронная сессия SQLAlchemy.
        user_id: Идентификатор пользователя.
        sport_type: Вид спорта (running, cycling и т.д.).
        days: Глубина поиска в днях (по умолчанию 30).
        limit: Ограничение количества записей (опционально).

    Returns:
        ToolResult с list[dict] активностей.
    """
    try:
        since_dt = datetime.utcnow() - timedelta(days=days)

        stmt = (
            select(Activity)
            .where(
                Activity.user_id == user_id,
                Activity.sport_type == sport_type,
                Activity.start_time >= since_dt,
            )
            .order_by(Activity.start_time.desc())
        )
        if limit is not None:
            stmt = stmt.limit(limit)

        result = await db.execute(stmt)
        activities = result.scalars().all()
        data = [_activity_to_dict(a) for a in activities]

        logger.info(
            "Tool get_activities_by_sport: user=%s sport=%s days=%d → %d записей",
            user_id, sport_type, days, len(data),
        )
        return ToolResult(tool_name="get_activities_by_sport", success=True, data=data, error=None)
    except Exception as exc:
        logger.error("Tool get_activities_by_sport ошибка: %s", exc)
        return ToolResult(tool_name="get_activities_by_sport", success=False, data=None, error=str(exc))


async def get_daily_facts(
    db: AsyncSession,
    user_id: str,
    date_from: date,
    date_to: date,
    metrics: list[str] | None = None,
    min_steps: int | None = None,
    max_steps: int | None = None,
    min_calories_kcal: int | None = None,
    max_calories_kcal: int | None = None,
    min_recovery_score: int | None = None,
    max_recovery_score: int | None = None,
    min_hrv_rmssd_milli: float | None = None,
    max_hrv_rmssd_milli: float | None = None,
    min_resting_heart_rate: int | None = None,
    max_resting_heart_rate: int | None = None,
    min_sleep_total_in_bed_milli: int | None = None,
    max_sleep_total_in_bed_milli: int | None = None,
    min_water_liters: float | None = None,
    max_water_liters: float | None = None,
    min_spo2_percentage: float | None = None,
    max_spo2_percentage: float | None = None,
) -> ToolResult:
    """Получить дневные метрики здоровья за период с whitelisted-фильтрами.

    Args:
        db: Асинхронная сессия SQLAlchemy.
        user_id: Идентификатор пользователя.
        date_from: Начало периода (включительно).
        date_to: Конец периода (включительно).
        metrics: Список полей для проекции (если None — все поля).
        min_*/max_* — диапазоны по числовым полям DailyFact. None-границы
            пропускаются. Сон передаётся в миллисекундах (как в БД).

    Returns:
        ToolResult с list[dict] дневных фактов.
    """
    try:
        stmt = (
            select(DailyFact)
            .where(
                DailyFact.user_id == user_id,
                DailyFact.iso_date >= date_from.isoformat(),
                DailyFact.iso_date <= date_to.isoformat(),
            )
            .order_by(DailyFact.iso_date.asc())
        )

        stmt = _apply_range_filters(stmt, DailyFact, {
            "steps": (min_steps, max_steps),
            "calories_kcal": (min_calories_kcal, max_calories_kcal),
            "recovery_score": (min_recovery_score, max_recovery_score),
            "hrv_rmssd_milli": (min_hrv_rmssd_milli, max_hrv_rmssd_milli),
            "resting_heart_rate": (min_resting_heart_rate, max_resting_heart_rate),
            "sleep_total_in_bed_milli": (
                min_sleep_total_in_bed_milli, max_sleep_total_in_bed_milli,
            ),
            "water_liters": (min_water_liters, max_water_liters),
            "spo2_percentage": (min_spo2_percentage, max_spo2_percentage),
        })

        result = await db.execute(stmt)
        facts = result.scalars().all()
        data = [_daily_fact_to_dict(f) for f in facts]

        # Если запрошены конкретные метрики — оставляем только нужные поля.
        # Метрики приходят в терминах MetricEnum («шаги», «калории», «сон»,
        # «recovery», ...) либо как имена столбцов; транслируем в имена полей
        # DailyFact. Неизвестные метрики молча пропускаем.
        if metrics:
            allowed: set[str] = {"id", "user_id", "iso_date"}
            for metric in metrics:
                allowed.update(_METRIC_TO_DAILY_FACT_FIELDS.get(metric, ()))
            data = [{k: v for k, v in row.items() if k in allowed} for row in data]

        logger.info(
            "Tool get_daily_facts: user=%s date_from=%s date_to=%s metrics=%s → %d записей",
            user_id, date_from, date_to, metrics, len(data),
        )
        return ToolResult(tool_name="get_daily_facts", success=True, data=data, error=None)
    except Exception as exc:
        logger.error("Tool get_daily_facts ошибка: %s", exc)
        return ToolResult(tool_name="get_daily_facts", success=False, data=None, error=str(exc))


async def get_user_profile(
    db: AsyncSession,
    user_id: str,
) -> ToolResult:
    """Получить профиль пользователя.

    Args:
        db: Асинхронная сессия SQLAlchemy.
        user_id: Идентификатор пользователя.

    Returns:
        ToolResult с dict профиля или data=None если профиль не найден.
    """
    try:
        stmt = select(UserProfile).where(UserProfile.user_id == user_id)
        result = await db.execute(stmt)
        profile = result.scalar_one_or_none()

        if profile is None:
            logger.info("Tool get_user_profile: user=%s профиль не найден", user_id)
            return ToolResult(tool_name="get_user_profile", success=True, data=None, error=None)

        data = _profile_to_dict(profile)
        logger.info("Tool get_user_profile: user=%s профиль загружен", user_id)
        return ToolResult(tool_name="get_user_profile", success=True, data=data, error=None)
    except Exception as exc:
        logger.error("Tool get_user_profile ошибка: %s", exc)
        return ToolResult(tool_name="get_user_profile", success=False, data=None, error=str(exc))


# ---------------------------------------------------------------------------
# Tool-функции (запись)
# ---------------------------------------------------------------------------


async def log_activity(
    db: AsyncSession,
    user_id: str,
    sport_type: str,
    duration: int,
    calories: int = 0,
    distance: float | None = None,
    notes: str | None = None,
) -> ToolResult:
    """Записать тренировку, введённую пользователем в чате.

    Args:
        db: Асинхронная сессия SQLAlchemy.
        user_id: Идентификатор пользователя.
        sport_type: Вид спорта.
        duration: Продолжительность в секундах.
        calories: Калории (по умолчанию 0).
        distance: Дистанция в метрах (опционально).
        notes: Краткое описание (используется как title).

    Returns:
        ToolResult с dict созданной активности.
    """
    try:
        now = datetime.utcnow()
        title = notes[:100] if notes else f"{sport_type.capitalize()} {now.strftime('%d.%m.%Y')}"

        activity = Activity(
            id=str(uuid.uuid4()),
            user_id=user_id,
            title=title,
            sport_type=sport_type,
            duration_seconds=duration,
            start_time=now - timedelta(seconds=duration),
            end_time=now,
            calories=calories,
            distance_meters=distance,
            source="manual",
            is_primary=True,
            anomaly_flags=[],
            raw_title=title,
        )
        db.add(activity)
        await db.commit()
        await db.refresh(activity)

        data = _activity_to_dict(activity)
        logger.info(
            "Tool log_activity: user=%s sport=%s duration=%ds calories=%d → id=%s",
            user_id, sport_type, duration, calories, activity.id,
        )
        return ToolResult(tool_name="log_activity", success=True, data=data, error=None)
    except Exception as exc:
        await db.rollback()
        logger.error("Tool log_activity ошибка: %s", exc)
        return ToolResult(tool_name="log_activity", success=False, data=None, error=str(exc))


async def update_profile(
    db: AsyncSession,
    user_id: str,
    field: str,
    value: Any,
) -> ToolResult:
    """Обновить одно поле профиля пользователя.

    Args:
        db: Асинхронная сессия SQLAlchemy.
        user_id: Идентификатор пользователя.
        field: Имя поля для обновления (из ALLOWED_FIELDS).
        value: Новое значение поля.

    Returns:
        ToolResult с обновлённым dict профиля.
    """
    ALLOWED_FIELDS = {
        "name", "age", "weight_kg", "height_cm", "gender",
        "max_heart_rate", "resting_heart_rate", "training_goals",
        "experience_level", "injuries", "chronic_conditions", "preferred_sports",
    }

    try:
        if field not in ALLOWED_FIELDS:
            logger.warning("Tool update_profile: недопустимое поле '%s'", field)
            return ToolResult(
                tool_name="update_profile",
                success=False,
                data=None,
                error=f"Поле '{field}' не разрешено для обновления",
            )

        stmt = select(UserProfile).where(UserProfile.user_id == user_id)
        result = await db.execute(stmt)
        profile = result.scalar_one_or_none()

        if profile is None:
            return ToolResult(
                tool_name="update_profile",
                success=False,
                data=None,
                error=f"Профиль пользователя {user_id} не найден",
            )

        setattr(profile, field, value)
        await db.commit()
        await db.refresh(profile)

        data = _profile_to_dict(profile)
        logger.info("Tool update_profile: user=%s поле=%s обновлено", user_id, field)
        return ToolResult(tool_name="update_profile", success=True, data=data, error=None)
    except Exception as exc:
        await db.rollback()
        logger.error("Tool update_profile ошибка: %s", exc)
        return ToolResult(tool_name="update_profile", success=False, data=None, error=str(exc))
