"""Модель тренировки (активности)."""

import uuid
from datetime import datetime

from sqlalchemy import String, Float, Integer, DateTime, Boolean, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Activity(Base):
    """Нормализованная тренировка."""

    __tablename__ = "activities"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    # running | cycling | swimming | gym | walking | activity | other
    sport_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    distance_meters: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    end_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    avg_speed: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_speed: Mapped[float | None] = mapped_column(Float, nullable=True)
    elevation_meters: Mapped[float | None] = mapped_column(Float, nullable=True)
    calories: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_heart_rate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_heart_rate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # whoop | apple_health | garmin | manual
    source: Mapped[str] = mapped_column(String(50), nullable=False, default="manual")
    # true = выбрана из группы дубликатов
    is_primary: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    # JSON-массив строк: ["calorie_outlier", ...]
    anomaly_flags: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    raw_title: Mapped[str] = mapped_column(String(255), nullable=False, default="")
