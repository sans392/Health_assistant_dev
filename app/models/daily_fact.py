"""Модель дневных метрик здоровья."""

import uuid

from sqlalchemy import String, Float, Integer, Date, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class DailyFact(Base):
    """Дневные метрики (одна запись на дату на пользователя)."""

    __tablename__ = "daily_facts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    iso_date: Mapped[str] = mapped_column(String(10), nullable=False, index=True)  # YYYY-MM-DD
    steps: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    calories_kcal: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Нативный recovery score (от Whoop)
    recovery_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    hrv_rmssd_milli: Mapped[float | None] = mapped_column(Float, nullable=True)
    resting_heart_rate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    spo2_percentage: Mapped[float | None] = mapped_column(Float, nullable=True)
    skin_temp_celsius: Mapped[float | None] = mapped_column(Float, nullable=True)
    sleep_total_in_bed_milli: Mapped[int | None] = mapped_column(Integer, nullable=True)
    water_liters: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Маппинг метрика → источник данных
    sources_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
