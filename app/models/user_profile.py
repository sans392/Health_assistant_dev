"""Модель профиля пользователя."""

import uuid
from datetime import datetime

from sqlalchemy import String, Float, Integer, DateTime, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class UserProfile(Base):
    """Профиль пользователя с параметрами здоровья и тренировочными целями."""

    __tablename__ = "user_profiles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    weight_kg: Mapped[float] = mapped_column(Float, nullable=False)
    height_cm: Mapped[float] = mapped_column(Float, nullable=False)
    # male | female | other
    gender: Mapped[str] = mapped_column(String(20), nullable=False, default="male")
    max_heart_rate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    resting_heart_rate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Хранится как JSON-массив строк
    training_goals: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    # beginner | intermediate | advanced
    experience_level: Mapped[str] = mapped_column(String(20), nullable=False, default="beginner")
    # Хранится как JSON-массив объектов травм
    injuries: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    # Хранится как JSON-массив строк
    chronic_conditions: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    # Хранится как JSON-массив строк
    preferred_sports: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )
