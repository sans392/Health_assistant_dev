"""Модель журнала запусков Seed Generator."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class SeedRun(Base):
    """Запись о запуске SeedGenerator через Admin UI."""

    __tablename__ = "seed_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    params: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    admin_user: Mapped[str] = mapped_column(String(100), nullable=False, default="admin")
    records_created: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
