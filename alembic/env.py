"""Конфигурация среды Alembic для миграций."""

from logging.config import fileConfig
import os

from sqlalchemy import engine_from_config, pool

from alembic import context

# Импортируем все модели, чтобы Alembic их обнаружил
from app.db import Base
import app.models  # noqa: F401 — регистрирует все модели

from app.config import settings

config = context.config

# Настройка логирования из alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Целевые метаданные для автогенерации миграций
target_metadata = Base.metadata


def get_url() -> str:
    """Получить URL базы данных (синхронный, для Alembic)."""
    return settings.database_url_sync


def run_migrations_offline() -> None:
    """Запуск миграций в offline-режиме (без подключения к БД)."""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Запуск миграций в online-режиме (с подключением к БД)."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
