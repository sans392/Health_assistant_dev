"""Update pipeline_logs schema for issue #12

Revision ID: 002
Revises: 001
Create Date: 2026-04-14 00:01:00.000000

Пересоздаёт таблицу pipeline_logs с новой схемой из issue #12.
SQLite не поддерживает DROP COLUMN и RENAME COLUMN, поэтому
использует подход: создать новую таблицу → скопировать данные → удалить старую → переименовать.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Создаём новую таблицу с обновлённой схемой
    op.create_table(
        "pipeline_logs_new",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), nullable=True, index=True),
        sa.Column("session_id", sa.String(36), nullable=True, index=True),
        sa.Column("timestamp", sa.DateTime, nullable=False),
        sa.Column("raw_query", sa.Text, nullable=False),
        sa.Column("intent", sa.String(50), nullable=True, index=True),
        sa.Column("intent_confidence", sa.Float, nullable=True),
        sa.Column("route", sa.String(50), nullable=True, index=True),
        sa.Column("fast_path", sa.Boolean, nullable=True),
        sa.Column("safety_level", sa.String(20), nullable=True),
        sa.Column("tools_called", sa.JSON, nullable=True),
        sa.Column("modules_used", sa.JSON, nullable=True),
        sa.Column("llm_model_used", sa.String(100), nullable=True),
        sa.Column("llm_calls_count", sa.Integer, nullable=True),
        sa.Column("total_duration_ms", sa.Integer, nullable=True),
        sa.Column("response_length", sa.Integer, nullable=True),
        sa.Column("errors", sa.JSON, nullable=True),
        sa.Column("response_text", sa.Text, nullable=True),
    )

    # Удаляем старую таблицу и переименовываем новую
    op.drop_table("pipeline_logs")
    op.rename_table("pipeline_logs_new", "pipeline_logs")


def downgrade() -> None:
    # Пересоздаём старую схему
    op.create_table(
        "pipeline_logs_old",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("session_id", sa.String(36), nullable=True, index=True),
        sa.Column("user_id", sa.String(36), nullable=True),
        sa.Column("user_query", sa.Text, nullable=False),
        sa.Column("intent", sa.String(50), nullable=True),
        sa.Column("safety_passed", sa.Boolean, nullable=True),
        sa.Column("route", sa.String(50), nullable=True),
        sa.Column("pipeline_path", sa.String(20), nullable=True),
        sa.Column("llm_model", sa.String(100), nullable=True),
        sa.Column("prompt_length", sa.Integer, nullable=True),
        sa.Column("response_length", sa.Integer, nullable=True),
        sa.Column("llm_duration_ms", sa.Float, nullable=True),
        sa.Column("total_duration_ms", sa.Float, nullable=True),
        sa.Column("extra_data", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )
    op.drop_table("pipeline_logs")
    op.rename_table("pipeline_logs_old", "pipeline_logs")
