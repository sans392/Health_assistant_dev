"""pending_clarifications: хранение незавершённого уточняющего диалога

Revision ID: 007
Revises: 006
Create Date: 2026-04-23 00:00:00.000000

Одна запись на session_id — PK. Оркестратор сохраняет состояние, когда
intent требует обязательные слоты, а intent detection их не извлёк. При
следующем user message проверяем pending, мёржим слоты из нового сообщения
и продолжаем обработку с полным контекстом. TTL ~3 минуты.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "pending_clarifications",
        sa.Column("session_id", sa.String(36), primary_key=True),
        sa.Column("intent", sa.String(50), nullable=False),
        sa.Column("original_query", sa.Text, nullable=False),
        sa.Column("filled_slots", sa.JSON, nullable=False),
        sa.Column("missing_slots", sa.JSON, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("expires_at", sa.DateTime, nullable=False),
    )
    op.create_index(
        "ix_pending_clarifications_expires_at",
        "pending_clarifications",
        ["expires_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_pending_clarifications_expires_at",
        table_name="pending_clarifications",
    )
    op.drop_table("pending_clarifications")
