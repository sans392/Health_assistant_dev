"""Seed runs journal table

Revision ID: 004
Revises: 003
Create Date: 2026-04-17 00:00:00.000000

Изменения:
- Новая таблица seed_runs: журнал запусков SeedGenerator через Admin UI
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "seed_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("params", sa.JSON, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("admin_user", sa.String(100), nullable=False, server_default="admin"),
        sa.Column("records_created", sa.JSON, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("seed_runs")
