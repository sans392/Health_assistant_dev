"""tool_calls: детальный лог вызовов Tool Executor

Revision ID: 006
Revises: 005
Create Date: 2026-04-18 00:00:00.000000

Создаётся таблица tool_calls, аналогичная llm_calls, для хранения полной
картины tool-вызовов (args, result, success, error, duration_ms) с привязкой
к request_id. Источники: tool_executor | planner | template.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "tool_calls",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "request_id",
            sa.String(36),
            sa.ForeignKey("pipeline_logs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("source", sa.String(20), nullable=False),
        sa.Column("iteration", sa.Integer, nullable=True),
        sa.Column("step_id", sa.String(100), nullable=True),
        sa.Column("args", sa.JSON, nullable=True),
        sa.Column("result", sa.JSON, nullable=True),
        sa.Column("success", sa.Boolean, nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("duration_ms", sa.Integer, nullable=True),
        sa.Column("timestamp", sa.DateTime, nullable=False),
    )
    op.create_index(
        "ix_tool_calls_request_id",
        "tool_calls",
        ["request_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_tool_calls_request_id", table_name="tool_calls")
    op.drop_table("tool_calls")
