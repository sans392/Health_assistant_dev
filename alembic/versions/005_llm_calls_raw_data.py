"""llm_calls: raw request/response payload + endpoint/stream/http_status/error

Revision ID: 005
Revises: 004
Create Date: 2026-04-18 00:00:00.000000

Изменения:
- llm_calls +6 колонок (все nullable, совместимо со старыми строками):
    * endpoint       (String)  — /api/generate | /api/chat | /api/embeddings
    * stream         (Boolean) — использовался ли стриминг
    * http_status    (Integer) — HTTP статус ответа Ollama
    * request_body   (JSON)    — сырой payload запроса
    * response_body  (JSON)    — сырой JSON ответа Ollama
    * error          (Text)    — текст ошибки (timeout/5xx/парсинг)

Старые записи (без этих полей) отображаются в админке как есть.
SQLite поддерживает ALTER TABLE ADD COLUMN для nullable-колонок.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("llm_calls", sa.Column("endpoint", sa.String(50), nullable=True))
    op.add_column("llm_calls", sa.Column("stream", sa.Boolean, nullable=True))
    op.add_column("llm_calls", sa.Column("http_status", sa.Integer, nullable=True))
    op.add_column("llm_calls", sa.Column("request_body", sa.JSON, nullable=True))
    op.add_column("llm_calls", sa.Column("response_body", sa.JSON, nullable=True))
    op.add_column("llm_calls", sa.Column("error", sa.Text, nullable=True))


def downgrade() -> None:
    # SQLite не поддерживает DROP COLUMN напрямую — пересобираем таблицу.
    op.create_table(
        "llm_calls_backup",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "request_id",
            sa.String(36),
            sa.ForeignKey("pipeline_logs.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("prompt", sa.Text, nullable=True),
        sa.Column("response", sa.Text, nullable=True),
        sa.Column("prompt_length", sa.Integer, nullable=True),
        sa.Column("response_length", sa.Integer, nullable=True),
        sa.Column("duration_ms", sa.Integer, nullable=True),
        sa.Column("iteration", sa.Integer, nullable=True),
        sa.Column("timestamp", sa.DateTime, nullable=False),
    )
    op.execute(
        "INSERT INTO llm_calls_backup "
        "SELECT id, request_id, role, model, prompt, response, "
        "prompt_length, response_length, duration_ms, iteration, timestamp "
        "FROM llm_calls"
    )
    op.drop_table("llm_calls")
    op.rename_table("llm_calls_backup", "llm_calls")
