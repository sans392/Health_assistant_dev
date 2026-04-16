"""Phase 2 schema — LLM registry, LLM calls log, RAG chunks, extended fields

Revision ID: 003
Revises: 002
Create Date: 2026-04-16 00:00:00.000000

Изменения:
- Новая таблица llm_role_config  : persistent конфиг моделей по ролям
- Новая таблица llm_calls        : детальный лог каждого LLM-вызова
- Новая таблица rag_chunks       : метаданные RAG-чанков
- pipeline_logs +3 колонки       : rag_chunks_used, stage_trace, llm_role_usage
- daily_facts +3 колонки         : recovery_score_calculated, strain_score, anomaly_flags

SQLite поддерживает ALTER TABLE ADD COLUMN для nullable-колонок и колонок
с простым server_default — именно такие все добавляемые колонки.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------ #
    # Новая таблица llm_role_config                                        #
    # ------------------------------------------------------------------ #
    op.create_table(
        "llm_role_config",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("role", sa.String(20), nullable=False, unique=True, index=True),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("updated_at", sa.DateTime, nullable=False),
    )

    # ------------------------------------------------------------------ #
    # Новая таблица llm_calls                                              #
    # ------------------------------------------------------------------ #
    op.create_table(
        "llm_calls",
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

    # ------------------------------------------------------------------ #
    # Новая таблица rag_chunks                                             #
    # ------------------------------------------------------------------ #
    op.create_table(
        "rag_chunks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("category", sa.String(50), nullable=False, index=True),
        sa.Column("source", sa.String(200), nullable=False),
        sa.Column("confidence", sa.String(10), nullable=False, server_default="medium"),
        sa.Column("sport_type", sa.String(50), nullable=True),
        sa.Column("experience_level", sa.String(20), nullable=True),
        sa.Column("embedding_id", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )

    # ------------------------------------------------------------------ #
    # Расширение pipeline_logs                                             #
    # ------------------------------------------------------------------ #
    op.add_column(
        "pipeline_logs",
        sa.Column("rag_chunks_used", sa.JSON, nullable=True),
    )
    op.add_column(
        "pipeline_logs",
        sa.Column("stage_trace", sa.JSON, nullable=True),
    )
    op.add_column(
        "pipeline_logs",
        sa.Column("llm_role_usage", sa.JSON, nullable=True),
    )

    # ------------------------------------------------------------------ #
    # Расширение daily_facts                                               #
    # ------------------------------------------------------------------ #
    op.add_column(
        "daily_facts",
        sa.Column("recovery_score_calculated", sa.Integer, nullable=True),
    )
    op.add_column(
        "daily_facts",
        sa.Column("strain_score", sa.Float, nullable=True),
    )
    op.add_column(
        "daily_facts",
        sa.Column("anomaly_flags", sa.JSON, nullable=True, server_default="[]"),
    )


def downgrade() -> None:
    # Удаляем новые таблицы
    op.drop_table("rag_chunks")
    op.drop_table("llm_calls")
    op.drop_table("llm_role_config")

    # Для daily_facts — SQLite не поддерживает DROP COLUMN напрямую,
    # используем пересоздание таблицы
    op.create_table(
        "daily_facts_backup",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), nullable=False, index=True),
        sa.Column("iso_date", sa.String(10), nullable=False, index=True),
        sa.Column("steps", sa.Integer, nullable=False, server_default="0"),
        sa.Column("calories_kcal", sa.Integer, nullable=True),
        sa.Column("recovery_score", sa.Integer, nullable=True),
        sa.Column("hrv_rmssd_milli", sa.Float, nullable=True),
        sa.Column("resting_heart_rate", sa.Integer, nullable=True),
        sa.Column("spo2_percentage", sa.Float, nullable=True),
        sa.Column("skin_temp_celsius", sa.Float, nullable=True),
        sa.Column("sleep_total_in_bed_milli", sa.Integer, nullable=True),
        sa.Column("water_liters", sa.Float, nullable=True),
        sa.Column("sources_json", sa.JSON, nullable=False, server_default="{}"),
    )
    op.execute(
        "INSERT INTO daily_facts_backup "
        "SELECT id, user_id, iso_date, steps, calories_kcal, recovery_score, "
        "hrv_rmssd_milli, resting_heart_rate, spo2_percentage, skin_temp_celsius, "
        "sleep_total_in_bed_milli, water_liters, sources_json "
        "FROM daily_facts"
    )
    op.drop_table("daily_facts")
    op.rename_table("daily_facts_backup", "daily_facts")

    # Для pipeline_logs — то же
    op.create_table(
        "pipeline_logs_backup",
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
    op.execute(
        "INSERT INTO pipeline_logs_backup "
        "SELECT id, user_id, session_id, timestamp, raw_query, intent, "
        "intent_confidence, route, fast_path, safety_level, tools_called, "
        "modules_used, llm_model_used, llm_calls_count, total_duration_ms, "
        "response_length, errors, response_text "
        "FROM pipeline_logs"
    )
    op.drop_table("pipeline_logs")
    op.rename_table("pipeline_logs_backup", "pipeline_logs")
