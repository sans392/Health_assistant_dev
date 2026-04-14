"""Initial schema — все таблицы

Revision ID: 001
Revises:
Create Date: 2026-04-14 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "user_profiles",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), nullable=False, unique=True, index=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("age", sa.Integer, nullable=False),
        sa.Column("weight_kg", sa.Float, nullable=False),
        sa.Column("height_cm", sa.Float, nullable=False),
        sa.Column("gender", sa.String(20), nullable=False, server_default="male"),
        sa.Column("max_heart_rate", sa.Integer, nullable=True),
        sa.Column("resting_heart_rate", sa.Integer, nullable=True),
        sa.Column("training_goals", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("experience_level", sa.String(20), nullable=False, server_default="beginner"),
        sa.Column("injuries", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("chronic_conditions", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("preferred_sports", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("updated_at", sa.DateTime, nullable=False),
    )

    op.create_table(
        "activities",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), nullable=False, index=True),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("sport_type", sa.String(50), nullable=False, index=True),
        sa.Column("distance_meters", sa.Float, nullable=True),
        sa.Column("duration_seconds", sa.Integer, nullable=False),
        sa.Column("start_time", sa.DateTime, nullable=False, index=True),
        sa.Column("end_time", sa.DateTime, nullable=False),
        sa.Column("avg_speed", sa.Float, nullable=True),
        sa.Column("max_speed", sa.Float, nullable=True),
        sa.Column("elevation_meters", sa.Float, nullable=True),
        sa.Column("calories", sa.Integer, nullable=False, server_default="0"),
        sa.Column("avg_heart_rate", sa.Integer, nullable=True),
        sa.Column("max_heart_rate", sa.Integer, nullable=True),
        sa.Column("source", sa.String(50), nullable=False, server_default="manual"),
        sa.Column("is_primary", sa.Boolean, nullable=False, server_default="1"),
        sa.Column("anomaly_flags", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("raw_title", sa.String(255), nullable=False, server_default=""),
    )

    op.create_table(
        "daily_facts",
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

    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), nullable=False, index=True),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("updated_at", sa.DateTime, nullable=False),
    )

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "session_id",
            sa.String(36),
            sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("order_index", sa.Integer, nullable=False, server_default="0"),
    )

    op.create_table(
        "pipeline_logs",
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


def downgrade() -> None:
    op.drop_table("pipeline_logs")
    op.drop_table("chat_messages")
    op.drop_table("chat_sessions")
    op.drop_table("daily_facts")
    op.drop_table("activities")
    op.drop_table("user_profiles")
