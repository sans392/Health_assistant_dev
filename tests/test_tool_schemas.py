"""Тесты Pydantic-схем аргументов tool'ов (app/tools/schemas.py)."""

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from app.tools.schemas import (
    CheckOvertrainingArgs,
    ComputeRecoveryArgs,
    ComputeStrainArgs,
    GetActivitiesArgs,
    GetActivitiesBySportArgs,
    GetDailyFactsArgs,
    GetUserProfileArgs,
    LogActivityArgs,
    MetricEnum,
    RagRetrieveArgs,
    SportTypeEnum,
    TimeRange,
    UpdateProfileArgs,
    tool_to_prompt_signature,
    validate_tool_args,
)


class TestTimeRange:
    def test_from_valid_dates(self) -> None:
        tr = TimeRange(date_from=date(2026, 4, 1), date_to=date(2026, 4, 7))
        assert tr.days == 7
        assert tr.as_tuple() == (date(2026, 4, 1), date(2026, 4, 7))

    def test_reject_inverted(self) -> None:
        with pytest.raises(ValidationError):
            TimeRange(date_from=date(2026, 4, 7), date_to=date(2026, 4, 1))

    def test_same_day_has_days_one(self) -> None:
        tr = TimeRange(date_from=date(2026, 4, 1), date_to=date(2026, 4, 1))
        assert tr.days == 1

    def test_last_n_days_inclusive(self) -> None:
        today = date(2026, 4, 23)
        tr = TimeRange.last_n_days(7, today=today)
        assert tr.date_from == today - timedelta(days=6)
        assert tr.date_to == today
        assert tr.days == 7

    def test_last_n_days_rejects_zero(self) -> None:
        with pytest.raises(ValueError):
            TimeRange.last_n_days(0)

    def test_frozen_model(self) -> None:
        tr = TimeRange(date_from=date(2026, 4, 1), date_to=date(2026, 4, 7))
        with pytest.raises(ValidationError):
            tr.date_from = date(2026, 4, 5)


class TestGetActivitiesArgs:
    def test_valid(self) -> None:
        args = GetActivitiesArgs(
            user_id="u1",
            date_from=date(2026, 4, 1),
            date_to=date(2026, 4, 7),
        )
        assert args.sport_type is None

    def test_enum_sport_type(self) -> None:
        args = GetActivitiesArgs(
            user_id="u1",
            date_from=date(2026, 4, 1),
            date_to=date(2026, 4, 7),
            sport_type="running",
        )
        assert args.sport_type == SportTypeEnum.RUNNING

    def test_invalid_sport_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GetActivitiesArgs(
                user_id="u1",
                date_from=date(2026, 4, 1),
                date_to=date(2026, 4, 7),
                sport_type="квиддич",
            )

    def test_inverted_dates_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GetActivitiesArgs(
                user_id="u1",
                date_from=date(2026, 4, 7),
                date_to=date(2026, 4, 1),
            )

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GetActivitiesArgs(
                user_id="u1",
                date_from=date(2026, 4, 1),
                date_to=date(2026, 4, 7),
                unknown_field="x",
            )

    def test_empty_user_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GetActivitiesArgs(
                user_id="",
                date_from=date(2026, 4, 1),
                date_to=date(2026, 4, 7),
            )


class TestGetActivitiesBySportArgs:
    def test_days_bounds(self) -> None:
        with pytest.raises(ValidationError):
            GetActivitiesBySportArgs(user_id="u1", sport_type="running", days=0)
        with pytest.raises(ValidationError):
            GetActivitiesBySportArgs(user_id="u1", sport_type="running", days=500)

    def test_limit_bounds(self) -> None:
        args = GetActivitiesBySportArgs(
            user_id="u1", sport_type="running", days=30, limit=10
        )
        assert args.limit == 10


class TestGetDailyFactsArgs:
    def test_metrics_coerced_to_enum(self) -> None:
        args = GetDailyFactsArgs(
            user_id="u1",
            date_from=date(2026, 4, 1),
            date_to=date(2026, 4, 7),
            metrics=["heart_rate", "шаги"],
        )
        assert args.metrics == [MetricEnum.HEART_RATE, MetricEnum.STEPS]

    def test_unknown_metric_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GetDailyFactsArgs(
                user_id="u1",
                date_from=date(2026, 4, 1),
                date_to=date(2026, 4, 7),
                metrics=["unknown_metric"],
            )


class TestComputeRecoveryArgs:
    def test_window_days_default(self) -> None:
        args = ComputeRecoveryArgs(user_id="u1")
        assert args.window_days == 14

    def test_window_days_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ComputeRecoveryArgs(user_id="u1", window_days=1)
        with pytest.raises(ValidationError):
            ComputeRecoveryArgs(user_id="u1", window_days=100)

    def test_custom_window(self) -> None:
        args = ComputeRecoveryArgs(user_id="u1", window_days=30)
        assert args.window_days == 30


class TestCheckOvertrainingArgs:
    def test_default_window(self) -> None:
        args = CheckOvertrainingArgs(user_id="u1")
        assert args.window_days == 14


class TestRagRetrieveArgs:
    def test_query_required(self) -> None:
        with pytest.raises(ValidationError):
            RagRetrieveArgs(query_text="")

    def test_top_k_clamped(self) -> None:
        with pytest.raises(ValidationError):
            RagRetrieveArgs(query_text="test", top_k=0)
        with pytest.raises(ValidationError):
            RagRetrieveArgs(query_text="test", top_k=100)

    def test_default_top_k(self) -> None:
        args = RagRetrieveArgs(query_text="сколько белка")
        assert args.top_k == 5


class TestLogActivityArgs:
    def test_valid(self) -> None:
        args = LogActivityArgs(
            user_id="u1", sport_type="running", duration=1800, calories=200
        )
        assert args.sport_type == SportTypeEnum.RUNNING

    def test_negative_duration_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LogActivityArgs(user_id="u1", sport_type="running", duration=-10)


class TestUpdateProfileArgs:
    def test_valid_field(self) -> None:
        args = UpdateProfileArgs(user_id="u1", field="weight_kg", value=75)
        assert args.field == "weight_kg"

    def test_invalid_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            UpdateProfileArgs(user_id="u1", field="password", value="x")


class TestValidateToolArgs:
    def test_unknown_tool_raises_keyerror(self) -> None:
        with pytest.raises(KeyError):
            validate_tool_args("nonexistent_tool", {"user_id": "u1"})

    def test_validates_get_activities(self) -> None:
        args = validate_tool_args(
            "get_activities",
            {
                "user_id": "u1",
                "date_from": "2026-04-01",
                "date_to": "2026-04-07",
                "sport_type": "running",
            },
        )
        assert isinstance(args, GetActivitiesArgs)
        assert args.date_from == date(2026, 4, 1)

    def test_validation_error_for_bad_args(self) -> None:
        with pytest.raises(ValidationError):
            validate_tool_args(
                "get_activities",
                {"user_id": "u1", "date_from": "2026-04-07", "date_to": "2026-04-01"},
            )


class TestToolPromptSignature:
    """Сигнатуры из Pydantic-схем для system-prompt планера.

    Сигнатура должна совпадать с реальным контрактом ToolExecutor'а — иначе
    планер просит несуществующие поля (`days` вместо `date_from`/`date_to`).
    """

    def test_get_activities_uses_date_from_to(self) -> None:
        sig = tool_to_prompt_signature("get_activities")
        assert sig.startswith("get_activities(")
        assert "date_from: YYYY-MM-DD" in sig
        assert "date_to: YYYY-MM-DD" in sig
        # sport_type — опциональное поле
        assert "sport_type?" in sig
        # user_id и tool скрыты — заполняются pipeline'ом, не LLM
        assert "user_id" not in sig
        assert "tool:" not in sig
        # Старого хардкода `days: int` для get_activities быть не должно
        assert "days:" not in sig

    def test_get_daily_facts_signature(self) -> None:
        sig = tool_to_prompt_signature("get_daily_facts")
        assert "date_from: YYYY-MM-DD" in sig
        assert "date_to: YYYY-MM-DD" in sig
        assert "metrics?" in sig

    def test_compute_recovery_window_days_marked_optional(self) -> None:
        sig = tool_to_prompt_signature("compute_recovery")
        # У window_days есть default — должно быть optional
        assert "window_days?: int" in sig

    def test_sport_type_enum_values_listed(self) -> None:
        sig = tool_to_prompt_signature("get_activities")
        # Хотя бы пара значений enum'а должна быть в сигнатуре —
        # без них LLM не знает, что туда подставлять.
        assert "running" in sig
        assert "cycling" in sig

    def test_rag_retrieve_signature(self) -> None:
        sig = tool_to_prompt_signature("rag_retrieve")
        assert "query_text: str" in sig
        assert "top_k?" in sig

    def test_signature_for_unknown_tool_raises(self) -> None:
        with pytest.raises(KeyError):
            tool_to_prompt_signature("nonexistent_tool")
