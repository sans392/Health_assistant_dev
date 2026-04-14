"""Тесты ToolExecutor (app/pipeline/tool_executor.py).

Используют моки БД — без реального SQLite.
"""

from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.pipeline.tool_executor import ToolExecutor, ToolExecutorResult
from app.tools.db_tools import ToolResult


def _make_tool_result(tool_name: str, data: object = None, success: bool = True) -> ToolResult:
    return ToolResult(tool_name=tool_name, success=success, data=data, error=None)


class TestToolExecutorResult:
    """Тесты датакласса ToolExecutorResult."""

    def test_success_true_if_any_success(self) -> None:
        result = ToolExecutorResult(results={
            "get_activities": _make_tool_result("get_activities", data=[], success=True),
            "get_daily_facts": _make_tool_result("get_daily_facts", success=False),
        })
        assert result.success is True

    def test_success_false_if_all_failed(self) -> None:
        result = ToolExecutorResult(results={
            "get_activities": _make_tool_result("get_activities", success=False),
        })
        assert result.success is False

    def test_get_data_returns_data(self) -> None:
        data = [{"id": "1"}]
        result = ToolExecutorResult(results={
            "get_activities": _make_tool_result("get_activities", data=data),
        })
        assert result.get_data("get_activities") == data

    def test_get_data_returns_none_for_failed(self) -> None:
        result = ToolExecutorResult(results={
            "get_activities": _make_tool_result("get_activities", success=False),
        })
        assert result.get_data("get_activities") is None

    def test_get_data_returns_none_for_missing(self) -> None:
        result = ToolExecutorResult()
        assert result.get_data("nonexistent") is None

    def test_all_data_returns_successful_only(self) -> None:
        result = ToolExecutorResult(results={
            "get_activities": _make_tool_result("get_activities", data=[1, 2]),
            "get_daily_facts": _make_tool_result("get_daily_facts", success=False),
        })
        all_data = result.all_data()
        assert "get_activities" in all_data
        assert "get_daily_facts" not in all_data


@pytest.mark.asyncio
class TestToolExecutorExecute:
    """Тесты метода ToolExecutor.execute."""

    async def test_dispatches_get_activities(self) -> None:
        executor = ToolExecutor()
        db = MagicMock()
        mock_result = _make_tool_result("get_activities", data=[])

        with patch("app.pipeline.tool_executor.get_activities", new=AsyncMock(return_value=mock_result)):
            result = await executor.execute(
                tool_calls=["get_activities"],
                user_id="user-1",
                entities={"time_range": "за неделю"},
                db=db,
            )

        assert "get_activities" in result.results
        assert result.results["get_activities"].success is True

    async def test_dispatches_get_daily_facts(self) -> None:
        executor = ToolExecutor()
        db = MagicMock()
        mock_result = _make_tool_result("get_daily_facts", data=[])

        with patch("app.pipeline.tool_executor.get_daily_facts", new=AsyncMock(return_value=mock_result)):
            result = await executor.execute(
                tool_calls=["get_daily_facts"],
                user_id="user-1",
                entities={},
                db=db,
            )

        assert "get_daily_facts" in result.results

    async def test_unknown_tool_returns_error(self) -> None:
        executor = ToolExecutor()
        db = MagicMock()

        result = await executor.execute(
            tool_calls=["nonexistent_tool"],
            user_id="user-1",
            entities={},
            db=db,
        )

        assert "nonexistent_tool" in result.results
        assert result.results["nonexistent_tool"].success is False
        assert "Неизвестный tool" in result.results["nonexistent_tool"].error

    async def test_multiple_tools_called(self) -> None:
        executor = ToolExecutor()
        db = MagicMock()
        mock_act = _make_tool_result("get_activities", data=[])
        mock_facts = _make_tool_result("get_daily_facts", data=[])

        with (
            patch("app.pipeline.tool_executor.get_activities", new=AsyncMock(return_value=mock_act)),
            patch("app.pipeline.tool_executor.get_daily_facts", new=AsyncMock(return_value=mock_facts)),
        ):
            result = await executor.execute(
                tool_calls=["get_activities", "get_daily_facts"],
                user_id="user-1",
                entities={"time_range": "сегодня"},
                db=db,
            )

        assert len(result.results) == 2
        assert "get_activities" in result.results
        assert "get_daily_facts" in result.results

    async def test_time_range_resolved_to_dates(self) -> None:
        """Проверяем, что time_range entity резолвится в конкретные даты."""
        executor = ToolExecutor()
        db = MagicMock()
        mock_result = _make_tool_result("get_activities", data=[])
        captured_args: dict = {}

        async def capture_get_activities(db, user_id, date_from, date_to, sport_type=None):
            captured_args["date_from"] = date_from
            captured_args["date_to"] = date_to
            return mock_result

        with patch("app.pipeline.tool_executor.get_activities", new=capture_get_activities):
            await executor.execute(
                tool_calls=["get_activities"],
                user_id="user-1",
                entities={"time_range": "за неделю"},
                db=db,
            )

        today = date.today()
        assert captured_args["date_to"] == today
        assert captured_args["date_from"] == today - timedelta(days=6)
