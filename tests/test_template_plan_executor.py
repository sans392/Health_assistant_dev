"""Тесты для Template Plan Executor (Issue #28)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.pipeline.template_plan_executor import (
    TEMPLATES,
    TemplatePlanExecutor,
    TemplateResult,
    TemplateStepResult,
)


def _make_mock_db() -> AsyncMock:
    return AsyncMock()


def _make_entities(sport_type: str | None = None) -> dict:
    return {"sport_type": sport_type} if sport_type else {}


class TestTemplateRegistry:
    """Тесты реестра шаблонов."""

    def test_all_four_templates_exist(self) -> None:
        assert "weekly_training_plan" in TEMPLATES
        assert "recovery_report" in TEMPLATES
        assert "overtraining_check" in TEMPLATES
        assert "progress_report" in TEMPLATES

    def test_recovery_report_steps(self) -> None:
        steps = TEMPLATES["recovery_report"]
        tools = [s["tool"] for s in steps]
        assert "get_daily_facts" in tools
        assert "compute_recovery" in tools
        assert "rag_retrieve" in tools

    def test_overtraining_check_steps(self) -> None:
        steps = TEMPLATES["overtraining_check"]
        tools = [s["tool"] for s in steps]
        assert "get_daily_facts" in tools
        assert "check_overtraining" in tools
        assert "rag_retrieve" in tools

    def test_weekly_training_plan_steps(self) -> None:
        steps = TEMPLATES["weekly_training_plan"]
        tools = [s["tool"] for s in steps]
        assert "get_user_profile" in tools
        assert "get_activities" in tools
        assert "rag_retrieve" in tools

    def test_progress_report_steps(self) -> None:
        steps = TEMPLATES["progress_report"]
        tools = [s["tool"] for s in steps]
        assert "get_activities" in tools
        assert "compute_trend" in tools


class TestTemplatePlanExecutor:
    """Тесты исполнителя шаблонов."""

    @pytest.mark.asyncio
    async def test_unknown_template_returns_empty_result(self) -> None:
        executor = TemplatePlanExecutor()
        result = await executor.execute(
            template_id="nonexistent",
            user_id="u1",
            query_text="test",
            entities={},
            db=_make_mock_db(),
        )
        assert isinstance(result, TemplateResult)
        assert result.template_id == "nonexistent"
        assert len(result.steps) == 0
        assert result.success is False

    @pytest.mark.asyncio
    async def test_recovery_report_executes_steps(self) -> None:
        executor = TemplatePlanExecutor()
        db = _make_mock_db()

        mock_facts = [{"iso_date": "2026-04-10", "hrv_rmssd_milli": 45.0}]
        mock_recovery = {"score": 72, "trend": "stable"}
        mock_rag = [{"text": "sleep is important", "category": "recovery_science", "confidence": "high", "score": 0.9}]

        async def mock_dispatch(tool, args, user_id, query_text, sport_type, db):
            if tool == "get_daily_facts":
                return mock_facts
            if tool == "compute_recovery":
                return mock_recovery
            if tool == "rag_retrieve":
                return mock_rag
            return None

        executor._dispatch = mock_dispatch

        result = await executor.execute(
            template_id="recovery_report",
            user_id="u1",
            query_text="как восстановление?",
            entities={},
            db=db,
        )

        assert result.success is True
        assert len(result.steps) == 3
        assert all(s.success for s in result.steps)
        assert "get_daily_facts" in result.structured_data
        assert "compute_recovery" in result.structured_data

    @pytest.mark.asyncio
    async def test_step_error_does_not_crash_executor(self) -> None:
        executor = TemplatePlanExecutor()
        db = _make_mock_db()

        call_count = 0

        async def mock_dispatch(tool, args, user_id, query_text, sport_type, db):
            nonlocal call_count
            call_count += 1
            if tool == "compute_recovery":
                raise RuntimeError("DB error")
            return [{"data": "ok"}]

        executor._dispatch = mock_dispatch

        result = await executor.execute(
            template_id="recovery_report",
            user_id="u1",
            query_text="тест",
            entities={},
            db=db,
        )

        # Все шаги запущены, один упал с ошибкой
        assert len(result.steps) == 3
        failed_steps = [s for s in result.steps if not s.success]
        assert len(failed_steps) == 1
        assert failed_steps[0].tool == "compute_recovery"

    @pytest.mark.asyncio
    async def test_structured_data_accumulates_rag_chunks(self) -> None:
        executor = TemplatePlanExecutor()
        db = _make_mock_db()

        async def mock_dispatch(tool, args, user_id, query_text, sport_type, db):
            if tool == "get_user_profile":
                return {"name": "Тест"}
            if tool == "get_activities":
                return []
            if tool == "compute_training_load":
                return {"weekly_load": 100}
            if tool == "rag_retrieve":
                category = args.get("category", "")
                return [{"text": f"chunk for {category}", "category": category, "confidence": "high", "score": 0.8}]
            return None

        executor._dispatch = mock_dispatch

        result = await executor.execute(
            template_id="weekly_training_plan",
            user_id="u1",
            query_text="тренировка",
            entities={},
            db=db,
        )

        assert result.success is True
        # Два rag_retrieve шага с разными категориями
        assert any("rag_retrieve_training_principles" in k for k in result.structured_data)
        assert any("rag_retrieve_sport_specific" in k for k in result.structured_data)


class TestTemplateStepResult:
    """Тесты TemplateStepResult."""

    def test_successful_step(self) -> None:
        step = TemplateStepResult(tool="get_activities", args={}, data=[1, 2, 3], success=True)
        assert step.success is True
        assert step.error is None

    def test_failed_step(self) -> None:
        step = TemplateStepResult(tool="compute_recovery", args={}, data=None, success=False, error="DB error")
        assert step.success is False
        assert step.error == "DB error"
