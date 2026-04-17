"""Тесты для модуля маршрутизации Router v2 (Issue #28)."""

import pytest

from app.pipeline.intent_detection import IntentResult
from app.pipeline.router import Router, RouteResult
from app.pipeline.safety_check import SafetyResult


@pytest.fixture
def router() -> Router:
    return Router()


def make_intent(intent: str, confidence: float = 0.9, query: str = "") -> IntentResult:
    return IntentResult(intent=intent, confidence=confidence, entities={}, raw_query=query)


def make_safety(
    safety_level: str = "ok",
    is_safe: bool = True,
    redirect_message: str | None = None,
    warning_suffix: str | None = None,
) -> SafetyResult:
    return SafetyResult(
        is_safe=is_safe,
        safety_level=safety_level,
        redirect_message=redirect_message,
        warning_suffix=warning_suffix,
    )


class TestFastPath:
    """Тесты fast_direct_answer маршрута."""

    def test_direct_question_fast_path(self, router: Router) -> None:
        result = router.route(make_intent("direct_question"), make_safety())
        assert result.route == "fast_direct_answer"
        assert result.fast_path is True
        assert result.blocked is False

    def test_general_chat_fast_path(self, router: Router) -> None:
        result = router.route(make_intent("general_chat"), make_safety())
        assert result.route == "fast_direct_answer"
        assert result.fast_path is True

    def test_off_topic_fast_path(self, router: Router) -> None:
        result = router.route(make_intent("off_topic"), make_safety())
        assert result.route == "fast_direct_answer"
        assert result.fast_path is True

    def test_emergency_fast_path(self, router: Router) -> None:
        # emergency не блокируется safety_level=ok — идёт как fast_direct
        result = router.route(make_intent("emergency"), make_safety())
        assert result.route == "fast_direct_answer"


class TestToolSimple:
    """Тесты tool_simple маршрута."""

    def test_data_retrieval_tool_simple(self, router: Router) -> None:
        result = router.route(make_intent("data_retrieval"), make_safety())
        assert result.route == "tool_simple"
        assert result.fast_path is False
        assert result.tool_calls is not None
        assert "get_activities" in result.tool_calls

    def test_data_analysis_tool_simple_with_modules(self, router: Router) -> None:
        result = router.route(make_intent("data_analysis"), make_safety())
        assert result.route == "tool_simple"
        assert result.modules is not None
        assert "activity_summary" in result.modules
        assert "trend_analyzer" in result.modules


class TestTemplatePlan:
    """Тесты template_plan маршрута."""

    def test_plan_request_weekly_keyword(self, router: Router) -> None:
        result = router.route(make_intent("plan_request", query="составь план на неделю"), make_safety())
        assert result.route == "template_plan"
        assert result.template_id == "weekly_training_plan"
        assert result.fast_path is False

    def test_plan_request_program_keyword(self, router: Router) -> None:
        result = router.route(make_intent("plan_request", query="хочу программу тренировок"), make_safety())
        assert result.route == "template_plan"
        assert result.template_id == "weekly_training_plan"

    def test_plan_request_progress_keyword(self, router: Router) -> None:
        result = router.route(make_intent("plan_request", query="покажи мой прогресс"), make_safety())
        assert result.route == "template_plan"
        assert result.template_id == "progress_report"

    def test_health_concern_overtraining_keyword(self, router: Router) -> None:
        result = router.route(
            make_intent("health_concern", query="есть признаки перетренированности"),
            make_safety(),
        )
        assert result.route == "template_plan"
        assert result.template_id == "overtraining_check"

    def test_health_concern_recovery_keyword(self, router: Router) -> None:
        result = router.route(
            make_intent("health_concern", query="как улучшить восстановление"),
            make_safety(),
        )
        assert result.route == "template_plan"
        assert result.template_id == "recovery_report"

    def test_template_result_has_no_tool_calls(self, router: Router) -> None:
        result = router.route(make_intent("plan_request", query="план на неделю"), make_safety())
        assert result.tool_calls is None
        assert result.modules is None


class TestPlanner:
    """Тесты planner маршрута."""

    def test_plan_request_no_keywords_goes_to_planner(self, router: Router) -> None:
        result = router.route(make_intent("plan_request", query="помоги с тренировками"), make_safety())
        assert result.route == "planner"
        assert result.template_id is None

    def test_health_concern_ambiguous_goes_to_planner(self, router: Router) -> None:
        result = router.route(
            make_intent("health_concern", query="что-то не так с самочувствием"),
            make_safety(),
        )
        assert result.route == "planner"
        assert result.template_id is None


class TestSafetyBlocking:
    """Тесты блокировки по safety."""

    def test_high_priority_blocks(self, router: Router) -> None:
        safety = make_safety(
            safety_level="high_priority",
            is_safe=False,
            redirect_message="Обратитесь к врачу",
        )
        result = router.route(make_intent("direct_question"), safety)
        assert result.blocked is True
        assert result.route == "blocked"
        assert result.block_message == "Обратитесь к врачу"

    def test_high_priority_blocks_any_intent(self, router: Router) -> None:
        safety = make_safety(safety_level="high_priority", is_safe=False)
        for intent in ["data_retrieval", "plan_request", "general_chat", "health_concern"]:
            result = router.route(make_intent(intent), safety)
            assert result.blocked is True, f"intent {intent!r} должен быть заблокирован"

    def test_medium_priority_does_not_block(self, router: Router) -> None:
        safety = make_safety(safety_level="medium_priority", is_safe=True)
        result = router.route(make_intent("health_concern"), safety)
        assert result.blocked is False

    def test_blocked_has_no_tool_calls(self, router: Router) -> None:
        safety = make_safety(safety_level="high_priority", is_safe=False)
        result = router.route(make_intent("data_retrieval"), safety)
        assert result.tool_calls is None
        assert result.modules is None
        assert result.template_id is None


class TestRouteResultFields:
    """Тесты полноты полей RouteResult."""

    def test_result_has_reason(self, router: Router) -> None:
        result = router.route(make_intent("general_chat"), make_safety())
        assert result.reason != ""

    def test_template_plan_has_template_id(self, router: Router) -> None:
        result = router.route(make_intent("plan_request", query="план на неделю"), make_safety())
        assert result.template_id is not None

    def test_planner_no_template_id(self, router: Router) -> None:
        result = router.route(make_intent("plan_request", query="помоги"), make_safety())
        assert result.template_id is None

    def test_fast_path_no_tools_no_modules(self, router: Router) -> None:
        result = router.route(make_intent("general_chat"), make_safety())
        assert result.tool_calls is None
        assert result.modules is None
