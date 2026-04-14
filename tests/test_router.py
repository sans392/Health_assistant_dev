"""Тесты для модуля маршрутизации (Router)."""

import pytest

from app.pipeline.intent_detection import IntentResult
from app.pipeline.router import Router, RouteResult
from app.pipeline.safety_check import SafetyResult


@pytest.fixture
def router() -> Router:
    return Router()


def make_intent(intent: str, confidence: float = 0.9) -> IntentResult:
    return IntentResult(intent=intent, confidence=confidence, entities={}, raw_query="")


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
    """Тесты fast path маршрутов."""

    def test_direct_question_fast_path(self, router: Router) -> None:
        result = router.route(make_intent("direct_question"), make_safety())
        assert result.route == "fast_direct_answer"
        assert result.fast_path is True
        assert result.blocked is False

    def test_general_chat_fast_path(self, router: Router) -> None:
        result = router.route(make_intent("general_chat"), make_safety())
        assert result.route == "fast_direct_answer"
        assert result.fast_path is True
        assert result.blocked is False


class TestStandardRoutes:
    """Тесты стандартных маршрутов (не fast path)."""

    def test_data_retrieval_route(self, router: Router) -> None:
        result = router.route(make_intent("data_retrieval"), make_safety())
        assert result.route == "tool_simple"
        assert result.fast_path is False
        assert result.blocked is False
        assert result.tool_calls is not None
        assert len(result.tool_calls) > 0

    def test_data_analysis_route(self, router: Router) -> None:
        result = router.route(make_intent("data_analysis"), make_safety())
        assert result.route == "data_analysis_simple"
        assert result.fast_path is False
        assert result.modules is not None
        assert len(result.modules) > 0

    def test_plan_request_route(self, router: Router) -> None:
        result = router.route(make_intent("plan_request"), make_safety())
        assert result.route == "plan_request"
        assert result.fast_path is False

    def test_health_concern_route(self, router: Router) -> None:
        result = router.route(make_intent("health_concern"), make_safety())
        assert result.route == "health_concern"
        assert result.fast_path is False
        assert result.blocked is False


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
        assert result.fast_path is False

    def test_high_priority_blocks_any_intent(self, router: Router) -> None:
        safety = make_safety(
            safety_level="high_priority",
            is_safe=False,
            redirect_message="Срочно к врачу!",
        )
        for intent in ["data_retrieval", "plan_request", "general_chat", "health_concern"]:
            result = router.route(make_intent(intent), safety)
            assert result.blocked is True, f"intent {intent!r} должен быть заблокирован"

    def test_medium_priority_does_not_block(self, router: Router) -> None:
        safety = make_safety(
            safety_level="medium_priority",
            is_safe=True,
            warning_suffix="⚠️ Проконсультируйтесь с врачом",
        )
        result = router.route(make_intent("health_concern"), safety)
        assert result.blocked is False
        assert result.route == "health_concern"

    def test_ok_safety_does_not_block(self, router: Router) -> None:
        result = router.route(make_intent("data_analysis"), make_safety("ok"))
        assert result.blocked is False

    def test_blocked_route_has_no_tool_calls(self, router: Router) -> None:
        safety = make_safety(
            safety_level="high_priority",
            is_safe=False,
            redirect_message="К врачу!",
        )
        result = router.route(make_intent("data_retrieval"), safety)
        assert result.tool_calls is None
        assert result.modules is None


class TestRouteResultFields:
    """Тесты полноты полей RouteResult."""

    def test_fast_path_no_tools(self, router: Router) -> None:
        result = router.route(make_intent("general_chat"), make_safety())
        assert result.tool_calls is None
        assert result.modules is None

    def test_data_analysis_has_tools_and_modules(self, router: Router) -> None:
        result = router.route(make_intent("data_analysis"), make_safety())
        assert result.tool_calls is not None
        assert result.modules is not None

    def test_not_blocked_has_no_block_message(self, router: Router) -> None:
        result = router.route(make_intent("general_chat"), make_safety())
        assert result.block_message is None
