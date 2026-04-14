"""Модуль маршрутизации запроса в нужную ветку обработки."""

from dataclasses import dataclass

from app.pipeline.intent_detection import IntentResult
from app.pipeline.safety_check import SafetyResult


@dataclass
class RouteResult:
    """Результат маршрутизации запроса."""

    route: str                    # название маршрута
    fast_path: bool               # True если fast_path (без Tool Executor и Data Processing)
    blocked: bool                 # True если safety заблокировал запрос
    block_message: str | None     # Сообщение при блокировке
    tool_calls: list[str] | None  # Какие tools нужно вызвать
    modules: list[str] | None     # Какие модули data processing нужны


# Маршрут при блокировке safety
_BLOCKED_ROUTE = "blocked"

# Маппинг intent → (route, fast_path, tool_calls, modules)
_INTENT_ROUTES: dict[str, tuple[str, bool, list[str] | None, list[str] | None]] = {
    "direct_question": ("fast_direct_answer", True, None, None),
    "general_chat": ("fast_direct_answer", True, None, None),
    "data_retrieval": ("tool_simple", False, ["get_activities", "get_daily_facts"], None),
    "data_analysis": (
        "data_analysis_simple",
        False,
        ["get_activities", "get_daily_facts"],
        ["activity_summary", "trend_analyzer"],
    ),
    "plan_request": (
        "plan_request",
        False,
        ["get_activities", "get_user_profile"],
        ["training_load"],
    ),
    "health_concern": ("health_concern", False, None, None),
}


class Router:
    """Маршрутизатор запросов пайплайна."""

    def route(self, intent: IntentResult, safety: SafetyResult) -> RouteResult:
        """Определяет маршрут обработки запроса.

        Args:
            intent: Результат определения намерения.
            safety: Результат проверки безопасности.

        Returns:
            RouteResult с выбранным маршрутом.
        """
        # Safety high_priority блокирует выполнение пайплайна
        if not safety.is_safe or safety.safety_level == "high_priority":
            return RouteResult(
                route=_BLOCKED_ROUTE,
                fast_path=False,
                blocked=True,
                block_message=safety.redirect_message,
                tool_calls=None,
                modules=None,
            )

        # health_concern с high_priority тоже блокируем (дополнительная защита)
        if intent.intent == "health_concern" and safety.safety_level == "high_priority":
            return RouteResult(
                route=_BLOCKED_ROUTE,
                fast_path=False,
                blocked=True,
                block_message=safety.redirect_message,
                tool_calls=None,
                modules=None,
            )

        # Выбираем маршрут по intent
        route_name, fast_path, tool_calls, modules = _INTENT_ROUTES.get(
            intent.intent,
            # Fallback: general_chat → fast path
            ("fast_direct_answer", True, None, None),
        )

        return RouteResult(
            route=route_name,
            fast_path=fast_path,
            blocked=False,
            block_message=None,
            tool_calls=tool_calls,
            modules=modules,
        )
