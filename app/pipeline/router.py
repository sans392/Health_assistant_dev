"""Модуль маршрутизации запроса (Phase 2, Issue #28).

4 маршрута:
- fast_direct_answer : greeting / off_topic / general_question — без tools
- tool_simple        : data_retrieval / data_analysis — прямые tool-calls
- template_plan      : plan_request / health_concern с ключевыми словами → шаблон
- planner            : сложные plan_request / неоднозначные health_concern → LLM loop
"""

from dataclasses import dataclass

from app.pipeline.intent_detection import IntentResult
from app.pipeline.safety_check import SafetyResult

_BLOCKED_ROUTE = "blocked"

# Ключевые слова для выбора шаблона
_WEEKLY_KEYWORDS: tuple[str, ...] = (
    "неделя", "на неделю", "программа", "расписание", "на следующую неделю",
    "недельный", "на 7 дней",
)
_OVERTRAINING_KEYWORDS: tuple[str, ...] = (
    "перетренированность", "перетренирован", "усталость", "переутомление",
    "перегруз", "перетрениров",
)
_PROGRESS_KEYWORDS: tuple[str, ...] = (
    "прогресс", "мой прогресс", "результаты", "моя динамика", "как я вырос",
    "насколько улучшил",
)
_RECOVERY_KEYWORDS: tuple[str, ...] = (
    "восстановление", "восстановиться", "восстанавливаться", "как восстановить",
    "помоги восстановиться",
)


def _matches(text: str, keywords: tuple[str, ...]) -> bool:
    return any(kw in text for kw in keywords)


@dataclass
class RouteResult:
    """Результат маршрутизации запроса."""

    route: str                          # fast_direct_answer | tool_simple | template_plan | planner | blocked
    fast_path: bool                     # True для fast_direct_answer
    blocked: bool                       # True если safety заблокировал
    block_message: str | None           # Сообщение при блокировке
    tool_calls: list[str] | None        # Инструменты для tool_simple
    modules: list[str] | None          # Модули data processing для tool_simple
    template_id: str | None = None      # ID шаблона для template_plan
    reason: str = ""                    # Причина выбора маршрута (для логов)


class Router:
    """Маршрутизатор — выбирает ветку обработки по intent + keywords."""

    def route(self, intent: IntentResult, safety: SafetyResult) -> RouteResult:
        """Определить маршрут запроса.

        Args:
            intent: Результат определения намерения.
            safety: Результат проверки безопасности.

        Returns:
            RouteResult с выбранным маршрутом.
        """
        if not safety.is_safe or safety.safety_level == "high_priority":
            return RouteResult(
                route=_BLOCKED_ROUTE,
                fast_path=False,
                blocked=True,
                block_message=safety.redirect_message,
                tool_calls=None,
                modules=None,
                template_id=None,
                reason="safety_high_priority",
            )

        query = intent.raw_query.lower()
        intent_name = intent.intent

        if intent_name in ("direct_question", "general_chat", "off_topic", "emergency"):
            return RouteResult(
                route="fast_direct_answer",
                fast_path=True,
                blocked=False,
                block_message=None,
                tool_calls=None,
                modules=None,
                reason=f"fast_direct_{intent_name}",
            )

        if intent_name == "data_retrieval":
            return RouteResult(
                route="tool_simple",
                fast_path=False,
                blocked=False,
                block_message=None,
                tool_calls=["get_activities", "get_daily_facts"],
                modules=None,
                reason="data_retrieval",
            )

        if intent_name == "data_analysis":
            return RouteResult(
                route="tool_simple",
                fast_path=False,
                blocked=False,
                block_message=None,
                tool_calls=["get_activities", "get_daily_facts"],
                modules=["activity_summary", "trend_analyzer"],
                reason="data_analysis",
            )

        if intent_name == "plan_request":
            return self._route_plan_request(query)

        if intent_name == "health_concern":
            return self._route_health_concern(query)

        return RouteResult(
            route="fast_direct_answer",
            fast_path=True,
            blocked=False,
            block_message=None,
            tool_calls=None,
            modules=None,
            reason=f"fallback_{intent_name}",
        )

    def _route_plan_request(self, query: str) -> RouteResult:
        """Выбрать шаблон или planner для plan_request."""
        if _matches(query, _PROGRESS_KEYWORDS):
            return RouteResult(
                route="template_plan",
                fast_path=False,
                blocked=False,
                block_message=None,
                tool_calls=None,
                modules=None,
                template_id="progress_report",
                reason="plan_request_progress",
            )
        if _matches(query, _WEEKLY_KEYWORDS):
            return RouteResult(
                route="template_plan",
                fast_path=False,
                blocked=False,
                block_message=None,
                tool_calls=None,
                modules=None,
                template_id="weekly_training_plan",
                reason="plan_request_weekly",
            )
        return RouteResult(
            route="planner",
            fast_path=False,
            blocked=False,
            block_message=None,
            tool_calls=None,
            modules=None,
            reason="plan_request_complex",
        )

    def _route_health_concern(self, query: str) -> RouteResult:
        """Выбрать шаблон или planner для health_concern."""
        if _matches(query, _OVERTRAINING_KEYWORDS):
            return RouteResult(
                route="template_plan",
                fast_path=False,
                blocked=False,
                block_message=None,
                tool_calls=None,
                modules=None,
                template_id="overtraining_check",
                reason="health_concern_overtraining",
            )
        if _matches(query, _RECOVERY_KEYWORDS):
            return RouteResult(
                route="template_plan",
                fast_path=False,
                blocked=False,
                block_message=None,
                tool_calls=None,
                modules=None,
                template_id="recovery_report",
                reason="health_concern_recovery",
            )
        return RouteResult(
            route="planner",
            fast_path=False,
            blocked=False,
            block_message=None,
            tool_calls=None,
            modules=None,
            reason="health_concern_ambiguous",
        )
