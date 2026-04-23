"""Модуль маршрутизации запроса (Phase 2, Issue #28).

4 маршрута:
- fast_direct_answer : greeting / off_topic / capability_question — без tools
- tool_simple        : data_query / reference_question — прямые tool-calls
- template_plan      : plan_request / health_concern с ключевыми словами → шаблон
- planner            : сложные plan_request / неоднозначные health_concern,
                       сложные data_query (breakdown / compare) → LLM loop
"""

from dataclasses import dataclass

from app.pipeline.capability_answer import build_capability_answer
from app.pipeline.intent_detection import IntentResult
from app.pipeline.safety_check import SafetyResult
from app.tools.schemas import AnalysisType

_BLOCKED_ROUTE = "blocked"

# Ключевые слова для выбора шаблона
_WEEKLY_KEYWORDS: tuple[str, ...] = (
    "недел", "на неделю", "программ", "расписани", "на следующую неделю",
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

# Метрики, для ответа по которым нужны динамические данные из БД
# (daily_facts / activities). Статические поля профиля (вес, рост) сюда
# не входят — они уже попадают в fast-path через system prompt.
_DYNAMIC_METRICS: frozenset[str] = frozenset({
    "heart_rate", "hrv", "шаги", "калории", "сон",
    "recovery", "strain", "дистанция", "время", "темп",
    "cadence", "rpe",
})


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
    # Статический ответ (минует LLM). Используется для capability_question,
    # где ответ детерминирован и не требует генерации. Orchestrator должен
    # проверять это поле раньше вызова ResponseGenerator.
    static_response: str | None = None


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
        entities = intent.entities or {}

        if intent_name == "direct_question":
            # Если спрашивают про динамическую метрику (шаги/HRV/пульс/…) —
            # без реальных данных из БД ответ будет галлюцинацией. Роутим
            # в tool_simple, чтобы подтянуть get_activities + get_daily_facts.
            metric = entities.get("metric")
            if metric in _DYNAMIC_METRICS:
                return RouteResult(
                    route="tool_simple",
                    fast_path=False,
                    blocked=False,
                    block_message=None,
                    tool_calls=["get_activities", "get_daily_facts"],
                    modules=None,
                    reason="direct_question_dynamic_metric",
                )
            return RouteResult(
                route="fast_direct_answer",
                fast_path=True,
                blocked=False,
                block_message=None,
                tool_calls=None,
                modules=None,
                reason="fast_direct_direct_question",
            )

        if intent_name in ("general_chat", "off_topic", "emergency"):
            return RouteResult(
                route="fast_direct_answer",
                fast_path=True,
                blocked=False,
                block_message=None,
                tool_calls=None,
                modules=None,
                reason=f"fast_direct_{intent_name}",
            )

        if intent_name == "capability_question":
            # Мета-вопрос «что ты умеешь» — статический ответ, без LLM.
            return RouteResult(
                route="fast_direct_answer",
                fast_path=True,
                blocked=False,
                block_message=None,
                tool_calls=None,
                modules=None,
                reason="capability_question_static",
                static_response=build_capability_answer(),
            )

        if intent_name == "reference_question":
            # FAQ — берём знания из RAG, без тяжёлых DB-тулов.
            return RouteResult(
                route="tool_simple",
                fast_path=False,
                blocked=False,
                block_message=None,
                tool_calls=["rag_retrieve"],
                modules=None,
                reason="reference_question",
            )

        if intent_name == "data_query":
            return self._route_data_query(intent)

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

    def _route_data_query(self, intent: IntentResult) -> RouteResult:
        """Маршрутизировать data_query по analysis_type.

        NONE / NORM_CHECK / TREND → tool_simple (retrieval ± лёгкие модули).
        BREAKDOWN / COMPARE       → planner (сложный многошаговый анализ).
        """
        analysis_type = intent.slots.analysis_type

        if analysis_type in (AnalysisType.BREAKDOWN, AnalysisType.COMPARE):
            return RouteResult(
                route="planner",
                fast_path=False,
                blocked=False,
                block_message=None,
                tool_calls=None,
                modules=None,
                reason=f"data_query_{analysis_type.value}",
            )

        modules: list[str] | None = None
        if analysis_type in (AnalysisType.NORM_CHECK, AnalysisType.TREND):
            modules = ["activity_summary", "trend_analyzer"]

        return RouteResult(
            route="tool_simple",
            fast_path=False,
            blocked=False,
            block_message=None,
            tool_calls=["get_activities", "get_daily_facts"],
            modules=modules,
            reason=f"data_query_{analysis_type.value}",
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
