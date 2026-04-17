"""Pipeline Orchestrator — сборка всех стадий пайплайна (Phase 2).

Flow:
  User Query
    → Context Builder         — обогащение запроса
    → Intent Detection        — rule-based + LLM fallback
    → Safety Check            — pattern-based
    → Router                  — 4 маршрута
    ↓
    [blocked]       → return redirect_message
    [fast_direct]   → Response Generator → return
    [tool_simple]   → Tool Executor → [Data Processing] → Response Generator → return
    [template_plan] → Template Plan Executor → Response Generator → return
    [planner]       → Planner Agent → Response Generator → return
    ↓
    Memory Update (async, не блокирует delivery)
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat import ChatMessage, ChatSession
from app.pipeline.context_builder import ContextBuilder
from app.pipeline.intent_detection import IntentDetector
from app.pipeline.memory_update import memory_updater
from app.pipeline.planner import planner_agent
from app.pipeline.response_generator import ResponseGenerator
from app.pipeline.router import Router
from app.pipeline.safety_check import SafetyChecker
from app.pipeline.stage_tracker import StageTracker
from app.pipeline.template_plan_executor import template_plan_executor
from app.pipeline.tool_executor import ToolExecutor
from app.services.data_processing.activity_summary import compute_activity_summary
from app.services.data_processing.training_load import compute_training_load
from app.services.data_processing.trend_analyzer import (
    analyze_trend,
    build_time_series_from_activities,
)
from app.services.llm_call_logger import llm_call_logger
from app.services.llm_registry import llm_registry
from app.services.llm_service import ollama_client

logger = logging.getLogger(__name__)

_FALLBACK_RESPONSE = "Извините, произошла ошибка. Попробуйте переформулировать запрос."


@dataclass
class PipelineResult:
    """Результат обработки запроса пайплайном."""

    response_text: str
    intent: str
    route: str
    fast_path: bool
    blocked: bool
    tools_called: list[str]
    modules_used: list[str]
    duration_ms: int
    errors: list[str]

    raw_query: str = ""
    intent_confidence: float = 0.0
    safety_level: str = "ok"
    llm_model: str = ""
    llm_calls_count: int = 0
    template_id: str | None = None

    # Phase 2 (Issue #31): observability
    request_id: str = ""
    stage_trace: list = field(default_factory=list)
    llm_role_usage: dict = field(default_factory=dict)


class PipelineOrchestrator:
    """Оркестратор пайплайна — связывает все модули в единый поток обработки."""

    def __init__(self) -> None:
        self._context_builder = ContextBuilder()
        self._intent_detector = IntentDetector()
        self._safety_checker = SafetyChecker()
        self._router = Router()
        self._tool_executor = ToolExecutor()
        self._response_generator = ResponseGenerator()

    async def process_query(
        self,
        user_id: str,
        session_id: str,
        raw_query: str,
        db: AsyncSession,
        on_token: Callable[[str], None] | None = None,
    ) -> PipelineResult:
        """Обработать запрос пользователя через полный пайплайн.

        Args:
            user_id: Идентификатор пользователя.
            session_id: Идентификатор сессии чата.
            raw_query: Текст запроса пользователя.
            db: Асинхронная сессия SQLAlchemy.
            on_token: Callback для каждого токена при streaming (опционально).

        Returns:
            PipelineResult со всеми метаданными и текстом ответа.
        """
        start_ms = time.monotonic() * 1000
        errors: list[str] = []

        try:
            return await self._run_pipeline(
                user_id=user_id,
                session_id=session_id,
                raw_query=raw_query,
                db=db,
                start_ms=start_ms,
                errors=errors,
                on_token=on_token,
            )
        except Exception as exc:
            logger.error(
                "Orchestrator: необработанная ошибка | user=%s session=%s error=%s",
                user_id, session_id, exc,
                exc_info=True,
            )
            errors.append(str(exc))
            duration = int(time.monotonic() * 1000 - start_ms)
            return PipelineResult(
                response_text=_FALLBACK_RESPONSE,
                intent="unknown",
                route="error",
                fast_path=False,
                blocked=False,
                tools_called=[],
                modules_used=[],
                duration_ms=duration,
                errors=errors,
                raw_query=raw_query,
                intent_confidence=0.0,
                safety_level="unknown",
                llm_model=ollama_client.model,
                llm_calls_count=0,
                request_id=str(uuid.uuid4()),
            )

    async def _run_pipeline(
        self,
        user_id: str,
        session_id: str,
        raw_query: str,
        db: AsyncSession,
        start_ms: float,
        errors: list[str],
        on_token: Callable[[str], None] | None,
    ) -> PipelineResult:
        """Внутренняя реализация пайплайна."""

        # Генерируем request_id в начале, чтобы привязать все LLM-вызовы и трейс
        request_id = str(uuid.uuid4())
        tracker = StageTracker(request_id)
        llm_token = llm_call_logger.start()

        intent_result = None

        try:
            # 1. Context Builder
            async with tracker.track_stage("context_build"):
                enriched = await self._context_builder.build(
                    query=raw_query, session_id=session_id, user_id=user_id, db=db,
                )

            # 2. Intent Detection
            history = (
                [
                    {"role": msg.role, "content": msg.content}
                    for msg in enriched.conversation_history[-3:]
                ]
                if enriched.conversation_history
                else []
            )
            async with tracker.track_stage("intent_stage1"):
                intent_result = await self._intent_detector.detect(
                    raw_query, llm_registry=llm_registry, history=history,
                )

            # 3. Safety Check
            async with tracker.track_stage("safety"):
                safety_result = self._safety_checker.check(raw_query)

            # 4. Router
            async with tracker.track_stage("routing"):
                route_result = self._router.route(intent_result, safety_result)

            logger.info(
                "Orchestrator: intent=%s (%.2f) safety=%s route=%s template=%s "
                "fast=%s blocked=%s | user=%s",
                intent_result.intent, intent_result.confidence,
                safety_result.safety_level, route_result.route,
                route_result.template_id, route_result.fast_path, route_result.blocked,
                user_id,
            )

            # 5. Blocked
            if route_result.blocked:
                response_text = route_result.block_message or _FALLBACK_RESPONSE
                await self._save_messages(session_id, user_id, raw_query, response_text, db)
                llm_calls = llm_call_logger.stop(llm_token)
                await llm_call_logger.flush_to_db(request_id, llm_calls, db)
                duration = int(time.monotonic() * 1000 - start_ms)
                return PipelineResult(
                    response_text=response_text,
                    intent=intent_result.intent,
                    route=route_result.route,
                    fast_path=False,
                    blocked=True,
                    tools_called=[],
                    modules_used=[],
                    duration_ms=duration,
                    errors=errors,
                    raw_query=raw_query,
                    intent_confidence=intent_result.confidence,
                    safety_level=safety_result.safety_level,
                    llm_model=ollama_client.model,
                    llm_calls_count=0,
                    request_id=request_id,
                    stage_trace=tracker.trace,
                    llm_role_usage={},
                )

            llm_model_used = ollama_client.model
            llm_calls_count = 0

            # 6. fast_direct_answer
            if route_result.fast_path:
                try:
                    async with tracker.track_stage("response_gen"):
                        generator_result = await self._response_generator.generate(
                            enriched_query=enriched,
                            route=route_result.route,
                            structured_result=None,
                            safety_level=safety_result.safety_level,
                            intent=intent_result.intent,
                            on_token=on_token,
                        )
                    llm_calls_count = 1
                    response_text = generator_result.content
                    llm_model_used = generator_result.llm_response.model
                except Exception as exc:
                    logger.error(
                        "Orchestrator: ResponseGenerator (fast_path) error: %s", exc,
                        exc_info=True,
                    )
                    errors.append(f"ResponseGenerator: {exc}")
                    response_text = _FALLBACK_RESPONSE

                await self._save_messages(session_id, user_id, raw_query, response_text, db)
                llm_calls = llm_call_logger.stop(llm_token)
                await llm_call_logger.flush_to_db(request_id, llm_calls, db)
                asyncio.create_task(memory_updater.update(
                    user_id=user_id,
                    session_id=session_id,
                    request_id=request_id,
                    query=raw_query,
                    response=response_text,
                    intent=intent_result.intent,
                    entities=intent_result.entities,
                ))
                duration = int(time.monotonic() * 1000 - start_ms)
                return PipelineResult(
                    response_text=response_text,
                    intent=intent_result.intent,
                    route=route_result.route,
                    fast_path=True,
                    blocked=False,
                    tools_called=[],
                    modules_used=[],
                    duration_ms=duration,
                    errors=errors,
                    raw_query=raw_query,
                    intent_confidence=intent_result.confidence,
                    safety_level=safety_result.safety_level,
                    llm_model=llm_model_used,
                    llm_calls_count=llm_calls_count,
                    request_id=request_id,
                    stage_trace=tracker.trace,
                    llm_role_usage=llm_call_logger.build_role_usage(llm_calls),
                )

            tools_called: list[str] = []
            modules_used: list[str] = []
            structured_result: dict = {}

            # 7. tool_simple
            if route_result.route == "tool_simple":
                if route_result.tool_calls:
                    try:
                        async with tracker.track_stage("tool_simple"):
                            tool_result = await self._tool_executor.execute(
                                tool_calls=route_result.tool_calls,
                                user_id=user_id,
                                entities=intent_result.entities,
                                db=db,
                                query_text=raw_query,
                            )
                        tools_called = list(tool_result.results.keys())
                        structured_result["tool_data"] = tool_result.all_data()
                    except Exception as exc:
                        logger.error(
                            "Orchestrator: ToolExecutor error: %s", exc, exc_info=True,
                        )
                        errors.append(f"ToolExecutor: {exc}")

                if route_result.modules:
                    activities = (
                        structured_result.get("tool_data", {}).get("get_activities") or []
                    )
                    processed: dict = {}
                    for module_name in route_result.modules:
                        try:
                            module_output = self._run_data_module(
                                module_name=module_name,
                                activities=activities,
                                entities=intent_result.entities,
                            )
                            processed[module_name] = module_output
                            modules_used.append(module_name)
                        except Exception as exc:
                            logger.error(
                                "Orchestrator: DataProcessing[%s] error: %s",
                                module_name, exc,
                                exc_info=True,
                            )
                            errors.append(f"DataProcessing[{module_name}]: {exc}")
                    if processed:
                        structured_result["processed"] = processed

            # 8. template_plan
            elif route_result.route == "template_plan" and route_result.template_id:
                try:
                    async with tracker.track_stage(
                        "template_plan",
                        metadata={"template_id": route_result.template_id},
                    ):
                        template_result = await template_plan_executor.execute(
                            template_id=route_result.template_id,
                            user_id=user_id,
                            query_text=raw_query,
                            entities=intent_result.entities,
                            db=db,
                        )
                    tools_called = [s.tool for s in template_result.steps if s.success]
                    structured_result = template_result.structured_data
                except Exception as exc:
                    logger.error(
                        "Orchestrator: TemplatePlanExecutor error: %s", exc, exc_info=True,
                    )
                    errors.append(f"TemplatePlanExecutor: {exc}")

            # 9. planner
            elif route_result.route == "planner":
                try:
                    user_context = self._build_user_context(enriched)
                    async with tracker.track_stage("planner"):
                        planner_result = await planner_agent.plan(
                            query=raw_query,
                            user_id=user_id,
                            user_context=user_context,
                            entities=intent_result.entities,
                            db=db,
                        )
                    tools_called = [tc["tool"] for tc in planner_result.tool_calls_history]
                    structured_result = planner_result.tool_results
                    llm_calls_count += planner_result.iterations

                    if planner_result.timeout_hit:
                        errors.append("PlannerAgent: timeout")
                except Exception as exc:
                    logger.error(
                        "Orchestrator: PlannerAgent error: %s", exc, exc_info=True,
                    )
                    errors.append(f"PlannerAgent: {exc}")

            # 10. Response Generator
            try:
                async with tracker.track_stage("response_gen"):
                    generator_result = await self._response_generator.generate(
                        enriched_query=enriched,
                        route=route_result.route,
                        structured_result=structured_result or None,
                        safety_level=safety_result.safety_level,
                        intent=intent_result.intent,
                        on_token=on_token,
                    )
                llm_calls_count += 1
                response_text = generator_result.content
                llm_model_used = generator_result.llm_response.model
            except Exception as exc:
                logger.error(
                    "Orchestrator: ResponseGenerator error: %s", exc, exc_info=True,
                )
                errors.append(f"ResponseGenerator: {exc}")
                response_text = _FALLBACK_RESPONSE

            await self._save_messages(session_id, user_id, raw_query, response_text, db)

            llm_calls = llm_call_logger.stop(llm_token)
            await llm_call_logger.flush_to_db(request_id, llm_calls, db)

            asyncio.create_task(memory_updater.update(
                user_id=user_id,
                session_id=session_id,
                request_id=request_id,
                query=raw_query,
                response=response_text,
                intent=intent_result.intent if intent_result else "",
                entities=intent_result.entities if intent_result else {},
            ))

            duration = int(time.monotonic() * 1000 - start_ms)
            logger.info(
                "Orchestrator: завершено | intent=%s route=%s template=%s tools=%s "
                "duration_ms=%d errors=%d",
                intent_result.intent if intent_result else "unknown",
                route_result.route,
                route_result.template_id,
                tools_called,
                duration,
                len(errors),
            )
            return PipelineResult(
                response_text=response_text,
                intent=intent_result.intent if intent_result else "unknown",
                route=route_result.route,
                fast_path=False,
                blocked=False,
                tools_called=tools_called,
                modules_used=modules_used,
                duration_ms=duration,
                errors=errors,
                raw_query=raw_query,
                intent_confidence=intent_result.confidence if intent_result else 0.0,
                safety_level=safety_result.safety_level,
                llm_model=llm_model_used,
                llm_calls_count=llm_calls_count,
                template_id=route_result.template_id,
                request_id=request_id,
                stage_trace=tracker.trace,
                llm_role_usage=llm_call_logger.build_role_usage(llm_calls),
            )

        except Exception:
            # Если исключение выше не поймано — очищаем LLM трекинг
            try:
                llm_call_logger.stop(llm_token)
            except Exception:
                pass
            raise

    def _build_user_context(self, enriched: object) -> str:
        """Сформировать текстовый контекст пользователя для Planner."""
        from app.pipeline.response_generator import _format_user_profile, _format_conversation_history
        profile_str = _format_user_profile(enriched.user_profile)
        history_str = _format_conversation_history(enriched.conversation_history)
        return f"Профиль:\n{profile_str}\n\nИстория:\n{history_str}"

    def _run_data_module(
        self,
        module_name: str,
        activities: list[dict],
        entities: dict,
    ) -> dict:
        """Запустить модуль обработки данных."""
        if module_name == "activity_summary":
            summary = compute_activity_summary(activities)
            return {
                "total_activities": summary.total_activities,
                "total_duration_minutes": summary.total_duration_minutes,
                "total_calories": summary.total_calories,
                "total_distance_km": summary.total_distance_km,
                "streak_days": summary.streak_days,
                "rest_days": summary.rest_days,
            }

        if module_name == "training_load":
            load = compute_training_load(activities)
            return {
                "weekly_load": load.weekly_load,
                "chronic_load": load.chronic_load,
                "acute_chronic_ratio": load.acute_chronic_ratio,
            }

        if module_name == "trend_analyzer":
            _ENTITY_TO_METRIC: dict[str, str] = {
                "калории": "calories",
                "дистанция": "distance_meters",
            }
            entity_metric = entities.get("metric", "")
            metric = _ENTITY_TO_METRIC.get(entity_metric, "duration_seconds")
            time_series = build_time_series_from_activities(activities, metric)
            trend = analyze_trend(time_series)
            return {
                "direction": trend.direction,
                "change_percent": trend.change_percent,
                "recent_avg": trend.recent_avg,
                "baseline_avg": trend.baseline_avg,
                "metric": metric,
            }

        logger.warning("Orchestrator: неизвестный модуль данных '%s'", module_name)
        return {}

    async def _save_messages(
        self,
        session_id: str,
        user_id: str,
        raw_query: str,
        response_text: str,
        db: AsyncSession,
    ) -> None:
        """Сохранить сообщения пользователя и ассистента в ChatMessage."""
        try:
            stmt = select(ChatSession).where(ChatSession.id == session_id)
            result = await db.execute(stmt)
            session = result.scalar_one_or_none()
            if session is None:
                session = ChatSession(id=session_id, user_id=user_id)
                db.add(session)
                await db.flush()

            max_order_stmt = select(func.max(ChatMessage.order_index)).where(
                ChatMessage.session_id == session_id
            )
            max_result = await db.execute(max_order_stmt)
            max_order: int = max_result.scalar() or 0

            db.add(ChatMessage(
                session_id=session_id, role="user",
                content=raw_query, order_index=max_order + 1,
            ))
            db.add(ChatMessage(
                session_id=session_id, role="assistant",
                content=response_text, order_index=max_order + 2,
            ))
            await db.commit()
        except Exception as exc:
            logger.error(
                "Orchestrator: ошибка сохранения сообщений | session=%s error=%s",
                session_id, exc, exc_info=True,
            )
            await db.rollback()


# Глобальный экземпляр оркестратора (singleton)
pipeline_orchestrator = PipelineOrchestrator()
