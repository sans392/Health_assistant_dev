"""Pipeline Orchestrator — сборка всех стадий пайплайна.

Связывает Context Builder, Intent Detection, Safety Check, Router,
Tool Executor, Data Processing и Response Generator в единый поток.
"""

import logging
import time
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat import ChatMessage, ChatSession
from app.pipeline.context_builder import ContextBuilder
from app.pipeline.intent_detection import IntentDetector
from app.pipeline.memory_update import memory_updater
from app.pipeline.response_generator import ResponseGenerator
from app.pipeline.router import Router
from app.pipeline.safety_check import SafetyChecker
from app.pipeline.tool_executor import ToolExecutor
from app.services.data_processing.activity_summary import compute_activity_summary
from app.services.data_processing.training_load import compute_training_load
from app.services.data_processing.trend_analyzer import (
    analyze_trend,
    build_time_series_from_activities,
)
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

    # Дополнительные метаданные для логирования (issue #12)
    raw_query: str = ""
    intent_confidence: float = 0.0
    safety_level: str = "ok"
    llm_model: str = ""
    llm_calls_count: int = 0


class PipelineOrchestrator:
    """Оркестратор пайплайна — связывает все модули в единый поток обработки.

    Полный flow:
        User Query
          → Context Builder         — обогащение запроса
          → Intent Detection        — определение намерения
          → Safety Check            — проверка безопасности
          → Router                  — выбор маршрута
          ↓
          [blocked] → return redirect_message
          ↓
          [fast_path] → Response Generator → return
          ↓
          [standard]
            → Tool Executor         — получение данных из БД
            → Data Processing       — вычисления
            → Response Generator    — генерация ответа LLM
            → return
    """

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
    ) -> PipelineResult:
        """Обработать запрос пользователя через полный пайплайн.

        Args:
            user_id: Идентификатор пользователя.
            session_id: Идентификатор сессии чата.
            raw_query: Текст запроса пользователя.
            db: Асинхронная сессия SQLAlchemy.

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
            )

    async def _run_pipeline(
        self,
        user_id: str,
        session_id: str,
        raw_query: str,
        db: AsyncSession,
        start_ms: float,
        errors: list[str],
    ) -> PipelineResult:
        """Внутренняя реализация пайплайна без верхнеуровневого try/except."""

        # 1. Context Builder — обогащение запроса контекстом
        enriched = await self._context_builder.build(
            query=raw_query,
            session_id=session_id,
            user_id=user_id,
            db=db,
        )

        # 2. Intent Detection — stage 1 (rule-based) + stage 2 LLM при низкой уверенности
        history = [
            {"role": msg.role, "content": msg.content}
            for msg in enriched.conversation_history[-3:]
        ] if enriched.conversation_history else []
        intent_result = await self._intent_detector.detect(
            raw_query,
            llm_registry=llm_registry,
            history=history,
        )

        # 3. Safety Check — проверяем безопасность (синхронный)
        safety_result = self._safety_checker.check(raw_query)

        # 4. Router — выбираем маршрут
        route_result = self._router.route(intent_result, safety_result)

        logger.info(
            "Orchestrator: intent=%s (%.2f) safety=%s route=%s fast_path=%s blocked=%s | user=%s",
            intent_result.intent,
            intent_result.confidence,
            safety_result.safety_level,
            route_result.route,
            route_result.fast_path,
            route_result.blocked,
            user_id,
        )

        # 5. [Blocked] — safety заблокировал запрос
        if route_result.blocked:
            response_text = route_result.block_message or _FALLBACK_RESPONSE
            await self._save_messages(
                session_id=session_id,
                user_id=user_id,
                raw_query=raw_query,
                response_text=response_text,
                db=db,
            )
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
            )

        llm_model_used = ollama_client.model
        llm_calls_count = 0

        # 6. Fast path — без Tool Executor и Data Processing
        if route_result.fast_path:
            try:
                generator_result = await self._response_generator.generate(
                    enriched_query=enriched,
                    route=route_result.route,
                    structured_result=None,
                    safety_level=safety_result.safety_level,
                )
                llm_calls_count = 1
                response_text = generator_result.content
                llm_model_used = generator_result.llm_response.model
            except Exception as exc:
                logger.error("Orchestrator: ResponseGenerator (fast_path) error: %s", exc, exc_info=True)
                errors.append(f"ResponseGenerator: {exc}")
                response_text = _FALLBACK_RESPONSE

            await self._save_messages(
                session_id=session_id,
                user_id=user_id,
                raw_query=raw_query,
                response_text=response_text,
                db=db,
            )
            await memory_updater.update(
                user_id=user_id,
                request_id=None,
                query=raw_query,
                response=response_text,
            )
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
            )

        # 7. Standard path — Tool Executor → Data Processing → Response Generator
        tools_called: list[str] = []
        modules_used: list[str] = []
        structured_result: dict = {}

        # Tool Executor — получаем данные из БД
        if route_result.tool_calls:
            try:
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
                logger.error("Orchestrator: ToolExecutor error: %s", exc, exc_info=True)
                errors.append(f"ToolExecutor: {exc}")

        # Data Processing — вычисления на полученных данных
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
                        "Orchestrator: DataProcessing[%s] error: %s", module_name, exc,
                        exc_info=True,
                    )
                    errors.append(f"DataProcessing[{module_name}]: {exc}")
            if processed:
                structured_result["processed"] = processed

        # Response Generator — генерируем ответ через LLM
        try:
            generator_result = await self._response_generator.generate(
                enriched_query=enriched,
                route=route_result.route,
                structured_result=structured_result or None,
                safety_level=safety_result.safety_level,
            )
            llm_calls_count = 1
            response_text = generator_result.content
            llm_model_used = generator_result.llm_response.model
        except Exception as exc:
            logger.error("Orchestrator: ResponseGenerator error: %s", exc, exc_info=True)
            errors.append(f"ResponseGenerator: {exc}")
            response_text = _FALLBACK_RESPONSE

        await self._save_messages(
            session_id=session_id,
            user_id=user_id,
            raw_query=raw_query,
            response_text=response_text,
            db=db,
        )
        await memory_updater.update(
            user_id=user_id,
            request_id=None,
            query=raw_query,
            response=response_text,
        )

        duration = int(time.monotonic() * 1000 - start_ms)
        logger.info(
            "Orchestrator: завершено | intent=%s route=%s tools=%s modules=%s "
            "duration_ms=%d errors=%d",
            intent_result.intent, route_result.route,
            tools_called, modules_used,
            duration, len(errors),
        )
        return PipelineResult(
            response_text=response_text,
            intent=intent_result.intent,
            route=route_result.route,
            fast_path=False,
            blocked=False,
            tools_called=tools_called,
            modules_used=modules_used,
            duration_ms=duration,
            errors=errors,
            raw_query=raw_query,
            intent_confidence=intent_result.confidence,
            safety_level=safety_result.safety_level,
            llm_model=llm_model_used,
            llm_calls_count=llm_calls_count,
        )

    def _run_data_module(
        self,
        module_name: str,
        activities: list[dict],
        entities: dict,
    ) -> dict:
        """Запустить модуль обработки данных и вернуть результат как dict.

        Args:
            module_name: Имя модуля (activity_summary, training_load, trend_analyzer).
            activities: Список активностей из ToolExecutor.
            entities: Извлечённые сущности из IntentResult.

        Returns:
            Словарь с результатами модуля для передачи в ResponseGenerator.
        """
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
            # Выбираем метрику по entity, если есть
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
        """Сохранить сообщения пользователя и ассистента в ChatMessage.

        Args:
            session_id: Идентификатор сессии.
            user_id: Идентификатор пользователя (для создания сессии, если нет).
            raw_query: Текст запроса пользователя.
            response_text: Текст ответа ассистента.
            db: Асинхронная сессия SQLAlchemy.
        """
        try:
            # Проверяем наличие сессии, создаём если нет
            stmt = select(ChatSession).where(ChatSession.id == session_id)
            result = await db.execute(stmt)
            session = result.scalar_one_or_none()
            if session is None:
                session = ChatSession(id=session_id, user_id=user_id)
                db.add(session)
                await db.flush()

            # Следующий порядковый индекс
            max_order_stmt = select(func.max(ChatMessage.order_index)).where(
                ChatMessage.session_id == session_id
            )
            max_result = await db.execute(max_order_stmt)
            max_order: int = max_result.scalar() or 0

            db.add(ChatMessage(
                session_id=session_id,
                role="user",
                content=raw_query,
                order_index=max_order + 1,
            ))
            db.add(ChatMessage(
                session_id=session_id,
                role="assistant",
                content=response_text,
                order_index=max_order + 2,
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
