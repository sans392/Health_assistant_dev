"""Диспетчер инструментов (Tool Executor).

Вызывает нужные tools по именам, валидирует аргументы через Pydantic-схемы
(app/tools/schemas.py) и агрегирует результаты.

Границы контракта:
— intent detection отдаёт SlotState (нормализованные слоты);
— ToolExecutor конвертирует слоты в типизированный ToolArgs и валидирует;
— при ValidationError tool получает ToolResult(success=False, error=...)
  вместо тихого падения глубже.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.pipeline.slot_state import SlotState, slot_state_from_entities
from app.services.tool_call_logger import tool_call_logger
from app.tools.db_tools import (
    ToolResult,
    get_activities,
    get_activities_by_sport,
    get_daily_facts,
    get_user_profile,
    log_activity,
    update_profile,
)
from app.tools.rag_retrieve import rag_retrieve
from app.tools.schemas import (
    TOOL_ARGS_REGISTRY,
    CheckOvertrainingArgs,
    ComputeRecoveryArgs,
    ComputeStrainArgs,
    GetActivitiesArgs,
    GetActivitiesBySportArgs,
    GetDailyFactsArgs,
    GetUserProfileArgs,
    LogActivityArgs,
    RagRetrieveArgs,
    UpdateProfileArgs,
    validate_tool_args,
)

logger = logging.getLogger(__name__)


# Дефолтное окно для compute_recovery / check_overtraining, когда пользователь
# не указал диапазон. Исторически было захардкожено в 14 дней.
_DEFAULT_WINDOW_DAYS = 14
_MIN_WINDOW_DAYS = 3
_MAX_WINDOW_DAYS = 60


# Whitelisted-поля args-моделей, которые передаются в db_tools «как есть»
# (числовые/строковые фильтры). sport_type/sport_types/metrics обрабатываются
# отдельно — у них enum-значения нужно развернуть в str.
_ACTIVITY_FILTER_FIELDS: tuple[str, ...] = (
    "min_distance_meters", "max_distance_meters",
    "min_duration_seconds", "max_duration_seconds",
    "min_calories", "max_calories",
    "min_avg_heart_rate", "max_avg_heart_rate",
    "min_avg_speed", "max_avg_speed",
    "min_elevation_meters", "max_elevation_meters",
    "title_contains",
)

_DAILY_FACT_FILTER_FIELDS: tuple[str, ...] = (
    "min_steps", "max_steps",
    "min_calories_kcal", "max_calories_kcal",
    "min_recovery_score", "max_recovery_score",
    "min_hrv_rmssd_milli", "max_hrv_rmssd_milli",
    "min_resting_heart_rate", "max_resting_heart_rate",
    "min_sleep_total_in_bed_milli", "max_sleep_total_in_bed_milli",
    "min_water_liters", "max_water_liters",
    "min_spo2_percentage", "max_spo2_percentage",
)


def _activities_kwargs(args: "GetActivitiesArgs") -> dict[str, Any]:
    """Развернуть GetActivitiesArgs в kwargs для get_activities.

    Передаёт только non-None фильтры — иначе тесты с узкой mock-сигнатурой
    падают на лишних kwargs.
    """
    kwargs: dict[str, Any] = {
        "user_id": args.user_id,
        "date_from": args.date_from,
        "date_to": args.date_to,
    }
    if args.sport_type is not None:
        kwargs["sport_type"] = args.sport_type.value
    if args.sport_types:
        kwargs["sport_types"] = [s.value for s in args.sport_types]
    for f in _ACTIVITY_FILTER_FIELDS:
        v = getattr(args, f, None)
        if v is not None:
            kwargs[f] = v
    return kwargs


def _daily_facts_kwargs(args: "GetDailyFactsArgs") -> dict[str, Any]:
    """Развернуть GetDailyFactsArgs в kwargs для get_daily_facts."""
    kwargs: dict[str, Any] = {
        "user_id": args.user_id,
        "date_from": args.date_from,
        "date_to": args.date_to,
    }
    if args.metrics:
        kwargs["metrics"] = [m.value for m in args.metrics]
    for f in _DAILY_FACT_FILTER_FIELDS:
        v = getattr(args, f, None)
        if v is not None:
            kwargs[f] = v
    return kwargs


@dataclass
class ToolExecutorResult:
    """Агрегированный результат работы ToolExecutor."""

    results: dict[str, ToolResult] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """True если хотя бы один tool выполнился успешно."""
        return any(r.success for r in self.results.values())

    def get_data(self, tool_name: str) -> Any:
        """Получить данные конкретного tool. None если не найден или ошибка."""
        result = self.results.get(tool_name)
        if result and result.success:
            return result.data
        return None

    def all_data(self) -> dict[str, Any]:
        """Все успешные данные: tool_name → data."""
        return {name: r.data for name, r in self.results.items() if r.success}


class ToolExecutor:
    """Диспетчер инструментов — вызывает tools по именам из RouteResult.

    Аргументы для каждого tool собираются из SlotState (или legacy entities
    dict) и валидируются через Pydantic перед вызовом.
    """

    async def execute(
        self,
        tool_calls: list[str],
        user_id: str,
        entities: dict,
        db: AsyncSession,
        query_text: str | None = None,
        slots: SlotState | None = None,
    ) -> ToolExecutorResult:
        """Выполнить список tool-вызовов.

        Args:
            tool_calls: Список имён tools (из RouteResult.tool_calls).
            user_id: Идентификатор пользователя.
            entities: Legacy entities dict (сохранён для совместимости).
            db: Асинхронная сессия SQLAlchemy.
            query_text: Исходный текст запроса (нужен для rag_retrieve).
            slots: SlotState — если передан, используется вместо entities.
                  Иначе слоты строятся из entities (back-compat).

        Returns:
            ToolExecutorResult с результатами всех вызовов.
        """
        if slots is None:
            slots = slot_state_from_entities(entities, raw_query=query_text or "")

        logger.info(
            "ToolExecutor: вызываем %s | user=%s time_range=%s sport=%s",
            tool_calls,
            user_id,
            slots.time_range.label if slots.time_range else None,
            slots.sport_type.value if slots.sport_type else None,
        )

        executor_result = ToolExecutorResult()

        for requested_tool in tool_calls:
            # Fallback: get_activities_by_sport без sport_type → get_activities
            # (сохраняем поведение до Phase 2, пока clarification loop не готов).
            tool_name = requested_tool
            if tool_name == "get_activities_by_sport" and slots.sport_type is None:
                logger.warning(
                    "ToolExecutor: get_activities_by_sport без sport_type, "
                    "делаем fallback на get_activities"
                )
                tool_name = "get_activities"

            args_model, args_snapshot, validation_error = self._build_args(
                tool_name=tool_name,
                user_id=user_id,
                slots=slots,
                query_text=query_text,
            )

            start_ms = time.monotonic() * 1000
            if validation_error is not None:
                logger.warning(
                    "ToolExecutor: args для %s не прошли валидацию: %s",
                    tool_name, validation_error,
                )
                result = ToolResult(
                    tool_name=tool_name,
                    success=False,
                    data=None,
                    error=f"Некорректные аргументы: {validation_error}",
                )
            else:
                result = await self._dispatch(
                    tool_name=tool_name,
                    args=args_model,
                    db=db,
                )
            duration_ms = int(time.monotonic() * 1000 - start_ms)
            tool_call_logger.record(
                name=tool_name,
                source="tool_executor",
                args=args_snapshot,
                result=result.data if result.success else None,
                success=result.success,
                error=result.error,
                duration_ms=duration_ms,
            )
            # Ключом в results остаётся имя tool'а, которое запросил роутер —
            # иначе потребитель (response_generator) не найдёт свой tool.
            executor_result.results[requested_tool] = result

        return executor_result

    # -----------------------------------------------------------------
    # Построение и валидация аргументов
    # -----------------------------------------------------------------

    @staticmethod
    def _build_args(
        tool_name: str,
        user_id: str,
        slots: SlotState,
        query_text: str | None,
    ) -> tuple[Any, dict[str, Any], str | None]:
        """Собирает raw-args по имени tool'а и валидирует через Pydantic.

        Returns:
            (args_model, args_snapshot, error)
            — args_model: валидированный Pydantic-объект или None при ошибке
            — args_snapshot: dict для логирования в tool_calls.args
            — error: текст ошибки валидации или None
        """
        if tool_name not in TOOL_ARGS_REGISTRY:
            return (
                None,
                {"user_id": user_id},
                f"Неизвестный tool: {tool_name}",
            )

        raw_args = ToolExecutor._collect_raw_args(
            tool_name=tool_name,
            user_id=user_id,
            slots=slots,
            query_text=query_text,
        )

        try:
            args_model = validate_tool_args(tool_name, raw_args)
        except ValidationError as exc:
            # Упрощаем сообщение для логов: берём первую ошибку.
            first = exc.errors()[0] if exc.errors() else {}
            msg = f"{first.get('loc', ('?',))[-1]}: {first.get('msg', str(exc))}"
            return None, raw_args, msg

        return args_model, args_model.model_dump(mode="json"), None

    @staticmethod
    def _collect_raw_args(
        tool_name: str,
        user_id: str,
        slots: SlotState,
        query_text: str | None,
    ) -> dict[str, Any]:
        """Собрать dict аргументов из SlotState для конкретного tool'а."""
        tr = slots.time_range
        # Дефолт: если time_range не указан, используем последние 7 дней.
        # Логируем явно — иначе случай «intent не распарсил дату» молча
        # маскируется и пользователь получает не тот период.
        if tr is not None:
            date_from, date_to = tr.date_from, tr.date_to
        else:
            today = date.today()
            date_from = today - timedelta(days=6)
            date_to = today
            if tool_name in {"get_activities", "get_daily_facts", "get_activities_by_sport"}:
                logger.warning(
                    "ToolExecutor: time_range не определён для %s, "
                    "используем дефолт last-7-days (%s..%s) | query=%r",
                    tool_name, date_from, date_to,
                    (query_text or "")[:120],
                )

        sport = slots.sport_type.value if slots.sport_type else None

        if tool_name == "get_activities":
            return {
                "user_id": user_id,
                "date_from": date_from,
                "date_to": date_to,
                "sport_type": sport,
            }

        if tool_name == "get_activities_by_sport":
            # Если slot пустой — fallback на get_activities обрабатывается
            # в dispatch. Здесь собираем args как есть; если sport нет,
            # валидация упадёт и dispatch сделает fallback.
            args: dict[str, Any] = {
                "user_id": user_id,
                "sport_type": sport or "",  # заведомо невалидный — триггер fallback
                "days": (tr.days if tr is not None else 30),
            }
            return args

        if tool_name == "get_daily_facts":
            metrics = [m.value for m in slots.metrics] or None
            return {
                "user_id": user_id,
                "date_from": date_from,
                "date_to": date_to,
                "metrics": metrics,
            }

        if tool_name == "get_user_profile":
            return {"user_id": user_id}

        if tool_name == "rag_retrieve":
            return {
                "query_text": query_text or "",
                "sport_type": sport,
                "top_k": 5,
            }

        if tool_name in ("compute_recovery", "check_overtraining"):
            # Если пользователь явно указал time_range — используем его длину
            # вместо захардкоженных 14 дней. Клипуем в допустимый диапазон
            # (меньше 3 дней бессмысленно, больше 60 — слишком шумно).
            if tr is not None:
                window = max(_MIN_WINDOW_DAYS, min(_MAX_WINDOW_DAYS, tr.days))
            else:
                window = _DEFAULT_WINDOW_DAYS
            return {"user_id": user_id, "window_days": window}

        if tool_name == "compute_strain":
            return {"user_id": user_id, "reference_date": date_to}

        if tool_name == "log_activity":
            return {
                "user_id": user_id,
                "sport_type": sport or "",
                "duration": 0,
            }

        if tool_name == "update_profile":
            return {"user_id": user_id, "field": "name", "value": None}

        return {"user_id": user_id}

    # -----------------------------------------------------------------
    # Диспетчеризация
    # -----------------------------------------------------------------

    async def _dispatch(
        self,
        tool_name: str,
        args: Any,
        db: AsyncSession,
    ) -> ToolResult:
        """Вызвать конкретный tool по имени с валидированными args."""
        if isinstance(args, GetActivitiesArgs):
            kwargs = _activities_kwargs(args)
            return await get_activities(db=db, **kwargs)

        if isinstance(args, GetActivitiesBySportArgs):
            return await get_activities_by_sport(
                db=db,
                user_id=args.user_id,
                sport_type=args.sport_type.value,
                days=args.days,
                limit=args.limit,
            )

        if isinstance(args, GetDailyFactsArgs):
            kwargs = _daily_facts_kwargs(args)
            return await get_daily_facts(db=db, **kwargs)

        if isinstance(args, GetUserProfileArgs):
            return await get_user_profile(db=db, user_id=args.user_id)

        if isinstance(args, RagRetrieveArgs):
            return await rag_retrieve(
                query=args.query_text,
                category=args.category,
                sport_type=args.sport_type.value if args.sport_type else None,
                top_k=args.top_k,
            )

        if isinstance(args, ComputeRecoveryArgs):
            return await self._compute_recovery(
                user_id=args.user_id, db=db, window_days=args.window_days
            )

        if isinstance(args, ComputeStrainArgs):
            return await self._compute_strain(
                user_id=args.user_id, db=db, reference_date=args.reference_date
            )

        if isinstance(args, CheckOvertrainingArgs):
            return await self._check_overtraining(
                user_id=args.user_id, db=db, window_days=args.window_days
            )

        logger.warning("ToolExecutor: неизвестный tool '%s'", tool_name)
        return ToolResult(
            tool_name=tool_name,
            success=False,
            data=None,
            error=f"Неизвестный tool: {tool_name}",
        )

    # -----------------------------------------------------------------
    # Aggregator tools (compute_*)
    # -----------------------------------------------------------------

    async def _compute_recovery(
        self, user_id: str, db: AsyncSession, window_days: int = _DEFAULT_WINDOW_DAYS
    ) -> ToolResult:
        """Рассчитать recovery score за заданное окно."""
        import dataclasses
        from app.services.data_processing.recovery_score import compute_recovery_score

        today = date.today()
        facts_res = await get_daily_facts(
            db=db, user_id=user_id,
            date_from=today - timedelta(days=window_days - 1),
            date_to=today,
        )
        # Для активностей берём окно x2 — чтобы посчитать trailing load.
        acts_res = await get_activities(
            db=db, user_id=user_id,
            date_from=today - timedelta(days=window_days * 2 - 1),
            date_to=today,
        )
        result = compute_recovery_score(
            daily_facts=facts_res.data or [],
            activities=acts_res.data or [],
        )
        return ToolResult(
            tool_name="compute_recovery",
            success=True,
            data=dataclasses.asdict(result),
        )

    async def _compute_strain(
        self, user_id: str, db: AsyncSession, reference_date: date
    ) -> ToolResult:
        """Рассчитать Strain Score за указанный день."""
        import dataclasses
        from app.services.data_processing.strain_score import compute_strain_score

        acts_res = await get_activities(
            db=db, user_id=user_id,
            date_from=reference_date,
            date_to=reference_date,
        )
        result = compute_strain_score(
            activities=acts_res.data or [],
            reference_date=reference_date,
        )
        return ToolResult(
            tool_name="compute_strain",
            success=True,
            data=dataclasses.asdict(result),
        )

    async def _check_overtraining(
        self, user_id: str, db: AsyncSession, window_days: int = _DEFAULT_WINDOW_DAYS
    ) -> ToolResult:
        """Проверить маркеры перетренированности за заданное окно."""
        import dataclasses
        from app.services.data_processing.overtraining_detection import detect_overtraining

        today = date.today()
        facts_res = await get_daily_facts(
            db=db, user_id=user_id,
            date_from=today - timedelta(days=window_days - 1),
            date_to=today,
        )
        acts_res = await get_activities(
            db=db, user_id=user_id,
            date_from=today - timedelta(days=window_days * 2 - 1),
            date_to=today,
        )
        result = detect_overtraining(
            daily_facts=facts_res.data or [],
            activities=acts_res.data or [],
        )
        return ToolResult(
            tool_name="check_overtraining",
            success=True,
            data=dataclasses.asdict(result),
        )

    # -----------------------------------------------------------------
    # Write-actions
    # -----------------------------------------------------------------

    async def execute_action(
        self,
        tool_name: str,
        user_id: str,
        params: dict,
        db: AsyncSession,
    ) -> ToolResult:
        """Вызвать write-инструмент (log_activity, update_profile).

        params валидируются через Pydantic-модель соответствующего tool'а.
        """
        start_ms = time.monotonic() * 1000
        raw_args: dict[str, Any] = {"user_id": user_id, **dict(params or {})}

        try:
            args_model = validate_tool_args(tool_name, raw_args)
        except KeyError:
            logger.warning("ToolExecutor.execute_action: неизвестный tool '%s'", tool_name)
            result = ToolResult(
                tool_name=tool_name,
                success=False,
                data=None,
                error=f"Неизвестный write-tool: {tool_name}",
            )
            duration_ms = int(time.monotonic() * 1000 - start_ms)
            tool_call_logger.record(
                name=tool_name, source="tool_executor", args=raw_args,
                result=None, success=False, error=result.error,
                duration_ms=duration_ms,
            )
            return result
        except ValidationError as exc:
            first = exc.errors()[0] if exc.errors() else {}
            err = f"{first.get('loc', ('?',))[-1]}: {first.get('msg', str(exc))}"
            result = ToolResult(
                tool_name=tool_name, success=False, data=None,
                error=f"Некорректные аргументы: {err}",
            )
            duration_ms = int(time.monotonic() * 1000 - start_ms)
            tool_call_logger.record(
                name=tool_name, source="tool_executor", args=raw_args,
                result=None, success=False, error=result.error,
                duration_ms=duration_ms,
            )
            return result

        if isinstance(args_model, LogActivityArgs):
            result = await log_activity(
                db=db,
                user_id=args_model.user_id,
                sport_type=args_model.sport_type.value,
                duration=args_model.duration,
                calories=args_model.calories,
                distance=args_model.distance,
                notes=args_model.notes,
            )
        elif isinstance(args_model, UpdateProfileArgs):
            result = await update_profile(
                db=db,
                user_id=args_model.user_id,
                field=args_model.field,
                value=args_model.value,
            )
        else:
            logger.warning(
                "ToolExecutor.execute_action: tool '%s' не является write-action",
                tool_name,
            )
            result = ToolResult(
                tool_name=tool_name,
                success=False,
                data=None,
                error=f"Tool {tool_name} не является write-action",
            )

        duration_ms = int(time.monotonic() * 1000 - start_ms)
        tool_call_logger.record(
            name=tool_name,
            source="tool_executor",
            args=args_model.model_dump(mode="json"),
            result=result.data if result.success else None,
            success=result.success,
            error=result.error,
            duration_ms=duration_ms,
        )
        return result
