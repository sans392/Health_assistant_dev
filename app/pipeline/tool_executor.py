"""Диспетчер инструментов (Tool Executor).

Вызывает нужные tools по именам, разрешает параметры из сущностей запроса,
агрегирует результаты.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

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
from app.tools.time_utils import resolve_time_range

logger = logging.getLogger(__name__)


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

    Параметры для каждого tool (date_from, date_to, sport_type) разрешаются
    из entities, извлечённых Intent Detection'ом.
    """

    async def execute(
        self,
        tool_calls: list[str],
        user_id: str,
        entities: dict,
        db: AsyncSession,
        query_text: str | None = None,
    ) -> ToolExecutorResult:
        """Выполнить список tool-вызовов.

        Args:
            tool_calls: Список имён tools (из RouteResult.tool_calls).
            user_id: Идентификатор пользователя.
            entities: Сущности из IntentResult (time_range, sport_type и т.д.).
            db: Асинхронная сессия SQLAlchemy.

        Returns:
            ToolExecutorResult с результатами всех вызовов.
        """
        # Разрешаем time_range → конкретные даты
        time_range_entity = entities.get("time_range")
        date_from, date_to = resolve_time_range(time_range_entity)
        sport_type: str | None = entities.get("sport_type")

        logger.info(
            "ToolExecutor: вызываем %s | user=%s date_from=%s date_to=%s sport=%s",
            tool_calls, user_id, date_from, date_to, sport_type,
        )

        executor_result = ToolExecutorResult()

        for tool_name in tool_calls:
            args = self._build_args_snapshot(
                tool_name=tool_name,
                user_id=user_id,
                date_from=date_from,
                date_to=date_to,
                sport_type=sport_type,
                query_text=query_text,
                entities=entities,
            )
            start_ms = time.monotonic() * 1000
            result = await self._dispatch(
                tool_name=tool_name,
                user_id=user_id,
                date_from=date_from,
                date_to=date_to,
                sport_type=sport_type,
                db=db,
                query_text=query_text,
                entities=entities,
            )
            duration_ms = int(time.monotonic() * 1000 - start_ms)
            tool_call_logger.record(
                name=tool_name,
                source="tool_executor",
                args=args,
                result=result.data if result.success else None,
                success=result.success,
                error=result.error,
                duration_ms=duration_ms,
            )
            executor_result.results[tool_name] = result

        return executor_result

    @staticmethod
    def _build_args_snapshot(
        tool_name: str,
        user_id: str,
        date_from: date,
        date_to: date,
        sport_type: str | None,
        query_text: str | None,
        entities: dict | None,
    ) -> dict[str, Any]:
        """Собрать snapshot аргументов, с которыми будет вызван tool.

        Используется для записи в tool_calls.args — чтобы в админке было видно,
        с чем именно tool был вызван (даты, sport_type, top_k, категория и т.д.).
        """
        ents = entities or {}
        base: dict[str, Any] = {
            "user_id": user_id,
            "date_from": date_from.isoformat() if date_from else None,
            "date_to": date_to.isoformat() if date_to else None,
            "sport_type": sport_type,
        }
        if tool_name == "rag_retrieve":
            base["query_text"] = query_text
            base["rag_category"] = ents.get("rag_category")
            base["rag_top_k"] = int(ents.get("rag_top_k", 5))
        if tool_name in {"compute_recovery", "check_overtraining"}:
            # эти tools смотрят свои окна, а не date_from/date_to из entities
            base.pop("date_from", None)
            base.pop("date_to", None)
            base["window_days"] = 14 if tool_name == "compute_recovery" else 14
        if tool_name == "compute_strain":
            base["reference_date"] = date_to.isoformat() if date_to else None
        return base

    async def _dispatch(
        self,
        tool_name: str,
        user_id: str,
        date_from: date,
        date_to: date,
        sport_type: str | None,
        db: AsyncSession,
        query_text: str | None = None,
        entities: dict | None = None,
    ) -> ToolResult:
        """Вызвать конкретный tool по имени с нужными параметрами."""
        if tool_name == "get_activities":
            return await get_activities(
                db=db,
                user_id=user_id,
                date_from=date_from,
                date_to=date_to,
                sport_type=sport_type,
            )

        if tool_name == "get_activities_by_sport":
            if not sport_type:
                logger.warning(
                    "ToolExecutor: get_activities_by_sport вызван без sport_type, "
                    "используем get_activities"
                )
                return await get_activities(
                    db=db,
                    user_id=user_id,
                    date_from=date_from,
                    date_to=date_to,
                )
            return await get_activities_by_sport(
                db=db,
                user_id=user_id,
                sport_type=sport_type,
            )

        if tool_name == "get_daily_facts":
            return await get_daily_facts(
                db=db,
                user_id=user_id,
                date_from=date_from,
                date_to=date_to,
            )

        if tool_name == "get_user_profile":
            return await get_user_profile(db=db, user_id=user_id)

        if tool_name == "rag_retrieve":
            if not query_text:
                logger.warning("ToolExecutor: rag_retrieve вызван без query_text")
                return ToolResult(
                    tool_name="rag_retrieve",
                    success=False,
                    data=None,
                    error="query_text не передан",
                )
            ents = entities or {}
            category = ents.get("rag_category")
            top_k = int(ents.get("rag_top_k", 5))
            return await rag_retrieve(
                query=query_text,
                category=category,
                sport_type=sport_type,
                top_k=top_k,
            )

        if tool_name == "compute_recovery":
            return await self._compute_recovery(user_id=user_id, db=db)

        if tool_name == "compute_strain":
            return await self._compute_strain(
                user_id=user_id, db=db, reference_date=date_to
            )

        if tool_name == "check_overtraining":
            return await self._check_overtraining(user_id=user_id, db=db)

        # log_activity и update_profile вызываются через execute_action (запись)
        logger.warning("ToolExecutor: неизвестный tool '%s'", tool_name)
        return ToolResult(
            tool_name=tool_name,
            success=False,
            data=None,
            error=f"Неизвестный tool: {tool_name}",
        )

    async def _compute_recovery(
        self, user_id: str, db: AsyncSession
    ) -> ToolResult:
        """Рассчитать recovery score за последние 14 дней."""
        import dataclasses
        from app.services.data_processing.recovery_score import compute_recovery_score

        today = date.today()
        facts_res = await get_daily_facts(
            db=db, user_id=user_id,
            date_from=today - timedelta(days=13),
            date_to=today,
        )
        acts_res = await get_activities(
            db=db, user_id=user_id,
            date_from=today - timedelta(days=27),
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
        self, user_id: str, db: AsyncSession
    ) -> ToolResult:
        """Проверить маркеры перетренированности за последние 14 дней."""
        import dataclasses
        from app.services.data_processing.overtraining_detection import detect_overtraining

        today = date.today()
        facts_res = await get_daily_facts(
            db=db, user_id=user_id,
            date_from=today - timedelta(days=13),
            date_to=today,
        )
        acts_res = await get_activities(
            db=db, user_id=user_id,
            date_from=today - timedelta(days=27),
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

    async def execute_action(
        self,
        tool_name: str,
        user_id: str,
        params: dict,
        db: AsyncSession,
    ) -> ToolResult:
        """Вызвать write-инструмент (log_activity, update_profile).

        Args:
            tool_name: Имя write-tool.
            user_id: Идентификатор пользователя.
            params: Параметры инструмента (sport_type, duration и т.д.).
            db: Асинхронная сессия SQLAlchemy.

        Returns:
            ToolResult с результатом операции записи.
        """
        args_snapshot = {"user_id": user_id, **dict(params or {})}
        start_ms = time.monotonic() * 1000
        if tool_name == "log_activity":
            result = await log_activity(
                db=db,
                user_id=user_id,
                sport_type=params.get("sport_type", "other"),
                duration=params.get("duration", 0),
                calories=params.get("calories", 0),
                distance=params.get("distance"),
                notes=params.get("notes"),
            )
        elif tool_name == "update_profile":
            result = await update_profile(
                db=db,
                user_id=user_id,
                field=params.get("field", ""),
                value=params.get("value"),
            )
        else:
            logger.warning("ToolExecutor.execute_action: неизвестный tool '%s'", tool_name)
            result = ToolResult(
                tool_name=tool_name,
                success=False,
                data=None,
                error=f"Неизвестный write-tool: {tool_name}",
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
        return result
