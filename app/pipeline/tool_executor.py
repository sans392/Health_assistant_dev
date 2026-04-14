"""Диспетчер инструментов (Tool Executor).

Вызывает нужные tools по именам, разрешает параметры из сущностей запроса,
агрегирует результаты.
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.tools.db_tools import (
    ToolResult,
    get_activities,
    get_activities_by_sport,
    get_daily_facts,
    get_user_profile,
    log_activity,
    update_profile,
)
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
            result = await self._dispatch(
                tool_name=tool_name,
                user_id=user_id,
                date_from=date_from,
                date_to=date_to,
                sport_type=sport_type,
                db=db,
            )
            executor_result.results[tool_name] = result

        return executor_result

    async def _dispatch(
        self,
        tool_name: str,
        user_id: str,
        date_from: date,
        date_to: date,
        sport_type: str | None,
        db: AsyncSession,
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

        # log_activity и update_profile вызываются через execute_action (запись)
        logger.warning("ToolExecutor: неизвестный tool '%s'", tool_name)
        return ToolResult(
            tool_name=tool_name,
            success=False,
            data=None,
            error=f"Неизвестный tool: {tool_name}",
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
        if tool_name == "log_activity":
            return await log_activity(
                db=db,
                user_id=user_id,
                sport_type=params.get("sport_type", "other"),
                duration=params.get("duration", 0),
                calories=params.get("calories", 0),
                distance=params.get("distance"),
                notes=params.get("notes"),
            )

        if tool_name == "update_profile":
            return await update_profile(
                db=db,
                user_id=user_id,
                field=params.get("field", ""),
                value=params.get("value"),
            )

        logger.warning("ToolExecutor.execute_action: неизвестный tool '%s'", tool_name)
        return ToolResult(
            tool_name=tool_name,
            success=False,
            data=None,
            error=f"Неизвестный write-tool: {tool_name}",
        )
