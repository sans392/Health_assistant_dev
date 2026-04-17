"""Template Plan Executor — детерминированный исполнитель шаблонных планов (Phase 2, Issue #28).

Каждый шаблон — фиксированная последовательность tool-вызовов.
Результаты собираются в TemplateResult и передаются в Response Generator.
"""

import dataclasses
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.tools.db_tools import get_activities, get_daily_facts, get_user_profile
from app.tools.rag_retrieve import rag_retrieve

logger = logging.getLogger(__name__)


@dataclass
class TemplateStepResult:
    """Результат одного шага шаблона."""

    tool: str
    args: dict
    data: Any
    success: bool
    error: str | None = None


@dataclass
class TemplateResult:
    """Результат выполнения шаблона."""

    template_id: str
    steps: list[TemplateStepResult] = field(default_factory=list)
    structured_data: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return bool(self.steps) and any(s.success for s in self.steps)


# Реестр шаблонов — фиксированные последовательности шагов
TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "weekly_training_plan": [
        {"tool": "get_user_profile", "args": {}},
        {"tool": "get_activities", "args": {"days": 14}},
        {"tool": "compute_training_load", "args": {}},
        {"tool": "rag_retrieve", "args": {"category": "training_principles", "top_k": 3}},
        {"tool": "rag_retrieve", "args": {"category": "sport_specific", "top_k": 2}},
    ],
    "recovery_report": [
        {"tool": "get_daily_facts", "args": {"days": 7}},
        {"tool": "compute_recovery", "args": {}},
        {"tool": "rag_retrieve", "args": {"category": "recovery_science", "top_k": 3}},
    ],
    "overtraining_check": [
        {"tool": "get_daily_facts", "args": {"days": 14}},
        {"tool": "check_overtraining", "args": {}},
        {"tool": "rag_retrieve", "args": {"category": "recovery_science", "top_k": 3}},
    ],
    "progress_report": [
        {"tool": "get_activities", "args": {"days": 30}},
        {"tool": "compute_trend", "args": {"days": 30}},
        {"tool": "rag_retrieve", "args": {"category": "training_principles", "top_k": 2}},
    ],
}


class TemplatePlanExecutor:
    """Исполняет шаблонные планы — фиксированную последовательность инструментов.

    Шаги выполняются последовательно. Результаты накапливаются в TemplateResult
    для последующей передачи в Response Generator.
    """

    async def execute(
        self,
        template_id: str,
        user_id: str,
        query_text: str,
        entities: dict,
        db: AsyncSession,
    ) -> TemplateResult:
        """Выполнить шаблон.

        Args:
            template_id: Идентификатор шаблона из TEMPLATES.
            user_id: ID пользователя.
            query_text: Текст запроса (для rag_retrieve).
            entities: Сущности из IntentResult (sport_type и т.д.).
            db: Async DB session.

        Returns:
            TemplateResult со всеми данными шагов.
        """
        if template_id not in TEMPLATES:
            logger.error("TemplatePlanExecutor: неизвестный шаблон '%s'", template_id)
            return TemplateResult(template_id=template_id)

        steps_def = TEMPLATES[template_id]
        result = TemplateResult(template_id=template_id)
        sport_type: str | None = entities.get("sport_type")

        logger.info(
            "TemplatePlanExecutor: шаблон '%s' | %d шагов | user=%s",
            template_id, len(steps_def), user_id,
        )

        for step_def in steps_def:
            tool = step_def["tool"]
            args = step_def["args"]
            step = await self._run_step(
                tool=tool, args=args,
                user_id=user_id, query_text=query_text,
                sport_type=sport_type, db=db,
            )
            result.steps.append(step)

            if step.success and step.data is not None:
                # Ключ: "tool_category" или просто "tool"
                category = args.get("category", "")
                key = f"{tool}_{category}" if category else tool
                # Несколько rag_retrieve одной категории — объединяем списки
                if key in result.structured_data and isinstance(step.data, list):
                    existing = result.structured_data[key]
                    if isinstance(existing, list):
                        result.structured_data[key] = existing + step.data
                        continue
                result.structured_data[key] = step.data

        success_count = sum(1 for s in result.steps if s.success)
        logger.info(
            "TemplatePlanExecutor: '%s' завершён | %d/%d шагов успешны",
            template_id, success_count, len(result.steps),
        )
        return result

    async def _run_step(
        self,
        tool: str,
        args: dict,
        user_id: str,
        query_text: str,
        sport_type: str | None,
        db: AsyncSession,
    ) -> TemplateStepResult:
        """Запустить один шаг шаблона."""
        try:
            data = await self._dispatch(
                tool=tool, args=args,
                user_id=user_id, query_text=query_text,
                sport_type=sport_type, db=db,
            )
            return TemplateStepResult(tool=tool, args=args, data=data, success=True)
        except Exception as exc:
            logger.error("TemplatePlanExecutor: шаг '%s' — ошибка: %s", tool, exc)
            return TemplateStepResult(tool=tool, args=args, data=None, success=False, error=str(exc))

    async def _dispatch(
        self,
        tool: str,
        args: dict,
        user_id: str,
        query_text: str,
        sport_type: str | None,
        db: AsyncSession,
    ) -> Any:
        """Диспатчер инструментов шаблона."""
        today = date.today()
        days = int(args.get("days", 7))
        date_from = today - timedelta(days=days - 1)

        if tool == "get_user_profile":
            res = await get_user_profile(db=db, user_id=user_id)
            return res.data

        if tool == "get_activities":
            res = await get_activities(db=db, user_id=user_id, date_from=date_from, date_to=today)
            return res.data

        if tool == "get_daily_facts":
            res = await get_daily_facts(db=db, user_id=user_id, date_from=date_from, date_to=today)
            return res.data

        if tool == "compute_recovery":
            from app.services.data_processing.recovery_score import compute_recovery_score
            facts_res = await get_daily_facts(
                db=db, user_id=user_id,
                date_from=today - timedelta(days=13), date_to=today,
            )
            acts_res = await get_activities(
                db=db, user_id=user_id,
                date_from=today - timedelta(days=27), date_to=today,
            )
            r = compute_recovery_score(
                daily_facts=facts_res.data or [],
                activities=acts_res.data or [],
            )
            return dataclasses.asdict(r)

        if tool == "compute_training_load":
            from app.services.data_processing.training_load import compute_training_load
            acts_res = await get_activities(
                db=db, user_id=user_id,
                date_from=today - timedelta(days=27), date_to=today,
            )
            r = compute_training_load(acts_res.data or [])
            return dataclasses.asdict(r)

        if tool == "check_overtraining":
            from app.services.data_processing.overtraining_detection import detect_overtraining
            facts_res = await get_daily_facts(
                db=db, user_id=user_id,
                date_from=today - timedelta(days=13), date_to=today,
            )
            acts_res = await get_activities(
                db=db, user_id=user_id,
                date_from=today - timedelta(days=27), date_to=today,
            )
            r = detect_overtraining(
                daily_facts=facts_res.data or [],
                activities=acts_res.data or [],
            )
            return dataclasses.asdict(r)

        if tool == "compute_trend":
            from app.services.data_processing.trend_analyzer import (
                analyze_trend, build_time_series_from_activities,
            )
            acts_res = await get_activities(
                db=db, user_id=user_id, date_from=date_from, date_to=today,
            )
            ts = build_time_series_from_activities(acts_res.data or [], "duration_seconds")
            r = analyze_trend(ts)
            return dataclasses.asdict(r)

        if tool == "rag_retrieve":
            category = args.get("category")
            top_k = int(args.get("top_k", 3))
            res = await rag_retrieve(
                query=query_text,
                category=category,
                sport_type=sport_type,
                top_k=top_k,
            )
            return res.data

        logger.warning("TemplatePlanExecutor: неизвестный tool '%s'", tool)
        return None


# Глобальный синглтон
template_plan_executor = TemplatePlanExecutor()
