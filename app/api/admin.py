"""API endpoints для Admin Panel (логи пайплайна).

GET /api/admin/logs          — список логов с пагинацией и фильтрами
GET /api/admin/logs/{id}     — детали одного запроса

Защита: HTTP Basic Auth (admin_username / admin_password из settings).
"""

import secrets
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy import and_, select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_db
from app.models.pipeline_log import PipelineLog

router = APIRouter(prefix="/api/admin", tags=["admin"])
_security = HTTPBasic()


def _require_admin(credentials: HTTPBasicCredentials = Depends(_security)) -> None:
    """Проверка Basic Auth для admin endpoints."""
    correct_username = secrets.compare_digest(
        credentials.username.encode(), settings.admin_username.encode()
    )
    correct_password = secrets.compare_digest(
        credentials.password.encode(), settings.admin_password.encode()
    )
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверные учётные данные",
            headers={"WWW-Authenticate": "Basic"},
        )


def _log_to_dict(log: PipelineLog) -> dict[str, Any]:
    """Сериализовать PipelineLog в словарь для API-ответа."""
    return {
        "id": log.id,
        "user_id": log.user_id,
        "session_id": log.session_id,
        "timestamp": log.timestamp.isoformat() if log.timestamp else None,
        "raw_query": log.raw_query,
        "intent": log.intent,
        "intent_confidence": log.intent_confidence,
        "route": log.route,
        "fast_path": log.fast_path,
        "safety_level": log.safety_level,
        "tools_called": log.tools_called,
        "modules_used": log.modules_used,
        "llm_model_used": log.llm_model_used,
        "llm_calls_count": log.llm_calls_count,
        "total_duration_ms": log.total_duration_ms,
        "response_length": log.response_length,
        "errors": log.errors,
        "has_errors": bool(log.errors),
    }


def _log_to_dict_full(log: PipelineLog) -> dict[str, Any]:
    """Полная сериализация PipelineLog (включая response_text для детального просмотра)."""
    data = _log_to_dict(log)
    data["response_text"] = log.response_text
    return data


@router.get("/logs", dependencies=[Depends(_require_admin)])
async def list_logs(
    page: int = Query(1, ge=1, description="Номер страницы"),
    per_page: int = Query(20, ge=1, le=100, description="Записей на странице"),
    date_from: str | None = Query(None, description="Фильтр от даты (YYYY-MM-DD)"),
    date_to: str | None = Query(None, description="Фильтр до даты (YYYY-MM-DD)"),
    intent: str | None = Query(None, description="Фильтр по intent"),
    route: str | None = Query(None, description="Фильтр по route"),
    has_errors: bool | None = Query(None, description="Только записи с ошибками"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Список логов пайплайна с пагинацией и фильтрами.

    Фильтры:
    - date_from / date_to: диапазон дат (YYYY-MM-DD)
    - intent: тип намерения (general_chat, data_retrieval, ...)
    - route: маршрут (fast_direct_answer, tool_simple, ...)
    - has_errors: True — только с ошибками, False — только без ошибок
    """
    conditions = []

    if date_from:
        try:
            dt_from = datetime.fromisoformat(date_from)
            conditions.append(PipelineLog.timestamp >= dt_from)
        except ValueError:
            raise HTTPException(status_code=400, detail="Неверный формат date_from (ожидается YYYY-MM-DD)")

    if date_to:
        try:
            # Берём конец дня
            dt_to = datetime.fromisoformat(date_to).replace(hour=23, minute=59, second=59)
            conditions.append(PipelineLog.timestamp <= dt_to)
        except ValueError:
            raise HTTPException(status_code=400, detail="Неверный формат date_to (ожидается YYYY-MM-DD)")

    if intent:
        conditions.append(PipelineLog.intent == intent)

    if route:
        conditions.append(PipelineLog.route == route)

    if has_errors is True:
        # JSON не None и не пустой список
        conditions.append(PipelineLog.errors.is_not(None))
    elif has_errors is False:
        conditions.append(PipelineLog.errors.is_(None))

    where_clause = and_(*conditions) if conditions else True

    # Подсчёт общего количества
    count_stmt = select(func.count()).select_from(PipelineLog).where(where_clause)
    total_result = await db.execute(count_stmt)
    total: int = total_result.scalar() or 0

    # Загрузка страницы (сортировка по убыванию даты)
    offset = (page - 1) * per_page
    stmt = (
        select(PipelineLog)
        .where(where_clause)
        .order_by(PipelineLog.timestamp.desc())
        .offset(offset)
        .limit(per_page)
    )
    result = await db.execute(stmt)
    logs = result.scalars().all()

    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page if total > 0 else 0,
        "items": [_log_to_dict(log) for log in logs],
    }


@router.get("/logs/{request_id}", dependencies=[Depends(_require_admin)])
async def get_log(
    request_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Детали одного запроса по request_id (включая полный текст ответа)."""
    stmt = select(PipelineLog).where(PipelineLog.id == request_id)
    result = await db.execute(stmt)
    log = result.scalar_one_or_none()

    if log is None:
        raise HTTPException(
            status_code=404,
            detail=f"Лог с id={request_id} не найден",
        )

    return _log_to_dict_full(log)
