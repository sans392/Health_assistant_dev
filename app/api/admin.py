"""API endpoints для Admin Panel.

Все endpoints защищены HTTP Basic Auth (admin_username / admin_password из settings).

Logs API (из issue #12):
  GET /api/admin/logs                       — список логов с фильтрами и пагинацией
  GET /api/admin/logs/{id}                  — детали одного лога
  GET /api/admin/logs/{id}/llm-calls        — LLM-вызовы конкретного лога (issue #34)
  GET /api/admin/logs/{id}/stage-trace      — stage trace конкретного лога (issue #34)

Admin API (issue #14):
  GET  /api/admin/stats            — статистика для dashboard
  GET  /api/admin/system-status    — статус Ollama + системы
  GET  /api/admin/activities       — тренировки с фильтрами
  GET  /api/admin/daily-facts      — дневные метрики
  GET  /api/admin/profiles         — профили пользователей
  PUT  /api/admin/profiles/{id}    — редактирование профиля (базовые поля)
  POST /api/admin/profiles/{id}    — полное обновление с валидацией JSON (issue #34)
  POST /api/admin/seed             — запуск SeedGenerator с параметрами (issue #33)
  POST /api/admin/seed/preview     — предпросмотр нескольких записей без записи (issue #33)
  GET  /api/admin/seed/runs        — журнал запусков SeedGenerator (issue #33)
  POST /api/admin/test-llm         — тест промпта через Ollama
"""

import json
import secrets
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_db
from app.models.activity import Activity
from app.models.daily_fact import DailyFact
from app.models.llm_call import LLMCall
from app.models.pipeline_log import PipelineLog
from app.models.user_profile import UserProfile

router = APIRouter(prefix="/api/admin", tags=["admin"])
_security = HTTPBasic()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers — сериализация
# ---------------------------------------------------------------------------

def _log_to_dict(log: PipelineLog) -> dict[str, Any]:
    stage_trace = log.stage_trace or []
    rag_chunks_used = log.rag_chunks_used or []
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
        "stages_count": len(stage_trace),
        "rag_chunks_count": len(rag_chunks_used),
        "total_duration_ms": log.total_duration_ms,
        "response_length": log.response_length,
        "errors": log.errors,
        "has_errors": bool(log.errors),
    }


def _log_to_dict_full(log: PipelineLog) -> dict[str, Any]:
    data = _log_to_dict(log)
    data["response_text"] = log.response_text
    data["stage_trace"] = log.stage_trace or []
    data["llm_role_usage"] = log.llm_role_usage or {}
    data["rag_chunks_used"] = log.rag_chunks_used or []
    return data


def _activity_to_dict(a: Activity) -> dict[str, Any]:
    return {
        "id": a.id,
        "user_id": a.user_id,
        "title": a.title,
        "sport_type": a.sport_type,
        "distance_meters": a.distance_meters,
        "duration_seconds": a.duration_seconds,
        "start_time": a.start_time.isoformat() if a.start_time else None,
        "end_time": a.end_time.isoformat() if a.end_time else None,
        "calories": a.calories,
        "avg_heart_rate": a.avg_heart_rate,
        "source": a.source,
    }


def _daily_fact_to_dict(df: DailyFact) -> dict[str, Any]:
    return {
        "id": df.id,
        "user_id": df.user_id,
        "iso_date": df.iso_date,
        "steps": df.steps,
        "calories_kcal": df.calories_kcal,
        "recovery_score": df.recovery_score,
        "hrv_rmssd_milli": df.hrv_rmssd_milli,
        "resting_heart_rate": df.resting_heart_rate,
        "spo2_percentage": df.spo2_percentage,
        "sleep_total_in_bed_milli": df.sleep_total_in_bed_milli,
        "water_liters": df.water_liters,
    }


def _profile_to_dict(p: UserProfile) -> dict[str, Any]:
    return {
        "id": p.id,
        "user_id": p.user_id,
        "name": p.name,
        "age": p.age,
        "weight_kg": p.weight_kg,
        "height_cm": p.height_cm,
        "gender": p.gender,
        "max_heart_rate": p.max_heart_rate,
        "resting_heart_rate": p.resting_heart_rate,
        "training_goals": p.training_goals,
        "experience_level": p.experience_level,
        "injuries": p.injuries,
        "chronic_conditions": p.chronic_conditions,
        "preferred_sports": p.preferred_sports,
        "created_at": p.created_at.isoformat() if p.created_at else None,
        "updated_at": p.updated_at.isoformat() if p.updated_at else None,
    }


# ---------------------------------------------------------------------------
# Logs API (issue #12)
# ---------------------------------------------------------------------------

@router.get("/logs", dependencies=[Depends(_require_admin)])
async def list_logs(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    date_from: str | None = Query(None, description="YYYY-MM-DD"),
    date_to: str | None = Query(None, description="YYYY-MM-DD"),
    intent: str | None = Query(None),
    route: str | None = Query(None),
    has_errors: bool | None = Query(None),
    model_role: str | None = Query(None, description="intent_llm | response | planner"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Список логов пайплайна с пагинацией и фильтрами."""
    conditions = []

    if date_from:
        try:
            conditions.append(PipelineLog.timestamp >= datetime.fromisoformat(date_from))
        except ValueError:
            raise HTTPException(status_code=400, detail="Неверный формат date_from (YYYY-MM-DD)")

    if date_to:
        try:
            dt_to = datetime.fromisoformat(date_to).replace(hour=23, minute=59, second=59)
            conditions.append(PipelineLog.timestamp <= dt_to)
        except ValueError:
            raise HTTPException(status_code=400, detail="Неверный формат date_to (YYYY-MM-DD)")

    if intent:
        conditions.append(PipelineLog.intent == intent)
    if route:
        conditions.append(PipelineLog.route == route)
    if has_errors is True:
        conditions.append(PipelineLog.errors.is_not(None))
    elif has_errors is False:
        conditions.append(PipelineLog.errors.is_(None))

    # Фильтр по роли LLM — через подзапрос к llm_calls
    if model_role:
        subq = select(LLMCall.request_id).where(LLMCall.role == model_role).distinct()
        conditions.append(PipelineLog.id.in_(subq))

    where_clause = and_(*conditions) if conditions else True

    total = (await db.execute(
        select(func.count()).select_from(PipelineLog).where(where_clause)
    )).scalar() or 0

    offset = (page - 1) * per_page
    logs = (await db.execute(
        select(PipelineLog).where(where_clause)
        .order_by(PipelineLog.timestamp.desc())
        .offset(offset).limit(per_page)
    )).scalars().all()

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
    """Детали одного запроса (полный pipeline trace)."""
    result = await db.execute(select(PipelineLog).where(PipelineLog.id == request_id))
    log = result.scalar_one_or_none()
    if log is None:
        raise HTTPException(status_code=404, detail=f"Лог {request_id} не найден")
    return _log_to_dict_full(log)


@router.get("/logs/{request_id}/llm-calls", dependencies=[Depends(_require_admin)])
async def get_log_llm_calls(
    request_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """LLM-вызовы конкретного запроса (issue #34)."""
    calls = (await db.execute(
        select(LLMCall)
        .where(LLMCall.request_id == request_id)
        .order_by(LLMCall.timestamp)
    )).scalars().all()

    return {
        "request_id": request_id,
        "total": len(calls),
        "items": [
            {
                "id": c.id,
                "role": c.role,
                "model": c.model,
                "duration_ms": c.duration_ms,
                "prompt_length": c.prompt_length,
                "response_length": c.response_length,
                "prompt_preview": (c.prompt or "")[:200],
                "response_preview": (c.response or "")[:200],
                "prompt": c.prompt or "",
                "response": c.response or "",
                "iteration": c.iteration,
                "timestamp": c.timestamp.isoformat() if c.timestamp else None,
            }
            for c in calls
        ],
    }


@router.get("/logs/{request_id}/stage-trace", dependencies=[Depends(_require_admin)])
async def get_log_stage_trace(
    request_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Stage trace конкретного запроса (issue #34)."""
    result = await db.execute(select(PipelineLog).where(PipelineLog.id == request_id))
    log = result.scalar_one_or_none()
    if log is None:
        raise HTTPException(status_code=404, detail=f"Лог {request_id} не найден")

    stage_trace = log.stage_trace or []
    total_ms = log.total_duration_ms or 1

    # Добавляем процент для waterfall-визуализации
    for stage in stage_trace:
        dur = stage.get("duration_ms") or 0
        stage["width_pct"] = round(dur / total_ms * 100, 1) if total_ms else 0
        start = stage.get("start_ms") or 0
        stage["offset_pct"] = round(start / total_ms * 100, 1) if total_ms else 0

    return {
        "request_id": request_id,
        "total_duration_ms": total_ms,
        "stages": stage_trace,
    }


# ---------------------------------------------------------------------------
# Dashboard stats
# ---------------------------------------------------------------------------

@router.get("/stats", dependencies=[Depends(_require_admin)])
async def get_stats(
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Статистика для dashboard: запросы, avg duration, ошибки, intent distribution."""
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    # За сегодня
    today_logs = (await db.execute(
        select(PipelineLog).where(PipelineLog.timestamp >= today_start)
    )).scalars().all()

    total_today = len(today_logs)
    avg_duration = (
        sum(l.total_duration_ms or 0 for l in today_logs) // total_today
        if total_today else 0
    )
    errors_today = sum(1 for l in today_logs if l.errors)

    # Intent distribution за сегодня
    intent_counts: dict[str, int] = {}
    for log in today_logs:
        key = log.intent or "unknown"
        intent_counts[key] = intent_counts.get(key, 0) + 1

    # Последние 10 запросов
    recent = (await db.execute(
        select(PipelineLog).order_by(PipelineLog.timestamp.desc()).limit(10)
    )).scalars().all()

    # Размер БД (примерный)
    import os
    db_size_bytes = 0
    try:
        db_size_bytes = os.path.getsize(settings.db_path)
    except OSError:
        pass

    return {
        "today": {
            "total_requests": total_today,
            "avg_duration_ms": avg_duration,
            "errors_count": errors_today,
        },
        "intent_distribution": intent_counts,
        "recent_requests": [_log_to_dict(l) for l in recent],
        "db_size_bytes": db_size_bytes,
    }


# ---------------------------------------------------------------------------
# System status
# ---------------------------------------------------------------------------

@router.get("/system-status", dependencies=[Depends(_require_admin)])
async def get_system_status() -> dict[str, Any]:
    """Статус Ollama и системы."""
    from app.services.llm_service import ollama_client
    ollama_status = await ollama_client.health_check()

    # Uptime приложения (читаем /proc/uptime для хоста, fallback — N/A)
    uptime_str = "n/a"
    try:
        with open("/proc/uptime") as f:
            uptime_sec = float(f.read().split()[0])
        h, rem = divmod(int(uptime_sec), 3600)
        m, s = divmod(rem, 60)
        uptime_str = f"{h}h {m}m {s}s"
    except OSError:
        pass

    return {
        "ollama": ollama_status,
        "config": {
            "ollama_model": settings.ollama_model,
            "ollama_host": settings.ollama_host,
            "ollama_timeout": settings.ollama_timeout,
            "log_level": settings.log_level,
        },
        "uptime": uptime_str,
    }


# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------

@router.get("/activities", dependencies=[Depends(_require_admin)])
async def list_activities(
    user_id: str | None = Query(None),
    sport_type: str | None = Query(None),
    date_from: str | None = Query(None, description="YYYY-MM-DD"),
    date_to: str | None = Query(None, description="YYYY-MM-DD"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Список тренировок с фильтрами."""
    conditions = []
    if user_id:
        conditions.append(Activity.user_id == user_id)
    if sport_type:
        conditions.append(Activity.sport_type == sport_type)
    if date_from:
        try:
            conditions.append(Activity.start_time >= datetime.fromisoformat(date_from))
        except ValueError:
            raise HTTPException(status_code=400, detail="Неверный формат date_from")
    if date_to:
        try:
            dt_to = datetime.fromisoformat(date_to).replace(hour=23, minute=59, second=59)
            conditions.append(Activity.start_time <= dt_to)
        except ValueError:
            raise HTTPException(status_code=400, detail="Неверный формат date_to")

    where_clause = and_(*conditions) if conditions else True
    total = (await db.execute(
        select(func.count()).select_from(Activity).where(where_clause)
    )).scalar() or 0

    activities = (await db.execute(
        select(Activity).where(where_clause)
        .order_by(Activity.start_time.desc())
        .offset((page - 1) * per_page).limit(per_page)
    )).scalars().all()

    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page if total > 0 else 0,
        "items": [_activity_to_dict(a) for a in activities],
    }


# ---------------------------------------------------------------------------
# Daily Facts
# ---------------------------------------------------------------------------

@router.get("/daily-facts", dependencies=[Depends(_require_admin)])
async def list_daily_facts(
    user_id: str | None = Query(None),
    date_from: str | None = Query(None, description="YYYY-MM-DD"),
    date_to: str | None = Query(None, description="YYYY-MM-DD"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Список дневных метрик с фильтрами."""
    conditions = []
    if user_id:
        conditions.append(DailyFact.user_id == user_id)
    if date_from:
        conditions.append(DailyFact.iso_date >= date_from)
    if date_to:
        conditions.append(DailyFact.iso_date <= date_to)

    where_clause = and_(*conditions) if conditions else True
    total = (await db.execute(
        select(func.count()).select_from(DailyFact).where(where_clause)
    )).scalar() or 0

    facts = (await db.execute(
        select(DailyFact).where(where_clause)
        .order_by(DailyFact.iso_date.desc())
        .offset((page - 1) * per_page).limit(per_page)
    )).scalars().all()

    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page if total > 0 else 0,
        "items": [_daily_fact_to_dict(df) for df in facts],
    }


# ---------------------------------------------------------------------------
# User Profiles
# ---------------------------------------------------------------------------

@router.get("/profiles", dependencies=[Depends(_require_admin)])
async def list_profiles(
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Список профилей пользователей."""
    profiles = (await db.execute(
        select(UserProfile).order_by(UserProfile.created_at.desc())
    )).scalars().all()
    return {"items": [_profile_to_dict(p) for p in profiles]}


@router.put("/profiles/{profile_id}", dependencies=[Depends(_require_admin)])
async def update_profile(
    profile_id: str,
    payload: dict[str, Any] = Body(...),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Редактирование профиля пользователя."""
    result = await db.execute(select(UserProfile).where(UserProfile.id == profile_id))
    profile = result.scalar_one_or_none()
    if profile is None:
        raise HTTPException(status_code=404, detail="Профиль не найден")

    allowed_fields = {
        "name", "age", "weight_kg", "height_cm", "gender",
        "max_heart_rate", "resting_heart_rate", "training_goals",
        "experience_level", "injuries", "chronic_conditions", "preferred_sports",
    }
    for field, value in payload.items():
        if field in allowed_fields:
            setattr(profile, field, value)

    profile.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(profile)
    return _profile_to_dict(profile)


@router.post("/profiles/{profile_id}", dependencies=[Depends(_require_admin)])
async def update_profile_full(
    profile_id: str,
    payload: dict[str, Any] = Body(...),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Полное обновление профиля с валидацией JSON-полей (issue #34).

    JSON-поля (injuries, chronic_conditions, preferred_sports, training_goals)
    принимаются как строки (из textarea) и парсятся на сервере.
    Невалидный JSON → HTTP 400.
    """
    result = await db.execute(select(UserProfile).where(UserProfile.id == profile_id))
    profile = result.scalar_one_or_none()
    if profile is None:
        raise HTTPException(status_code=404, detail="Профиль не найден")

    allowed_fields = {
        "name", "age", "weight_kg", "height_cm", "gender",
        "max_heart_rate", "resting_heart_rate", "training_goals",
        "experience_level", "injuries", "chronic_conditions", "preferred_sports",
    }
    json_list_fields = {"injuries", "chronic_conditions", "preferred_sports", "training_goals"}

    for field, value in payload.items():
        if field not in allowed_fields:
            continue
        if field in json_list_fields and isinstance(value, str):
            value = value.strip()
            if not value:
                value = []
            else:
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Поле '{field}' должно быть валидным JSON-массивом",
                    )
                if not isinstance(parsed, list):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Поле '{field}' должно быть JSON-массивом (list)",
                    )
                value = parsed
        setattr(profile, field, value)

    profile.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(profile)
    return _profile_to_dict(profile)


# ---------------------------------------------------------------------------
# Chat Sessions (data browser)
# ---------------------------------------------------------------------------

@router.get("/chat-sessions", dependencies=[Depends(_require_admin)])
async def list_chat_sessions(
    user_id: str | None = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Список чат-сессий."""
    from app.models.chat import ChatSession, ChatMessage

    conditions = []
    if user_id:
        conditions.append(ChatSession.user_id == user_id)

    where_clause = and_(*conditions) if conditions else True
    total = (await db.execute(
        select(func.count()).select_from(ChatSession).where(where_clause)
    )).scalar() or 0

    sessions = (await db.execute(
        select(ChatSession).where(where_clause)
        .order_by(ChatSession.updated_at.desc())
        .offset((page - 1) * per_page).limit(per_page)
    )).scalars().all()

    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page if total > 0 else 0,
        "items": [
            {
                "id": s.id,
                "user_id": s.user_id,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
            }
            for s in sessions
        ],
    }


@router.get("/chat-sessions/{session_id}/messages", dependencies=[Depends(_require_admin)])
async def get_chat_session_messages(
    session_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """История сообщений чат-сессии."""
    from app.models.chat import ChatMessage

    messages = (await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.order_index)
    )).scalars().all()

    return {
        "session_id": session_id,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat(),
                "order_index": m.order_index,
            }
            for m in messages
        ],
    }


# ---------------------------------------------------------------------------
# Seed Generator v2 (Issue #33)
# ---------------------------------------------------------------------------

@router.post("/seed", dependencies=[Depends(_require_admin)])
async def run_seed(
    payload: dict[str, Any] = Body(default={}),
    credentials: HTTPBasicCredentials = Depends(_security),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Запустить SeedGenerator с параметрами.

    Body (все поля опциональны):
      scenario: normal_load | overreaching | recovery_phase | injury_recovery
      profile_preset: beginner | intermediate | advanced
      days: int (1–365)
      user_count: int (1–10)
      add_anomalies: bool
      missing_data_rate: float 0–1
      truncate_before: bool
    """
    import asyncio
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session as SyncSession
    from scripts.seed_data import SeedGenerator
    from app.models.seed_run import SeedRun

    days = int(payload.get("days", 30))
    user_count = int(payload.get("user_count", 1))
    scenario = payload.get("scenario", "normal_load")
    profile_preset = payload.get("profile_preset", "intermediate")
    add_anomalies = bool(payload.get("add_anomalies", False))
    missing_data_rate = float(payload.get("missing_data_rate", 0.0))
    truncate_before = bool(payload.get("truncate_before", False))

    # Валидация
    valid_scenarios = {"normal_load", "overreaching", "recovery_phase", "injury_recovery"}
    valid_profiles = {"beginner", "intermediate", "advanced"}
    if scenario not in valid_scenarios:
        raise HTTPException(status_code=400, detail=f"Неверный scenario: {scenario}")
    if profile_preset not in valid_profiles:
        raise HTTPException(status_code=400, detail=f"Неверный profile_preset: {profile_preset}")
    days = max(1, min(365, days))
    user_count = max(1, min(10, user_count))

    gen = SeedGenerator(
        days=days,
        user_count=user_count,
        profile_preset=profile_preset,
        scenario=scenario,
        add_anomalies=add_anomalies,
        missing_data_rate=missing_data_rate,
        truncate_before=truncate_before,
    )

    try:
        sync_engine = create_engine(
            settings.database_url_sync,
            connect_args={"check_same_thread": False},
        )

        def _run_sync() -> dict[str, Any]:
            with SyncSession(sync_engine) as session:
                result = gen.generate(session)
                return {
                    "profiles_created": result.profiles_created,
                    "activities_created": result.activities_created,
                    "daily_facts_created": result.daily_facts_created,
                    "users": result.users,
                }

        loop = asyncio.get_event_loop()
        counts = await loop.run_in_executor(None, _run_sync)
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
    finally:
        try:
            sync_engine.dispose()
        except Exception:
            pass

    # Запись в журнал
    params_stored = {
        "scenario": scenario,
        "profile_preset": profile_preset,
        "days": days,
        "user_count": user_count,
        "add_anomalies": add_anomalies,
        "missing_data_rate": missing_data_rate,
        "truncate_before": truncate_before,
    }
    run_record = SeedRun(
        params=params_stored,
        admin_user=credentials.username,
        records_created=counts,
    )
    db.add(run_record)
    await db.commit()

    return {"status": "ok", **counts}


@router.post("/seed/preview", dependencies=[Depends(_require_admin)])
async def preview_seed(
    payload: dict[str, Any] = Body(default={}),
) -> dict[str, Any]:
    """Предпросмотр: ~5 записей без записи в БД.

    Body — те же поля, что и для /seed.
    """
    from scripts.seed_data import SeedGenerator

    days = max(1, min(365, int(payload.get("days", 30))))
    scenario = payload.get("scenario", "normal_load")
    profile_preset = payload.get("profile_preset", "intermediate")
    add_anomalies = bool(payload.get("add_anomalies", False))
    missing_data_rate = float(payload.get("missing_data_rate", 0.0))

    valid_scenarios = {"normal_load", "overreaching", "recovery_phase", "injury_recovery"}
    valid_profiles = {"beginner", "intermediate", "advanced"}
    if scenario not in valid_scenarios:
        raise HTTPException(status_code=400, detail=f"Неверный scenario: {scenario}")
    if profile_preset not in valid_profiles:
        raise HTTPException(status_code=400, detail=f"Неверный profile_preset: {profile_preset}")

    gen = SeedGenerator(
        days=days,
        scenario=scenario,
        profile_preset=profile_preset,
        add_anomalies=add_anomalies,
        missing_data_rate=missing_data_rate,
    )
    return gen.preview(count=5)


@router.get("/seed/runs", dependencies=[Depends(_require_admin)])
async def list_seed_runs(
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Журнал запусков SeedGenerator."""
    from app.models.seed_run import SeedRun

    runs = (await db.execute(
        select(SeedRun).order_by(SeedRun.created_at.desc()).limit(limit)
    )).scalars().all()

    return {
        "total": len(runs),
        "items": [
            {
                "id": r.id,
                "params": r.params,
                "records_created": r.records_created,
                "admin_user": r.admin_user,
                "created_at": r.created_at.isoformat(),
            }
            for r in runs
        ],
    }


# ---------------------------------------------------------------------------
# Semantic Memory (Issue #25)
# ---------------------------------------------------------------------------


@router.get("/memory", dependencies=[Depends(_require_admin)])
async def list_semantic_memory(
    user_id: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, Any]:
    """Список записей semantic_memory.

    Args:
        user_id: Фильтр по пользователю (опционально).
        limit: Максимум записей.
    """
    from app.services.semantic_memory import semantic_memory
    from app.services.vector_store import vector_store

    if not vector_store.available:
        return {"available": False, "items": [], "total": 0}

    records = semantic_memory.list_records(user_id=user_id, limit=limit)
    return {
        "available": True,
        "total": len(records),
        "items": [r.to_dict() for r in records],
    }


@router.delete("/memory", dependencies=[Depends(_require_admin)])
async def clear_semantic_memory(
    user_id: str | None = Query(None),
) -> dict[str, Any]:
    """Очистить semantic_memory (для пользователя или всё).

    Args:
        user_id: Если задан — только записи этого пользователя.
    """
    from app.services.semantic_memory import semantic_memory
    from app.services.vector_store import vector_store

    if not vector_store.available:
        return {"status": "skipped", "reason": "chromadb unavailable", "deleted": 0}

    deleted = semantic_memory.clear(user_id=user_id)
    return {"status": "ok", "deleted": deleted, "user_id": user_id}


@router.post("/test-llm", dependencies=[Depends(_require_admin)])
async def test_llm(
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    """Тест промпта через Ollama.

    Body: { "prompt": "...", "system_prompt": "..." (optional) }
    """
    from app.services.llm_service import ollama_client
    import time

    prompt = payload.get("prompt", "").strip()
    system_prompt = payload.get("system_prompt", "").strip() or None
    if not prompt:
        raise HTTPException(status_code=400, detail="Поле 'prompt' обязательно")

    try:
        start = time.monotonic()
        llm_resp = await ollama_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
        )
        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "status": "ok",
            "response": llm_resp.content,
            "model": llm_resp.model,
            "duration_ms": duration_ms,
            "prompt_length": llm_resp.prompt_length,
            "response_length": llm_resp.response_length,
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc), "response": None}
