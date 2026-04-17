"""Admin Panel — Jinja2 views (HTML страницы).

Роуты:
  GET /admin           → dashboard
  GET /admin/logs      → список логов
  GET /admin/logs/{id} → детали лога
  GET /admin/data      → data browser
  GET /admin/control   → agent control

HTMX-партиалы (фрагменты для data browser):
  GET /admin/data/activities-partial
  GET /admin/data/daily-facts-partial
  GET /admin/data/profiles-partial
  GET /admin/data/sessions-partial
  GET /admin/data/session-messages/{session_id}

Защита: HTTP Basic Auth (shared с api/admin.py).
"""

import secrets
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from sqlalchemy import and_, distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_db
from app.models.activity import Activity
from app.models.chat import ChatMessage, ChatSession
from app.models.daily_fact import DailyFact
from app.models.pipeline_log import PipelineLog
from app.models.user_profile import UserProfile

router = APIRouter(tags=["admin-ui"])
_security = HTTPBasic()
templates = Jinja2Templates(directory="app/admin/templates")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _require_admin(
    request: Request,
    credentials: HTTPBasicCredentials = Depends(_security),
) -> None:
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


def _base_ctx(active_page: str) -> dict[str, Any]:
    return {
        "active_page": active_page,
        "admin_user": settings.admin_username,
        "admin_pass": settings.admin_password,
    }


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@router.get("/admin", response_class=HTMLResponse, dependencies=[Depends(_require_admin)])
async def admin_dashboard(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    """Главная страница admin — статус системы и статистика."""
    from app.services.llm_service import ollama_client
    from datetime import datetime
    import os

    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_logs = (await db.execute(
        select(PipelineLog).where(PipelineLog.timestamp >= today_start)
    )).scalars().all()

    total_today = len(today_logs)
    avg_dur = sum(l.total_duration_ms or 0 for l in today_logs) // total_today if total_today else 0
    errors_today = sum(1 for l in today_logs if l.errors)

    intent_counts: dict[str, int] = {}
    for log in today_logs:
        k = log.intent or "unknown"
        intent_counts[k] = intent_counts.get(k, 0) + 1

    recent = (await db.execute(
        select(PipelineLog).order_by(PipelineLog.timestamp.desc()).limit(10)
    )).scalars().all()

    ollama_status = await ollama_client.health_check()

    uptime_str = "n/a"
    try:
        with open("/proc/uptime") as f:
            sec = float(f.read().split()[0])
        h, rem = divmod(int(sec), 3600)
        m, s = divmod(rem, 60)
        uptime_str = f"{h}h {m}m {s}s"
    except OSError:
        pass

    db_size = 0
    try:
        db_size = os.path.getsize(settings.db_path)
    except OSError:
        pass

    def log_to_dict(log: PipelineLog) -> dict:
        return {
            "id": log.id,
            "timestamp": log.timestamp.isoformat() if log.timestamp else None,
            "raw_query": log.raw_query,
            "intent": log.intent,
            "route": log.route,
            "total_duration_ms": log.total_duration_ms,
            "has_errors": bool(log.errors),
        }

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            **_base_ctx("dashboard"),
            "stats": {
                "today": {
                    "total_requests": total_today,
                    "avg_duration_ms": avg_dur,
                    "errors_count": errors_today,
                },
                "intent_distribution": intent_counts,
                "recent_requests": [log_to_dict(l) for l in recent],
                "db_size_bytes": db_size,
            },
            "status": {
                "ollama": ollama_status,
                "uptime": uptime_str,
                "config": {
                    "ollama_model": settings.ollama_model,
                    "ollama_host": settings.ollama_host,
                    "ollama_timeout": settings.ollama_timeout,
                    "log_level": settings.log_level,
                },
            },
        },
    )


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------

@router.get("/admin/logs", response_class=HTMLResponse, dependencies=[Depends(_require_admin)])
async def admin_logs(
    request: Request,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    intent: str | None = Query(None),
    route: str | None = Query(None),
    has_errors: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    """Список логов с фильтрами и пагинацией."""
    from datetime import datetime

    conditions = []
    if date_from:
        try:
            conditions.append(PipelineLog.timestamp >= datetime.fromisoformat(date_from))
        except ValueError:
            pass
    if date_to:
        try:
            dt_to = datetime.fromisoformat(date_to).replace(hour=23, minute=59, second=59)
            conditions.append(PipelineLog.timestamp <= dt_to)
        except ValueError:
            pass
    if intent:
        conditions.append(PipelineLog.intent == intent)
    if route:
        conditions.append(PipelineLog.route == route)
    if has_errors == "true":
        conditions.append(PipelineLog.errors.is_not(None))
    elif has_errors == "false":
        conditions.append(PipelineLog.errors.is_(None))

    where_clause = and_(*conditions) if conditions else True
    total = (await db.execute(
        select(func.count()).select_from(PipelineLog).where(where_clause)
    )).scalar() or 0

    logs = (await db.execute(
        select(PipelineLog).where(where_clause)
        .order_by(PipelineLog.timestamp.desc())
        .offset((page - 1) * per_page).limit(per_page)
    )).scalars().all()

    # Список уникальных intent и route для фильтров
    all_intents = (await db.execute(
        select(distinct(PipelineLog.intent)).where(PipelineLog.intent.is_not(None))
    )).scalars().all()
    all_routes = (await db.execute(
        select(distinct(PipelineLog.route)).where(PipelineLog.route.is_not(None))
    )).scalars().all()

    pages = (total + per_page - 1) // per_page if total > 0 else 0

    # Строка фильтров для пагинации
    filter_parts = []
    if date_from: filter_parts.append(f"date_from={date_from}")
    if date_to: filter_parts.append(f"date_to={date_to}")
    if intent: filter_parts.append(f"intent={intent}")
    if route: filter_parts.append(f"route={route}")
    if has_errors: filter_parts.append(f"has_errors={has_errors}")
    filter_qs = ("&" + "&".join(filter_parts)) if filter_parts else ""

    def log_dict(log: PipelineLog) -> dict:
        return {
            "id": log.id,
            "timestamp": log.timestamp.isoformat() if log.timestamp else None,
            "raw_query": log.raw_query,
            "intent": log.intent,
            "route": log.route,
            "safety_level": log.safety_level,
            "total_duration_ms": log.total_duration_ms,
            "has_errors": bool(log.errors),
        }

    return templates.TemplateResponse(
        "logs.html",
        {
            "request": request,
            **_base_ctx("logs"),
            "logs": [log_dict(l) for l in logs],
            "total": total,
            "page": page,
            "pages": pages,
            "per_page": per_page,
            "filter_qs": filter_qs,
            "filters": {
                "date_from": date_from,
                "date_to": date_to,
                "intent": intent,
                "route": route,
                "has_errors": has_errors,
            },
            "intents": sorted(all_intents),
            "routes": sorted(all_routes),
        },
    )


@router.get("/admin/logs/{request_id}", response_class=HTMLResponse, dependencies=[Depends(_require_admin)])
async def admin_log_detail(
    request: Request,
    request_id: str,
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    """Детальный просмотр одного лога."""
    result = await db.execute(select(PipelineLog).where(PipelineLog.id == request_id))
    log = result.scalar_one_or_none()
    if log is None:
        raise HTTPException(status_code=404, detail="Лог не найден")

    log_dict = {
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
        "response_text": log.response_text,
    }

    return templates.TemplateResponse(
        "log_detail.html",
        {"request": request, **_base_ctx("logs"), "log": log_dict},
    )


# ---------------------------------------------------------------------------
# Data Browser
# ---------------------------------------------------------------------------

@router.get("/admin/data", response_class=HTMLResponse, dependencies=[Depends(_require_admin)])
async def admin_data(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    """Data browser — вкладки с тренировками, метриками, профилями, сессиями."""
    activities_total = (await db.execute(select(func.count()).select_from(Activity))).scalar() or 0
    facts_total = (await db.execute(select(func.count()).select_from(DailyFact))).scalar() or 0
    profiles_total = (await db.execute(select(func.count()).select_from(UserProfile))).scalar() or 0
    sessions_total = (await db.execute(select(func.count()).select_from(ChatSession))).scalar() or 0

    # Данные по умолчанию — тренировки (первая вкладка)
    activities = (await db.execute(
        select(Activity).order_by(Activity.start_time.desc()).limit(20)
    )).scalars().all()

    sport_types = (await db.execute(
        select(distinct(Activity.sport_type)).order_by(Activity.sport_type)
    )).scalars().all()

    def act_dict(a: Activity) -> dict:
        return {
            "id": a.id,
            "user_id": a.user_id,
            "title": a.title,
            "sport_type": a.sport_type,
            "distance_meters": a.distance_meters,
            "duration_seconds": a.duration_seconds,
            "start_time": a.start_time.isoformat() if a.start_time else None,
            "calories": a.calories,
            "avg_heart_rate": a.avg_heart_rate,
        }

    return templates.TemplateResponse(
        "data.html",
        {
            "request": request,
            **_base_ctx("data"),
            "activities_total": activities_total,
            "facts_total": facts_total,
            "profiles_total": profiles_total,
            "sessions_total": sessions_total,
            # Default content (activities)
            "activities": [act_dict(a) for a in activities],
            "total": activities_total,
            "page": 1,
            "pages": max(1, (activities_total + 19) // 20),
            "sport_types": list(sport_types),
            "user_id": None,
            "sport_type": None,
            "date_from": None,
            "date_to": None,
        },
    )


# HTMX-партиал: тренировки
@router.get("/admin/data/activities-partial", response_class=HTMLResponse, dependencies=[Depends(_require_admin)])
async def data_activities_partial(
    request: Request,
    page: int = Query(1, ge=1),
    user_id: str | None = Query(None),
    sport_type: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    from datetime import datetime

    per_page = 20
    conditions = []
    if user_id: conditions.append(Activity.user_id == user_id)
    if sport_type: conditions.append(Activity.sport_type == sport_type)
    if date_from:
        try: conditions.append(Activity.start_time >= datetime.fromisoformat(date_from))
        except ValueError: pass
    if date_to:
        try:
            conditions.append(Activity.start_time <= datetime.fromisoformat(date_to).replace(hour=23, minute=59, second=59))
        except ValueError: pass

    where = and_(*conditions) if conditions else True
    total = (await db.execute(select(func.count()).select_from(Activity).where(where))).scalar() or 0
    activities = (await db.execute(
        select(Activity).where(where)
        .order_by(Activity.start_time.desc())
        .offset((page - 1) * per_page).limit(per_page)
    )).scalars().all()

    sport_types = (await db.execute(
        select(distinct(Activity.sport_type)).order_by(Activity.sport_type)
    )).scalars().all()

    def act_dict(a: Activity) -> dict:
        return {
            "id": a.id, "user_id": a.user_id, "title": a.title, "sport_type": a.sport_type,
            "distance_meters": a.distance_meters, "duration_seconds": a.duration_seconds,
            "start_time": a.start_time.isoformat() if a.start_time else None,
            "calories": a.calories, "avg_heart_rate": a.avg_heart_rate,
        }

    return templates.TemplateResponse(
        "partials/activities_table.html",
        {
            "request": request,
            "activities": [act_dict(a) for a in activities],
            "total": total,
            "page": page,
            "pages": max(1, (total + per_page - 1) // per_page),
            "sport_types": list(sport_types),
            "user_id": user_id,
            "sport_type": sport_type,
            "date_from": date_from,
            "date_to": date_to,
        },
    )


# HTMX-партиал: дневные метрики
@router.get("/admin/data/daily-facts-partial", response_class=HTMLResponse, dependencies=[Depends(_require_admin)])
async def data_daily_facts_partial(
    request: Request,
    page: int = Query(1, ge=1),
    user_id: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    per_page = 20
    conditions = []
    if user_id: conditions.append(DailyFact.user_id == user_id)
    if date_from: conditions.append(DailyFact.iso_date >= date_from)
    if date_to: conditions.append(DailyFact.iso_date <= date_to)

    where = and_(*conditions) if conditions else True
    total = (await db.execute(select(func.count()).select_from(DailyFact).where(where))).scalar() or 0
    facts = (await db.execute(
        select(DailyFact).where(where)
        .order_by(DailyFact.iso_date.desc())
        .offset((page - 1) * per_page).limit(per_page)
    )).scalars().all()

    def fact_dict(f: DailyFact) -> dict:
        return {
            "id": f.id, "user_id": f.user_id, "iso_date": f.iso_date,
            "steps": f.steps, "calories_kcal": f.calories_kcal,
            "recovery_score": f.recovery_score, "hrv_rmssd_milli": f.hrv_rmssd_milli,
            "resting_heart_rate": f.resting_heart_rate, "spo2_percentage": f.spo2_percentage,
            "sleep_total_in_bed_milli": f.sleep_total_in_bed_milli, "water_liters": f.water_liters,
        }

    return templates.TemplateResponse(
        "partials/daily_facts_table.html",
        {
            "request": request,
            "facts": [fact_dict(f) for f in facts],
            "total": total,
            "page": page,
            "pages": max(1, (total + per_page - 1) // per_page),
            "user_id": user_id,
            "date_from": date_from,
            "date_to": date_to,
        },
    )


# HTMX-партиал: профили
@router.get("/admin/data/profiles-partial", response_class=HTMLResponse, dependencies=[Depends(_require_admin)])
async def data_profiles_partial(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    profiles = (await db.execute(
        select(UserProfile).order_by(UserProfile.created_at.desc())
    )).scalars().all()

    def profile_dict(p: UserProfile) -> dict:
        return {
            "id": p.id, "user_id": p.user_id, "name": p.name, "age": p.age,
            "weight_kg": p.weight_kg, "height_cm": p.height_cm,
            "experience_level": p.experience_level, "training_goals": p.training_goals,
            "max_heart_rate": p.max_heart_rate,
        }

    return templates.TemplateResponse(
        "partials/profiles_table.html",
        {"request": request, "profiles": [profile_dict(p) for p in profiles]},
    )


# HTMX-партиал: сессии
@router.get("/admin/data/sessions-partial", response_class=HTMLResponse, dependencies=[Depends(_require_admin)])
async def data_sessions_partial(
    request: Request,
    page: int = Query(1, ge=1),
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    per_page = 20
    total = (await db.execute(select(func.count()).select_from(ChatSession))).scalar() or 0
    sessions = (await db.execute(
        select(ChatSession).order_by(ChatSession.updated_at.desc())
        .offset((page - 1) * per_page).limit(per_page)
    )).scalars().all()

    def sess_dict(s: ChatSession) -> dict:
        return {
            "id": s.id, "user_id": s.user_id,
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "updated_at": s.updated_at.isoformat() if s.updated_at else None,
        }

    return templates.TemplateResponse(
        "partials/sessions_table.html",
        {
            "request": request,
            "sessions": [sess_dict(s) for s in sessions],
            "total": total,
            "page": page,
            "pages": max(1, (total + per_page - 1) // per_page),
        },
    )


# HTMX-партиал: сообщения сессии
@router.get("/admin/data/session-messages/{session_id}", response_class=HTMLResponse, dependencies=[Depends(_require_admin)])
async def data_session_messages(
    request: Request,
    session_id: str,
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    messages = (await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.order_index)
    )).scalars().all()

    def msg_dict(m: ChatMessage) -> dict:
        return {
            "id": m.id, "role": m.role, "content": m.content,
            "created_at": m.created_at.isoformat() if m.created_at else None,
            "order_index": m.order_index,
        }

    return templates.TemplateResponse(
        "partials/session_messages.html",
        {
            "request": request,
            "session_id": session_id,
            "messages": [msg_dict(m) for m in messages],
        },
    )


# ---------------------------------------------------------------------------
# Seed Generator v2 (Issue #33)
# ---------------------------------------------------------------------------

@router.get("/admin/seed", response_class=HTMLResponse, dependencies=[Depends(_require_admin)])
async def admin_seed(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    """Страница управления Seed Generator v2."""
    from app.models.seed_run import SeedRun

    runs = (await db.execute(
        select(SeedRun).order_by(SeedRun.created_at.desc()).limit(20)
    )).scalars().all()

    def run_dict(r: SeedRun) -> dict:
        return {
            "id": r.id,
            "params": r.params,
            "records_created": r.records_created,
            "admin_user": r.admin_user,
            "created_at": r.created_at.isoformat(),
        }

    return templates.TemplateResponse(
        "seed.html",
        {
            "request": request,
            **_base_ctx("seed"),
            "recent_runs": [run_dict(r) for r in runs],
        },
    )


# ---------------------------------------------------------------------------
# Agent Control
# ---------------------------------------------------------------------------

@router.get("/admin/control", response_class=HTMLResponse, dependencies=[Depends(_require_admin)])
async def admin_control(
    request: Request,
) -> HTMLResponse:
    """Страница управления агентом — тест LLM, конфиг, seed data."""
    from app.services.llm_service import ollama_client

    models = await ollama_client.list_models()

    return templates.TemplateResponse(
        "control.html",
        {
            "request": request,
            **_base_ctx("control"),
            "models": models,
            "config": {
                "ollama_model": settings.ollama_model,
                "ollama_host": settings.ollama_host,
                "ollama_timeout": settings.ollama_timeout,
                "log_level": settings.log_level,
            },
        },
    )
