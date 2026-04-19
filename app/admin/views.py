"""Admin Panel — Jinja2 views (HTML страницы).

Роуты:
  GET /admin              → dashboard
  GET /admin/logs         → список логов
  GET /admin/logs/{id}    → детали лога
  GET /admin/data         → data browser
  GET /admin/control      → agent control
  GET /admin/seed         → seed generator
  GET /admin/llm          → LLM role config (issue #35)
  GET /admin/knowledge    → knowledge base browser (issue #35)
  GET /admin/memory       → semantic memory browser (issue #35)
  GET /admin/diagnostics  → diagnostics dashboard (issue #35)

HTMX-партиалы (фрагменты для data browser):
  GET /admin/data/activities-partial
  GET /admin/data/daily-facts-partial
  GET /admin/data/profiles-partial
  GET /admin/data/sessions-partial
  GET /admin/data/session-messages/{session_id}

Защита: HTTP Basic Auth (shared с api/admin.py).
"""

import json
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
from app.models.tool_call import ToolCall
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
    model_role: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    """Список логов с фильтрами и пагинацией."""
    from datetime import datetime
    from app.models.llm_call import LLMCall

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
    if model_role:
        subq = select(LLMCall.request_id).where(LLMCall.role == model_role).distinct()
        conditions.append(PipelineLog.id.in_(subq))

    where_clause = and_(*conditions) if conditions else True
    total = (await db.execute(
        select(func.count()).select_from(PipelineLog).where(where_clause)
    )).scalar() or 0

    logs = (await db.execute(
        select(PipelineLog).where(where_clause)
        .order_by(PipelineLog.timestamp.desc())
        .offset((page - 1) * per_page).limit(per_page)
    )).scalars().all()

    all_intents = (await db.execute(
        select(distinct(PipelineLog.intent)).where(PipelineLog.intent.is_not(None))
    )).scalars().all()
    all_routes = (await db.execute(
        select(distinct(PipelineLog.route)).where(PipelineLog.route.is_not(None))
    )).scalars().all()

    pages = (total + per_page - 1) // per_page if total > 0 else 0

    filter_parts = []
    if date_from: filter_parts.append(f"date_from={date_from}")
    if date_to: filter_parts.append(f"date_to={date_to}")
    if intent: filter_parts.append(f"intent={intent}")
    if route: filter_parts.append(f"route={route}")
    if has_errors: filter_parts.append(f"has_errors={has_errors}")
    if model_role: filter_parts.append(f"model_role={model_role}")
    filter_qs = ("&" + "&".join(filter_parts)) if filter_parts else ""

    def log_dict(log: PipelineLog) -> dict:
        stage_trace = log.stage_trace or []
        rag_chunks = log.rag_chunks_used or []
        return {
            "id": log.id,
            "timestamp": log.timestamp.isoformat() if log.timestamp else None,
            "raw_query": log.raw_query,
            "intent": log.intent,
            "route": log.route,
            "safety_level": log.safety_level,
            "total_duration_ms": log.total_duration_ms,
            "llm_calls_count": log.llm_calls_count or 0,
            "stages_count": len(stage_trace),
            "rag_chunks_count": len(rag_chunks),
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
                "model_role": model_role,
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
    """Детальный просмотр одного лога — 5 вкладок (issue #34)."""
    from app.models.llm_call import LLMCall

    result = await db.execute(select(PipelineLog).where(PipelineLog.id == request_id))
    log = result.scalar_one_or_none()
    if log is None:
        raise HTTPException(status_code=404, detail="Лог не найден")

    # LLM calls для вкладки LLM Calls
    llm_calls_rows = (await db.execute(
        select(LLMCall)
        .where(LLMCall.request_id == request_id)
        .order_by(LLMCall.timestamp)
    )).scalars().all()

    # Tool calls для вкладки Tool Results
    tool_calls_rows = (await db.execute(
        select(ToolCall)
        .where(ToolCall.request_id == request_id)
        .order_by(ToolCall.timestamp)
    )).scalars().all()

    total_ms = log.total_duration_ms or 1
    stage_trace = log.stage_trace or []
    for stage in stage_trace:
        dur = stage.get("duration_ms") or 0
        start = stage.get("start_ms") or 0
        stage["width_pct"] = round(dur / total_ms * 100, 1) if total_ms else 0
        stage["offset_pct"] = round(start / total_ms * 100, 1) if total_ms else 0

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
        "tools_called": log.tools_called or [],
        "modules_used": log.modules_used or [],
        "llm_model_used": log.llm_model_used,
        "llm_calls_count": log.llm_calls_count,
        "total_duration_ms": log.total_duration_ms,
        "response_length": log.response_length,
        "errors": log.errors,
        "response_text": log.response_text,
        "stage_trace": stage_trace,
        "llm_role_usage": log.llm_role_usage or {},
        "rag_chunks_used": log.rag_chunks_used or [],
    }

    llm_calls_list = [
        {
            "id": c.id,
            "role": c.role,
            "model": c.model,
            "endpoint": c.endpoint,
            "stream": c.stream,
            "http_status": c.http_status,
            "duration_ms": c.duration_ms,
            "prompt_length": c.prompt_length,
            "response_length": c.response_length,
            "prompt_preview": (c.prompt or "")[:200],
            "response_preview": (c.response or "")[:200],
            "prompt": c.prompt or "",
            "response": c.response or "",
            "request_body_json": (
                json.dumps(c.request_body, ensure_ascii=False, indent=2)
                if c.request_body is not None else ""
            ),
            "response_body_json": (
                json.dumps(c.response_body, ensure_ascii=False, indent=2)
                if c.response_body is not None else ""
            ),
            "error": c.error,
            "iteration": c.iteration,
            "timestamp": c.timestamp.isoformat() if c.timestamp else None,
        }
        for c in llm_calls_rows
    ]

    tool_calls_list = [
        {
            "id": c.id,
            "name": c.name,
            "source": c.source,
            "iteration": c.iteration,
            "step_id": c.step_id,
            "success": c.success,
            "error": c.error,
            "duration_ms": c.duration_ms,
            "args_json": (
                json.dumps(c.args, ensure_ascii=False, indent=2, default=str)
                if c.args is not None else ""
            ),
            "result_json": (
                json.dumps(c.result, ensure_ascii=False, indent=2, default=str)
                if c.result is not None else ""
            ),
            "timestamp": c.timestamp.isoformat() if c.timestamp else None,
        }
        for c in tool_calls_rows
    ]

    return templates.TemplateResponse(
        "log_detail.html",
        {
            "request": request,
            **_base_ctx("logs"),
            "log": log_dict,
            "llm_calls": llm_calls_list,
            "tool_calls": tool_calls_list,
        },
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

    import json as _json

    def profile_dict(p: UserProfile) -> dict:
        return {
            "id": p.id,
            "user_id": p.user_id,
            "name": p.name,
            "age": p.age,
            "weight_kg": p.weight_kg,
            "height_cm": p.height_cm,
            "gender": p.gender,
            "experience_level": p.experience_level,
            "training_goals": p.training_goals or [],
            "max_heart_rate": p.max_heart_rate,
            "resting_heart_rate": p.resting_heart_rate,
            "injuries": p.injuries or [],
            "chronic_conditions": p.chronic_conditions or [],
            "preferred_sports": p.preferred_sports or [],
        }

    return templates.TemplateResponse(
        "partials/profiles_table.html",
        {
            "request": request,
            "profiles": [profile_dict(p) for p in profiles],
            "admin_user": settings.admin_username,
            "admin_pass": settings.admin_password,
        },
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


# ---------------------------------------------------------------------------
# LLM Config (Issue #35)
# ---------------------------------------------------------------------------

@router.get("/admin/llm", response_class=HTMLResponse, dependencies=[Depends(_require_admin)])
async def admin_llm(
    request: Request,
) -> HTMLResponse:
    """LLM Role Configuration страница."""
    from app.services.llm_registry import llm_registry, ALL_ROLES
    from app.services.llm_service import ollama_client

    try:
        models = await ollama_client.list_models()
    except Exception:
        models = []

    roles_status = await llm_registry.health_check()

    roles = []
    for role in ALL_ROLES:
        info = roles_status.get(role, {})
        roles.append({
            "role": role,
            "model": info.get("model", llm_registry.get_model(role)),
            "model_loaded": info.get("model_loaded", False),
        })

    return templates.TemplateResponse(
        "llm.html",
        {
            "request": request,
            **_base_ctx("llm"),
            "roles": roles,
            "available_models": models,
        },
    )


# ---------------------------------------------------------------------------
# Knowledge Base Browser (Issue #35)
# ---------------------------------------------------------------------------

@router.get("/admin/knowledge", response_class=HTMLResponse, dependencies=[Depends(_require_admin)])
async def admin_knowledge(
    request: Request,
    category: str | None = Query(None),
    sport_type: str | None = Query(None),
    confidence: str | None = Query(None),
    q: str | None = Query(None),
    page: int = Query(1, ge=1),
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    """Knowledge Base Browser страница."""
    from app.models.rag_chunk import RAGChunk
    from sqlalchemy import distinct as _distinct

    per_page = 20
    conditions = []
    if category:
        conditions.append(RAGChunk.category == category)
    if sport_type:
        conditions.append(RAGChunk.sport_type == sport_type)
    if confidence:
        conditions.append(RAGChunk.confidence == confidence)
    if q:
        conditions.append(RAGChunk.text.ilike(f"%{q}%"))

    where = and_(*conditions) if conditions else True
    total = (await db.execute(
        select(func.count()).select_from(RAGChunk).where(where)
    )).scalar() or 0

    chunks = (await db.execute(
        select(RAGChunk).where(where)
        .order_by(RAGChunk.created_at.desc())
        .offset((page - 1) * per_page).limit(per_page)
    )).scalars().all()

    all_categories = (await db.execute(
        select(_distinct(RAGChunk.category)).where(RAGChunk.category.is_not(None))
    )).scalars().all()

    all_sport_types = (await db.execute(
        select(_distinct(RAGChunk.sport_type)).where(RAGChunk.sport_type.is_not(None))
    )).scalars().all()

    pages = (total + per_page - 1) // per_page if total > 0 else 0

    def chunk_dict(c: RAGChunk) -> dict:
        return {
            "id": c.id,
            "category": c.category,
            "source": c.source,
            "confidence": c.confidence,
            "sport_type": c.sport_type,
            "experience_level": c.experience_level,
            "text_preview": c.text[:100],
            "embedding_id": c.embedding_id,
            "created_at": c.created_at.isoformat() if c.created_at else None,
        }

    return templates.TemplateResponse(
        "knowledge.html",
        {
            "request": request,
            **_base_ctx("knowledge"),
            "chunks": [chunk_dict(c) for c in chunks],
            "total": total,
            "page": page,
            "pages": pages,
            "filters": {
                "category": category,
                "sport_type": sport_type,
                "confidence": confidence,
                "q": q,
            },
            "all_categories": sorted(all_categories),
            "all_sport_types": sorted(all_sport_types),
        },
    )


# ---------------------------------------------------------------------------
# Semantic Memory Browser (Issue #35)
# ---------------------------------------------------------------------------

@router.get("/admin/memory", response_class=HTMLResponse, dependencies=[Depends(_require_admin)])
async def admin_memory(
    request: Request,
    user_id: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    """Semantic Memory Browser страница."""
    from app.services.semantic_memory import semantic_memory
    from app.services.vector_store import vector_store
    from app.models.user_profile import UserProfile

    # Все user_id для dropdown
    all_users = (await db.execute(
        select(UserProfile.user_id).order_by(UserProfile.user_id)
    )).scalars().all()

    records = []
    total = 0
    available = vector_store.available

    if available:
        raw_records = semantic_memory.list_records(user_id=user_id, limit=200)
        records = [r.to_dict() for r in raw_records]
        total = len(records)

    return templates.TemplateResponse(
        "memory.html",
        {
            "request": request,
            **_base_ctx("memory"),
            "records": records,
            "total": total,
            "available": available,
            "selected_user_id": user_id,
            "all_users": list(all_users),
        },
    )


# ---------------------------------------------------------------------------
# Diagnostics (Issue #35)
# ---------------------------------------------------------------------------

@router.get("/admin/diagnostics", response_class=HTMLResponse, dependencies=[Depends(_require_admin)])
async def admin_diagnostics(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    """Diagnostics Dashboard страница."""
    import os
    from app.services.llm_service import ollama_client
    from app.services.llm_registry import llm_registry
    from app.services.vector_store import vector_store
    from app.models.rag_chunk import RAGChunk
    from app.models.llm_call import LLMCall

    # Ollama
    try:
        ollama_ok = await ollama_client.health_check()
    except Exception:
        ollama_ok = {"available": False}
    roles_status = await llm_registry.health_check()

    # ChromaDB
    chroma_status = vector_store.health_check()
    chroma_dir_ok = os.path.isdir(settings.chroma_path)

    # DB counts
    from app.models.activity import Activity
    from app.models.daily_fact import DailyFact
    from app.models.user_profile import UserProfile
    from app.models.chat import ChatSession

    db_counts: dict[str, int] = {}
    for model_cls, key in [
        (PipelineLog, "pipeline_logs"),
        (Activity, "activities"),
        (DailyFact, "daily_facts"),
        (UserProfile, "profiles"),
        (ChatSession, "chat_sessions"),
        (RAGChunk, "rag_chunks"),
        (LLMCall, "llm_calls"),
    ]:
        try:
            db_counts[key] = (
                await db.execute(select(func.count()).select_from(model_cls))
            ).scalar() or 0
        except Exception:
            db_counts[key] = -1

    db_size_bytes = 0
    try:
        db_size_bytes = os.path.getsize(settings.db_path)
    except OSError:
        pass

    # Runtime метрики: последние 100 логов
    recent_logs = (await db.execute(
        select(PipelineLog).order_by(PipelineLog.timestamp.desc()).limit(100)
    )).scalars().all()

    total_logs = len(recent_logs)
    avg_duration = (
        sum(lo.total_duration_ms or 0 for lo in recent_logs) // total_logs
        if total_logs else 0
    )
    error_count = sum(1 for lo in recent_logs if lo.errors)
    error_rate = round(error_count / total_logs * 100, 1) if total_logs else 0.0

    stage_durations: dict[str, list[int]] = {}
    for lo in recent_logs:
        for stage in (lo.stage_trace or []):
            name = stage.get("stage", "unknown")
            dur = stage.get("duration_ms") or 0
            stage_durations.setdefault(name, []).append(dur)
    avg_stage_ms = {
        stage: sum(vals) // len(vals)
        for stage, vals in stage_durations.items()
        if vals
    }

    return templates.TemplateResponse(
        "diagnostics.html",
        {
            "request": request,
            **_base_ctx("diagnostics"),
            "ollama": {
                "status": ollama_ok,
                "roles": roles_status,
            },
            "chromadb": {
                **chroma_status,
                "persistent_dir_ok": chroma_dir_ok,
                "path": settings.chroma_path,
            },
            "database": {
                "size_bytes": db_size_bytes,
                "counts": db_counts,
            },
            "runtime": {
                "last_100_requests": total_logs,
                "avg_duration_ms": avg_duration,
                "error_count": error_count,
                "error_rate_pct": error_rate,
                "avg_stage_ms": avg_stage_ms,
            },
        },
    )
