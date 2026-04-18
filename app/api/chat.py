"""Chat API — WebSocket endpoint и REST-методы для работы с сессиями.

WebSocket: WS /ws/chat/{session_id}
REST:
  POST /api/chat/sessions           — создать сессию
  GET  /api/chat/sessions           — список сессий
  GET  /api/chat/sessions/{id}/messages — история сообщений
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models.chat import ChatMessage, ChatSession
from app.pipeline.orchestrator import pipeline_orchestrator
from app.pipeline.stage_events import StageEvent, stage_event_bus
from app.services.logging_service import logging_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


# ---------------------------------------------------------------------------
# REST — управление сессиями
# ---------------------------------------------------------------------------

@router.post("/api/chat/sessions")
async def create_session(
    user_id: str = Query("user_1", description="ID пользователя"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Создать новую чат-сессию."""
    session = ChatSession(id=str(uuid.uuid4()), user_id=user_id)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    logger.info("Создана новая сессия: id=%s user_id=%s", session.id, session.user_id)
    return {
        "id": session.id,
        "user_id": session.user_id,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
    }


@router.get("/api/chat/sessions")
async def list_sessions(
    user_id: str | None = Query(None, description="Фильтр по user_id"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Список сессий (все или конкретного пользователя)."""
    stmt = select(ChatSession).order_by(ChatSession.updated_at.desc())
    if user_id:
        stmt = stmt.where(ChatSession.user_id == user_id)
    result = await db.execute(stmt)
    sessions = result.scalars().all()

    return {
        "items": [
            {
                "id": s.id,
                "user_id": s.user_id,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
            }
            for s in sessions
        ]
    }


@router.get("/api/chat/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """История сообщений конкретной сессии (в порядке order_index)."""
    # Проверяем, что сессия существует
    stmt = select(ChatSession).where(ChatSession.id == session_id)
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена")

    msgs_stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.order_index)
    )
    msgs_result = await db.execute(msgs_stmt)
    messages = msgs_result.scalars().all()

    return {
        "session_id": session_id,
        "user_id": session.user_id,
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
# WebSocket — чат в реальном времени
# ---------------------------------------------------------------------------

@router.websocket("/ws/chat/{session_id}")
async def websocket_chat(
    websocket: WebSocket,
    session_id: str,
    user_id: str = Query("user_1", description="ID пользователя"),
    db: AsyncSession = Depends(get_db),
) -> None:
    """WebSocket endpoint для чата.

    При подключении:
    - Создаёт сессию, если не существует.
    - Отправляет историю предыдущих сообщений.

    При получении сообщения:
    - Запускает PipelineOrchestrator.process_query().
    - Сохраняет сообщения (user + assistant) в БД.
    - Отправляет ответ клиенту в виде JSON.
    """
    await websocket.accept()
    logger.info("WS подключение: session_id=%s user_id=%s", session_id, user_id)

    # Создаём или возобновляем сессию
    stmt = select(ChatSession).where(ChatSession.id == session_id)
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()
    if session is None:
        session = ChatSession(id=session_id, user_id=user_id)
        db.add(session)
        await db.commit()
        logger.info("WS: создана новая сессия %s", session_id)

    # Загружаем и отправляем историю сообщений
    msgs_stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.order_index)
    )
    msgs_result = await db.execute(msgs_stmt)
    history = msgs_result.scalars().all()

    if history:
        await websocket.send_text(json.dumps({
            "type": "history",
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "order_index": m.order_index,
                }
                for m in history
            ],
        }, ensure_ascii=False))

    # Основной цикл — каждое сообщение запускает пайплайн и транслирует stage events
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
                query = data.get("message", "").strip()
            except (json.JSONDecodeError, AttributeError):
                query = raw.strip()

            if not query:
                continue

            logger.info("WS: получено сообщение | session=%s len=%d", session_id, len(query))

            request_id = str(uuid.uuid4())
            pipeline_result = None
            pipeline_exc: Exception | None = None

            def on_token(token: str) -> None:
                """Публикует token-событие синхронно из callback response_generator."""
                stage_event_bus.publish_nowait(
                    request_id,
                    StageEvent(
                        type="token",
                        request_id=request_id,
                        token=token,
                        timestamp=datetime.utcnow(),
                    ),
                )

            async def run_pipeline() -> None:
                nonlocal pipeline_result, pipeline_exc
                try:
                    pipeline_result = await pipeline_orchestrator.process_query(
                        user_id=user_id,
                        session_id=session_id,
                        raw_query=query,
                        db=db,
                        on_token=on_token,
                        request_id=request_id,
                    )
                except Exception as exc:
                    logger.error(
                        "WS: ошибка пайплайна | session=%s error=%s", session_id, exc, exc_info=True,
                    )
                    pipeline_exc = exc
                finally:
                    # Публикуем done/error чтобы завершить подписку
                    if pipeline_exc is not None:
                        await stage_event_bus.publish(
                            request_id,
                            StageEvent(
                                type="error",
                                request_id=request_id,
                                message=str(pipeline_exc),
                                timestamp=datetime.utcnow(),
                            ),
                        )
                    else:
                        await stage_event_bus.publish(
                            request_id,
                            StageEvent(
                                type="done",
                                request_id=request_id,
                                message=pipeline_result.response_text if pipeline_result else "",
                                timestamp=datetime.utcnow(),
                            ),
                        )

            pipeline_task = asyncio.create_task(run_pipeline())

            # Пересылаем stage events клиенту (loop завершается при type=done/error)
            try:
                async for event in stage_event_bus.subscribe(request_id):
                    if event.type == "stage_start":
                        await websocket.send_text(json.dumps({
                            "type": "stage_start",
                            "stage": event.stage,
                            "request_id": request_id,
                        }, ensure_ascii=False))
                    elif event.type == "stage_end":
                        await websocket.send_text(json.dumps({
                            "type": "stage_end",
                            "stage": event.stage,
                            "duration_ms": event.duration_ms,
                        }, ensure_ascii=False))
                    elif event.type == "token":
                        await websocket.send_text(json.dumps({
                            "type": "token",
                            "token": event.token,
                        }, ensure_ascii=False))
                    # type=done/error breaks the async generator automatically
            except Exception as fwd_exc:
                logger.error(
                    "WS: ошибка пересылки событий | session=%s error=%s", session_id, fwd_exc,
                )
                pipeline_task.cancel()
                continue

            await pipeline_task

            if pipeline_exc is not None:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Произошла ошибка при обработке запроса. Попробуйте ещё раз.",
                }, ensure_ascii=False))
            elif pipeline_result is not None:
                try:
                    await logging_service.log_pipeline_request(
                        user_id=user_id,
                        session_id=session_id,
                        result=pipeline_result,
                        db=db,
                    )
                except Exception as log_exc:
                    logger.warning("WS: ошибка логирования | %s", log_exc)

                await websocket.send_text(json.dumps({
                    "type": "done",
                    "message": pipeline_result.response_text,
                    "request_id": request_id,
                    "debug": {
                        "intent": pipeline_result.intent,
                        "intent_confidence": pipeline_result.intent_confidence,
                        "entities": pipeline_result.entities,
                        "route": pipeline_result.route,
                        "safety_level": pipeline_result.safety_level,
                        "fast_path": pipeline_result.fast_path,
                        "blocked": pipeline_result.blocked,
                        "tools_called": pipeline_result.tools_called,
                        "modules_used": pipeline_result.modules_used,
                        "duration_ms": pipeline_result.duration_ms,
                        "errors": pipeline_result.errors,
                        "stage_trace": pipeline_result.stage_trace,
                        "llm_role_usage": pipeline_result.llm_role_usage,
                        "llm_calls_detail": pipeline_result.llm_calls_detail,
                    },
                }, ensure_ascii=False))

    except WebSocketDisconnect:
        logger.info("WS: клиент отключился | session=%s user=%s", session_id, user_id)
