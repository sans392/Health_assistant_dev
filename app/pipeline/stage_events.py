"""Stage Events pub/sub — механизм передачи событий пайплайна в WebSocket.

Phase 2, Issue #31. Простой in-memory pub/sub через asyncio.Queue.
Подписчик получает события по request_id; итерация завершается на type='done'/'error'.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from typing import AsyncIterator, Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class StageEvent(BaseModel):
    """Событие стадии пайплайна."""

    type: Literal["stage_start", "stage_end", "token", "error", "done"]
    request_id: str
    stage: str | None = None
    duration_ms: int | None = None
    token: str | None = None
    message: str | None = None  # для type=done: полный ответ
    timestamp: datetime


class StageEventBus:
    """In-memory pub/sub для событий пайплайна по request_id.

    Использование:
        # Публиковать (из orchestrator):
        await event_bus.publish(request_id, StageEvent(...))

        # Подписаться (из WebSocket handler):
        async for event in event_bus.subscribe(request_id):
            await ws.send_json(event.model_dump(mode="json"))
    """

    def __init__(self) -> None:
        self._queues: dict[str, list[asyncio.Queue]] = defaultdict(list)

    async def publish(self, request_id: str, event: StageEvent) -> None:
        """Опубликовать событие всем подписчикам request_id."""
        for q in self._queues.get(request_id, []):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(
                    "StageEventBus: очередь переполнена request_id=%s stage=%s",
                    request_id,
                    event.stage,
                )

    async def subscribe(
        self,
        request_id: str,
        maxsize: int = 100,
    ) -> AsyncIterator[StageEvent]:
        """Подписаться на события request_id.

        Итерация завершается при получении type='done'/'error' или при close().
        """
        q: asyncio.Queue[StageEvent | None] = asyncio.Queue(maxsize=maxsize)
        self._queues[request_id].append(q)
        try:
            while True:
                event = await q.get()
                if event is None:
                    break
                yield event
                if event.type in ("done", "error"):
                    break
        finally:
            try:
                self._queues[request_id].remove(q)
            except ValueError:
                pass
            if not self._queues.get(request_id):
                self._queues.pop(request_id, None)

    def publish_nowait(self, request_id: str, event: StageEvent) -> None:
        """Синхронная публикация события (put_nowait) — для on_token callback из sync-контекста."""
        for q in self._queues.get(request_id, []):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(
                    "StageEventBus: очередь переполнена request_id=%s stage=%s",
                    request_id,
                    event.stage,
                )

    def close(self, request_id: str) -> None:
        """Закрыть все подписки request_id (отправить sentinel None)."""
        for q in list(self._queues.get(request_id, [])):
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass


# Глобальный синглтон
stage_event_bus = StageEventBus()
