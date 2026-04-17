"""Stage Tracker — отслеживание стадий пайплайна для одного запроса.

Phase 2, Issue #31. Фиксирует start/end каждой стадии, записывает в trace list,
публикует stage_start / stage_end события в StageEventBus.

Стадии пайплайна:
  context_build, intent_stage1, intent_stage2, safety, routing,
  template_step_N, planner_iter_N, response_gen, memory_update
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator

from app.pipeline.stage_events import StageEvent, StageEventBus, stage_event_bus

logger = logging.getLogger(__name__)


class StageTracker:
    """Трекер стадий пайплайна для одного запроса.

    Использование:
        tracker = StageTracker(request_id)

        async with tracker.track_stage("intent_stage1"):
            result = await intent_detector.detect(...)

        # Получить список трейс-записей:
        trace = tracker.trace  # [{stage, start_ms, duration_ms, metadata?}, ...]
    """

    def __init__(
        self,
        request_id: str,
        event_bus: StageEventBus | None = None,
    ) -> None:
        self.request_id = request_id
        self._event_bus = event_bus or stage_event_bus
        self._trace: list[dict[str, Any]] = []
        self._request_start_ms = time.monotonic() * 1000

    @property
    def trace(self) -> list[dict[str, Any]]:
        """Снимок списка трейс-записей."""
        return list(self._trace)

    @asynccontextmanager
    async def track_stage(
        self,
        stage: str,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[None]:
        """Контекст-менеджер отслеживания стадии.

        Публикует stage_start в начале и stage_end в конце (даже при исключении).
        Добавляет запись с длительностью в trace list.
        """
        stage_start_abs = time.monotonic() * 1000
        start_offset_ms = int(stage_start_abs - self._request_start_ms)

        await self._event_bus.publish(
            self.request_id,
            StageEvent(
                type="stage_start",
                request_id=self.request_id,
                stage=stage,
                timestamp=datetime.utcnow(),
            ),
        )

        try:
            yield
        finally:
            duration_ms = int(time.monotonic() * 1000 - stage_start_abs)

            entry: dict[str, Any] = {
                "stage": stage,
                "start_ms": start_offset_ms,
                "duration_ms": duration_ms,
            }
            if metadata:
                entry["metadata"] = metadata
            self._trace.append(entry)

            await self._event_bus.publish(
                self.request_id,
                StageEvent(
                    type="stage_end",
                    request_id=self.request_id,
                    stage=stage,
                    duration_ms=duration_ms,
                    timestamp=datetime.utcnow(),
                ),
            )
