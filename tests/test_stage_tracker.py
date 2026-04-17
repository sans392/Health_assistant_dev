"""Тесты StageTracker — отслеживание стадий пайплайна."""

from __future__ import annotations

import asyncio

import pytest

from app.pipeline.stage_events import StageEvent, StageEventBus
from app.pipeline.stage_tracker import StageTracker


@pytest.mark.asyncio
async def test_trace_recorded_after_stage() -> None:
    """После track_stage запись появляется в trace."""
    tracker = StageTracker("req-1", event_bus=StageEventBus())

    async with tracker.track_stage("context_build"):
        await asyncio.sleep(0)

    trace = tracker.trace
    assert len(trace) == 1
    assert trace[0]["stage"] == "context_build"
    assert trace[0]["duration_ms"] >= 0
    assert "start_ms" in trace[0]


@pytest.mark.asyncio
async def test_multiple_stages_ordered() -> None:
    """Несколько стадий записываются в правильном порядке."""
    tracker = StageTracker("req-2", event_bus=StageEventBus())

    stages = ["context_build", "intent_stage1", "safety", "routing", "response_gen"]
    for stage in stages:
        async with tracker.track_stage(stage):
            pass

    trace = tracker.trace
    assert len(trace) == len(stages)
    assert [t["stage"] for t in trace] == stages


@pytest.mark.asyncio
async def test_trace_includes_metadata() -> None:
    """Метаданные попадают в запись трейса."""
    tracker = StageTracker("req-3", event_bus=StageEventBus())

    async with tracker.track_stage("template_plan", metadata={"template_id": "plan_weekly"}):
        pass

    trace = tracker.trace
    assert trace[0]["metadata"] == {"template_id": "plan_weekly"}


@pytest.mark.asyncio
async def test_stage_events_published() -> None:
    """stage_start и stage_end публикуются в event bus."""
    bus = StageEventBus()
    tracker = StageTracker("req-4", event_bus=bus)
    rid = "req-4"

    received: list[StageEvent] = []

    async def consume() -> None:
        async for ev in bus.subscribe(rid):
            received.append(ev)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)

    async with tracker.track_stage("safety"):
        pass

    # Закрываем подписку
    bus.close(rid)
    await task

    types = [e.type for e in received]
    assert "stage_start" in types
    assert "stage_end" in types
    assert all(e.stage == "safety" for e in received)


@pytest.mark.asyncio
async def test_trace_recorded_even_on_exception() -> None:
    """Трейс записывается даже если внутри стадии произошло исключение."""
    tracker = StageTracker("req-5", event_bus=StageEventBus())

    with pytest.raises(ValueError):
        async with tracker.track_stage("intent_stage1"):
            raise ValueError("тест")

    trace = tracker.trace
    assert len(trace) == 1
    assert trace[0]["stage"] == "intent_stage1"


@pytest.mark.asyncio
async def test_trace_is_snapshot() -> None:
    """tracker.trace возвращает снимок, изменения не влияют на оригинал."""
    tracker = StageTracker("req-6", event_bus=StageEventBus())

    async with tracker.track_stage("routing"):
        pass

    snap1 = tracker.trace
    async with tracker.track_stage("response_gen"):
        pass
    snap2 = tracker.trace

    assert len(snap1) == 1
    assert len(snap2) == 2
