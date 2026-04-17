"""Тесты StageEventBus — pub/sub для событий пайплайна."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from app.pipeline.stage_events import StageEvent, StageEventBus


def _make_event(request_id: str, etype: str, stage: str | None = None) -> StageEvent:
    return StageEvent(
        type=etype,
        request_id=request_id,
        stage=stage,
        timestamp=datetime.utcnow(),
    )


@pytest.mark.asyncio
async def test_publish_subscribe_basic() -> None:
    """Подписчик получает опубликованное событие."""
    bus = StageEventBus()
    rid = "req-1"

    received: list[StageEvent] = []

    async def consume() -> None:
        async for event in bus.subscribe(rid):
            received.append(event)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)  # дать задаче запуститься

    ev = _make_event(rid, "stage_start", "intent")
    await bus.publish(rid, ev)
    done_ev = _make_event(rid, "done")
    await bus.publish(rid, done_ev)

    await task

    assert len(received) == 2
    assert received[0].type == "stage_start"
    assert received[1].type == "done"


@pytest.mark.asyncio
async def test_subscribe_stops_on_done() -> None:
    """Итерация завершается при type='done'."""
    bus = StageEventBus()
    rid = "req-2"

    received: list[StageEvent] = []

    async def consume() -> None:
        async for event in bus.subscribe(rid):
            received.append(event)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)

    await bus.publish(rid, _make_event(rid, "stage_start", "safety"))
    await bus.publish(rid, _make_event(rid, "done"))
    # Это событие уже не должно быть получено
    await bus.publish(rid, _make_event(rid, "stage_end", "safety"))

    await task

    assert len(received) == 2
    assert received[-1].type == "done"


@pytest.mark.asyncio
async def test_subscribe_stops_on_error() -> None:
    """Итерация завершается при type='error'."""
    bus = StageEventBus()
    rid = "req-err"

    received: list[StageEvent] = []

    async def consume() -> None:
        async for event in bus.subscribe(rid):
            received.append(event)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)

    await bus.publish(rid, _make_event(rid, "error"))
    await task

    assert len(received) == 1
    assert received[0].type == "error"


@pytest.mark.asyncio
async def test_close_unblocks_subscriber() -> None:
    """bus.close() разблокирует зависшего подписчика."""
    bus = StageEventBus()
    rid = "req-close"

    received: list[StageEvent] = []

    async def consume() -> None:
        async for event in bus.subscribe(rid):
            received.append(event)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)

    bus.close(rid)
    await task

    assert received == []


@pytest.mark.asyncio
async def test_no_cross_request_leakage() -> None:
    """События одного request_id не попадают к подписчику другого."""
    bus = StageEventBus()
    rid_a = "req-a"
    rid_b = "req-b"

    received_a: list[StageEvent] = []

    async def consume_a() -> None:
        async for event in bus.subscribe(rid_a):
            received_a.append(event)

    task = asyncio.create_task(consume_a())
    await asyncio.sleep(0)

    await bus.publish(rid_b, _make_event(rid_b, "stage_start", "routing"))
    await bus.publish(rid_a, _make_event(rid_a, "done"))
    await task

    assert len(received_a) == 1
    assert received_a[0].request_id == rid_a


@pytest.mark.asyncio
async def test_multiple_subscribers_same_request() -> None:
    """Несколько подписчиков на один request_id получают все события."""
    bus = StageEventBus()
    rid = "req-multi"

    received_1: list[StageEvent] = []
    received_2: list[StageEvent] = []

    async def consume(store: list) -> None:
        async for event in bus.subscribe(rid):
            store.append(event)

    t1 = asyncio.create_task(consume(received_1))
    t2 = asyncio.create_task(consume(received_2))
    await asyncio.sleep(0)

    await bus.publish(rid, _make_event(rid, "stage_start", "context_build"))
    await bus.publish(rid, _make_event(rid, "done"))
    await asyncio.gather(t1, t2)

    assert len(received_1) == 2
    assert len(received_2) == 2
