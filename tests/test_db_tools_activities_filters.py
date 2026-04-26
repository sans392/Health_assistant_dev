"""Тесты фильтров get_activities (app/tools/db_tools.py).

Проверяют, что whitelisted-фильтры реально доходят до SQL — для этого
поднимается in-memory SQLite и проверяется подмножество результата.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db import Base
from app.models.activity import Activity
from app.tools.db_tools import get_activities


@pytest_asyncio.fixture
async def db() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session = factory()
    try:
        yield session
    finally:
        await session.close()
        await engine.dispose()


def _act(
    user_id: str,
    sport: str,
    when: datetime,
    *,
    distance: float = 5000.0,
    duration: int = 1800,
    calories: int = 300,
    avg_hr: int = 140,
    avg_speed: float = 3.0,
    elevation: float = 0.0,
    title: str = "",
) -> Activity:
    return Activity(
        user_id=user_id,
        title=title or f"{sport} {when.date()}",
        sport_type=sport,
        distance_meters=distance,
        duration_seconds=duration,
        start_time=when,
        end_time=when + timedelta(seconds=duration),
        avg_speed=avg_speed,
        elevation_meters=elevation,
        calories=calories,
        avg_heart_rate=avg_hr,
        source="manual",
        is_primary=True,
        anomaly_flags=[],
        raw_title="",
    )


@pytest.mark.asyncio
async def test_sport_type_filter_works(db: AsyncSession) -> None:
    base = datetime(2026, 4, 20, 9, 0, 0)
    db.add_all([
        _act("u1", "running", base),
        _act("u1", "cycling", base + timedelta(days=1)),
    ])
    await db.commit()

    res = await get_activities(
        db=db, user_id="u1",
        date_from=date(2026, 4, 19), date_to=date(2026, 4, 25),
        sport_type="running",
    )
    assert res.success
    assert len(res.data) == 1
    assert res.data[0]["sport_type"] == "running"


@pytest.mark.asyncio
async def test_sport_types_list_filter(db: AsyncSession) -> None:
    base = datetime(2026, 4, 20, 9, 0, 0)
    db.add_all([
        _act("u1", "running", base),
        _act("u1", "cycling", base + timedelta(days=1)),
        _act("u1", "swimming", base + timedelta(days=2)),
    ])
    await db.commit()

    res = await get_activities(
        db=db, user_id="u1",
        date_from=date(2026, 4, 19), date_to=date(2026, 4, 25),
        sport_types=["running", "cycling"],
    )
    assert res.success
    sports = {r["sport_type"] for r in res.data}
    assert sports == {"running", "cycling"}


@pytest.mark.asyncio
async def test_distance_range_filter(db: AsyncSession) -> None:
    base = datetime(2026, 4, 20, 9, 0, 0)
    db.add_all([
        _act("u1", "running", base, distance=3000),
        _act("u1", "running", base + timedelta(days=1), distance=10000),
        _act("u1", "running", base + timedelta(days=2), distance=15000),
    ])
    await db.commit()

    res = await get_activities(
        db=db, user_id="u1",
        date_from=date(2026, 4, 19), date_to=date(2026, 4, 25),
        min_distance_meters=5000, max_distance_meters=12000,
    )
    assert res.success
    distances = sorted(int(r["distance_meters"]) for r in res.data)
    assert distances == [10000]


@pytest.mark.asyncio
async def test_avg_heart_rate_min_only(db: AsyncSession) -> None:
    base = datetime(2026, 4, 20, 9, 0, 0)
    db.add_all([
        _act("u1", "running", base, avg_hr=120),
        _act("u1", "running", base + timedelta(days=1), avg_hr=160),
    ])
    await db.commit()

    res = await get_activities(
        db=db, user_id="u1",
        date_from=date(2026, 4, 19), date_to=date(2026, 4, 25),
        min_avg_heart_rate=150,
    )
    assert res.success
    assert len(res.data) == 1
    assert res.data[0]["avg_heart_rate"] == 160


@pytest.mark.asyncio
async def test_title_contains_case_insensitive(db: AsyncSession) -> None:
    base = datetime(2026, 4, 20, 9, 0, 0)
    db.add_all([
        _act("u1", "running", base, title="Long run"),
        _act("u1", "running", base + timedelta(days=1), title="Tempo"),
    ])
    await db.commit()

    res = await get_activities(
        db=db, user_id="u1",
        date_from=date(2026, 4, 19), date_to=date(2026, 4, 25),
        title_contains="long",
    )
    assert res.success
    assert len(res.data) == 1
    assert "Long" in res.data[0]["title"]


@pytest.mark.asyncio
async def test_filters_combine_logical_and(db: AsyncSession) -> None:
    """sport_type + min_distance — оба фильтра должны применяться."""
    base = datetime(2026, 4, 20, 9, 0, 0)
    db.add_all([
        _act("u1", "running", base, distance=3000),         # отсеивается по distance
        _act("u1", "running", base + timedelta(days=1), distance=12000),  # ✓
        _act("u1", "cycling", base + timedelta(days=2), distance=20000),  # отсеивается по sport
    ])
    await db.commit()

    res = await get_activities(
        db=db, user_id="u1",
        date_from=date(2026, 4, 19), date_to=date(2026, 4, 25),
        sport_type="running",
        min_distance_meters=5000,
    )
    assert res.success
    assert len(res.data) == 1
    assert res.data[0]["sport_type"] == "running"
    assert res.data[0]["distance_meters"] == 12000


@pytest.mark.asyncio
async def test_no_filters_returns_all_in_range(db: AsyncSession) -> None:
    base = datetime(2026, 4, 20, 9, 0, 0)
    db.add_all([
        _act("u1", "running", base),
        _act("u1", "cycling", base + timedelta(days=1)),
    ])
    await db.commit()

    res = await get_activities(
        db=db, user_id="u1",
        date_from=date(2026, 4, 19), date_to=date(2026, 4, 25),
    )
    assert res.success
    assert len(res.data) == 2
