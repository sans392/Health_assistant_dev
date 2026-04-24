"""Тесты get_daily_facts (app/tools/db_tools.py).

Фокус — фильтр по metrics: заросит-язык MetricEnum («шаги», «калории»,
«сон», «recovery», ...) должен корректно маппиться на поля DailyFact.
"""

from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.tools.db_tools import get_daily_facts


def _fake_fact(iso_date: str) -> SimpleNamespace:
    return SimpleNamespace(
        id="fact-" + iso_date,
        user_id="u1",
        iso_date=iso_date,
        steps=11957,
        calories_kcal=2483,
        recovery_score=72,
        hrv_rmssd_milli=65.0,
        resting_heart_rate=55,
        spo2_percentage=96.8,
        skin_temp_celsius=35.03,
        sleep_total_in_bed_milli=27198203,
        water_liters=2.7,
    )


def _db_returning(facts: list[SimpleNamespace]) -> AsyncMock:
    """Мок AsyncSession: db.execute(...) → result.scalars().all() == facts."""
    scalars = MagicMock()
    scalars.all.return_value = facts
    exec_result = MagicMock()
    exec_result.scalars.return_value = scalars

    db = MagicMock()
    db.execute = AsyncMock(return_value=exec_result)
    return db


@pytest.mark.asyncio
async def test_metrics_none_returns_all_fields() -> None:
    db = _db_returning([_fake_fact("2026-04-18")])
    res = await get_daily_facts(
        db=db, user_id="u1",
        date_from=date(2026, 4, 18), date_to=date(2026, 4, 18),
        metrics=None,
    )
    assert res.success
    assert len(res.data) == 1
    row = res.data[0]
    assert row["steps"] == 11957
    assert row["calories_kcal"] == 2483
    assert "hrv_rmssd_milli" in row


@pytest.mark.asyncio
async def test_metrics_russian_steps_keeps_steps_field() -> None:
    """Регрессия: запрос «шаги» не должен выкидывать steps из результата."""
    db = _db_returning([_fake_fact("2026-04-18")])
    res = await get_daily_facts(
        db=db, user_id="u1",
        date_from=date(2026, 4, 18), date_to=date(2026, 4, 18),
        metrics=["шаги"],
    )
    assert res.success
    row = res.data[0]
    assert row["steps"] == 11957
    assert row["iso_date"] == "2026-04-18"
    # Остальные метрики должны быть отфильтрованы
    assert "calories_kcal" not in row
    assert "hrv_rmssd_milli" not in row


@pytest.mark.asyncio
async def test_metrics_multiple_russian_and_english_aliases() -> None:
    db = _db_returning([_fake_fact("2026-04-18")])
    res = await get_daily_facts(
        db=db, user_id="u1",
        date_from=date(2026, 4, 18), date_to=date(2026, 4, 18),
        metrics=["калории", "hrv", "recovery"],
    )
    row = res.data[0]
    assert row["calories_kcal"] == 2483
    assert row["hrv_rmssd_milli"] == 65.0
    assert row["recovery_score"] == 72
    assert "steps" not in row


@pytest.mark.asyncio
async def test_metrics_column_name_also_works() -> None:
    """Если модель передаёт имя столбца напрямую — тоже должно сработать."""
    db = _db_returning([_fake_fact("2026-04-18")])
    res = await get_daily_facts(
        db=db, user_id="u1",
        date_from=date(2026, 4, 18), date_to=date(2026, 4, 18),
        metrics=["steps"],
    )
    assert res.data[0]["steps"] == 11957


@pytest.mark.asyncio
async def test_metrics_unknown_metric_is_ignored() -> None:
    """Неизвестная метрика не должна рушить запрос — просто пропускаем."""
    db = _db_returning([_fake_fact("2026-04-18")])
    res = await get_daily_facts(
        db=db, user_id="u1",
        date_from=date(2026, 4, 18), date_to=date(2026, 4, 18),
        metrics=["темп", "шаги"],
    )
    row = res.data[0]
    assert row["steps"] == 11957
    # «темп» нет в DailyFact — игнорируем, остальных метрик не должно быть
    assert "calories_kcal" not in row
