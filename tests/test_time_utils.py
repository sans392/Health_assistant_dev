"""Тесты для утилиты резолвинга time_range (app/tools/time_utils.py)."""

from datetime import date, datetime, timedelta

import pytest

from app.tools.time_utils import current_datetime_str, resolve_time_range


@pytest.fixture
def today() -> date:
    return date.today()


class TestResolveTimeRange:
    """Тесты resolve_time_range."""

    def test_none_returns_last_7_days(self, today: date) -> None:
        date_from, date_to = resolve_time_range(None)
        assert date_to == today
        assert date_from == today - timedelta(days=6)

    def test_сегодня(self, today: date) -> None:
        date_from, date_to = resolve_time_range("сегодня")
        assert date_from == today
        assert date_to == today

    def test_вчера(self, today: date) -> None:
        date_from, date_to = resolve_time_range("вчера")
        yesterday = today - timedelta(days=1)
        assert date_from == yesterday
        assert date_to == yesterday

    def test_за_неделю(self, today: date) -> None:
        date_from, date_to = resolve_time_range("за неделю")
        assert date_to == today
        assert date_from == today - timedelta(days=6)

    def test_за_месяц(self, today: date) -> None:
        date_from, date_to = resolve_time_range("за месяц")
        assert date_to == today
        assert date_from == today - timedelta(days=29)

    def test_за_последние_n_дней(self, today: date) -> None:
        date_from, date_to = resolve_time_range("за последние 14 дней")
        assert date_to == today
        assert date_from == today - timedelta(days=13)

    def test_за_последние_30_дней(self, today: date) -> None:
        date_from, date_to = resolve_time_range("за последние 30 дней")
        assert date_to == today
        assert date_from == today - timedelta(days=29)

    def test_month_name_январь(self, today: date) -> None:
        date_from, date_to = resolve_time_range("январь")
        assert date_from.month == 1
        assert date_from.day == 1
        assert date_to.month == 1
        assert date_to.day == 31

    def test_month_name_декабрь(self, today: date) -> None:
        date_from, date_to = resolve_time_range("декабрь")
        assert date_from.month == 12
        assert date_from.day == 1
        assert date_to.month == 12
        assert date_to.day == 31

    def test_unknown_fallback_to_7_days(self, today: date) -> None:
        date_from, date_to = resolve_time_range("неизвестный период")
        assert date_to == today
        assert date_from == today - timedelta(days=6)

    def test_date_from_lte_date_to(self) -> None:
        """date_from всегда ≤ date_to."""
        for entity in ["сегодня", "вчера", "за неделю", "за месяц", None]:
            date_from, date_to = resolve_time_range(entity)
            assert date_from <= date_to


class TestCurrentDatetimeStr:
    """Тесты current_datetime_str — форматированная дата и время сервера."""

    def test_contains_date_and_time(self) -> None:
        dt = datetime(2026, 4, 21, 14, 30)
        result = current_datetime_str(dt)
        assert "21" in result
        assert "2026" in result
        assert "14:30" in result
        # ISO-блок должен присутствовать для однозначного парсинга моделью
        assert "2026-04-21" in result

    def test_weekday_in_russian(self) -> None:
        # 2026-04-21 — вторник
        dt = datetime(2026, 4, 21, 9, 0)
        result = current_datetime_str(dt)
        assert "вторник" in result

    def test_month_in_russian_genitive(self) -> None:
        dt = datetime(2026, 4, 21, 9, 0)
        result = current_datetime_str(dt)
        assert "апреля" in result

    def test_without_arg_uses_current_time(self) -> None:
        # Без аргумента функция работает и возвращает непустую строку
        result = current_datetime_str()
        assert len(result) > 0
        assert "ISO:" in result
