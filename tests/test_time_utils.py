"""Тесты для утилиты резолвинга time_range (app/tools/time_utils.py)."""

from datetime import date, datetime, timedelta

import pytest

from app.tools.time_utils import (
    build_time_range,
    current_datetime_str,
    extract_time_range_label,
    resolve_time_range,
)


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


class TestExtractTimeRangeLabel:
    """Извлечение нормализованного time_range label из свободного текста."""

    def test_сегодня(self) -> None:
        assert extract_time_range_label("Что у меня сегодня?") == "сегодня"

    def test_вчера(self) -> None:
        assert extract_time_range_label("Сколько шагов вчера") == "вчера"

    def test_за_неделю(self) -> None:
        assert extract_time_range_label("Покажи тренировки за неделю") == "за неделю"

    def test_за_месяц(self) -> None:
        assert extract_time_range_label("Прогресс за месяц") == "за месяц"

    def test_range_3_to_4_days(self) -> None:
        """«3-4 дня» → за последние 3 дня (консервативный минимум)."""
        label = extract_time_range_label("пробежки за 3-4 дня")
        assert label == "за последние 3 дней"

    def test_за_10_дней(self) -> None:
        assert extract_time_range_label("за 10 дней") == "за последние 10 дней"

    def test_за_последние_14_дней(self) -> None:
        assert (
            extract_time_range_label("покажи за последние 14 дней")
            == "за последние 14 дней"
        )

    def test_month_name(self) -> None:
        assert extract_time_range_label("что было в январе") == "январь"
        assert extract_time_range_label("события в мае") == "май"

    def test_no_match_returns_none(self) -> None:
        assert extract_time_range_label("привет как дела") is None


class TestBuildTimeRange:
    """Построение TimeRange из нормализованного label."""

    def test_none_returns_none(self) -> None:
        assert build_time_range(None) is None

    def test_empty_returns_none(self) -> None:
        assert build_time_range("") is None

    def test_сегодня_builds_today(self) -> None:
        tr = build_time_range("сегодня")
        assert tr is not None
        assert tr.date_from == date.today()
        assert tr.date_to == date.today()
        assert tr.label == "сегодня"

    def test_за_неделю_builds_7_days(self) -> None:
        tr = build_time_range("за неделю")
        assert tr is not None
        assert tr.days == 7

    def test_numeric_days(self) -> None:
        tr = build_time_range("за последние 14 дней")
        assert tr is not None
        assert tr.days == 14


class TestSpecificDateExtraction:
    """Парсинг конкретных дат («16 числа», «16 апреля») в ISO-label."""

    def test_day_of_month_with_ordinal_dash(self) -> None:
        today = date(2026, 4, 25)
        assert extract_time_range_label("Мои шаги 16-го числа", today=today) == "2026-04-16"

    def test_day_of_month_no_dash(self) -> None:
        today = date(2026, 4, 25)
        assert extract_time_range_label("Мои тренировки 16го числа", today=today) == "2026-04-16"

    def test_day_of_month_full_ordinal(self) -> None:
        today = date(2026, 4, 25)
        assert extract_time_range_label("Шаги 16ого числа", today=today) == "2026-04-16"

    def test_day_of_month_no_ordinal(self) -> None:
        today = date(2026, 4, 25)
        assert extract_time_range_label("Шаги 16 числа", today=today) == "2026-04-16"

    def test_future_day_in_month_falls_back_to_prev_month(self) -> None:
        """«25-го числа» при сегодня 2026-04-10 → 2026-03-25 (ближайшее прошлое)."""
        today = date(2026, 4, 10)
        assert extract_time_range_label("шаги 25 числа", today=today) == "2026-03-25"

    def test_day_with_month_name(self) -> None:
        today = date(2026, 4, 25)
        assert extract_time_range_label("Шаги 16 апреля", today=today) == "2026-04-16"

    def test_day_with_month_name_dash_ordinal(self) -> None:
        today = date(2026, 4, 25)
        assert extract_time_range_label("Шаги 16-го апреля", today=today) == "2026-04-16"

    def test_day_with_future_month_uses_prev_year(self) -> None:
        """«16 мая» при сегодня 2026-04-25 → 2025-05-16 (май ещё не наступил)."""
        today = date(2026, 4, 25)
        assert extract_time_range_label("Шаги 16 мая", today=today) == "2025-05-16"

    def test_invalid_day_returns_no_match(self) -> None:
        """31 февраля невалидно — функция не должна падать, возвращает None."""
        today = date(2026, 4, 25)
        assert extract_time_range_label("31 февраля", today=today) is None

    def test_specific_date_overrides_relative_phrase(self) -> None:
        """«16 числа за неделю» — конкретная дата приоритетнее общего интервала."""
        today = date(2026, 4, 25)
        label = extract_time_range_label("Активность 16 числа за неделю", today=today)
        assert label == "2026-04-16"


class TestResolveIsoLabel:
    """resolve_time_range понимает ISO-метку YYYY-MM-DD как одиночный день."""

    def test_iso_label_returns_single_day(self) -> None:
        date_from, date_to = resolve_time_range("2026-04-16")
        assert date_from == date(2026, 4, 16)
        assert date_to == date(2026, 4, 16)

    def test_build_time_range_from_iso_label(self) -> None:
        tr = build_time_range("2026-04-16")
        assert tr is not None
        assert tr.days == 1
        assert tr.date_from == date(2026, 4, 16)
        assert tr.label == "2026-04-16"
