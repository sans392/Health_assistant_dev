"""Утилиты для резолвинга time_range entity в конкретные даты."""

import re
from datetime import date, timedelta


def resolve_time_range(time_range_entity: str | None) -> tuple[date, date]:
    """Преобразует строку time_range entity в конкретные даты.

    Args:
        time_range_entity: Строка вида «сегодня», «вчера», «за неделю», «за месяц»,
                           «за последние N дней» или название месяца.

    Returns:
        Кортеж (date_from, date_to). По умолчанию — последние 7 дней.
    """
    today = date.today()

    if time_range_entity is None:
        return today - timedelta(days=6), today

    entity = time_range_entity.lower().strip()

    if entity == "сегодня":
        return today, today

    if entity == "вчера":
        yesterday = today - timedelta(days=1)
        return yesterday, yesterday

    if entity in ("за неделю", "за последнюю неделю"):
        return today - timedelta(days=6), today

    if entity in ("за месяц", "за последний месяц"):
        return today - timedelta(days=29), today

    # «за последние N дней»
    m = re.search(r"за последни\w*\s+(\d+)\s+дн\w+", entity)
    if m:
        n = int(m.group(1))
        return today - timedelta(days=n - 1), today

    # Названия месяцев
    months: dict[str, int] = {
        "январь": 1, "февраль": 2, "март": 3, "апрель": 4,
        "май": 5, "июнь": 6, "июль": 7, "август": 8,
        "сентябрь": 9, "октябрь": 10, "ноябрь": 11, "декабрь": 12,
    }
    for month_name, month_num in months.items():
        if month_name in entity:
            year = today.year
            # Если месяц ещё не наступил — берём прошлый год
            if month_num > today.month:
                year -= 1
            first_day = date(year, month_num, 1)
            if month_num == 12:
                last_day = date(year, 12, 31)
            else:
                last_day = date(year, month_num + 1, 1) - timedelta(days=1)
            return first_day, last_day

    # Fallback: последние 7 дней
    return today - timedelta(days=6), today
