"""Утилиты для резолвинга time_range entity в конкретные даты."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.tools.schemas import TimeRange


_WEEKDAYS_RU = [
    "понедельник", "вторник", "среда", "четверг",
    "пятница", "суббота", "воскресенье",
]

_MONTHS_GEN_RU = [
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
]

# Корни месяцев в родительном падеже для распознавания «16 апреля» и т.п.
# Перечисляем именно корни, чтобы покрывать любые окончания (-я / -е / -й).
_MONTH_STEMS_RU: list[tuple[str, int]] = [
    ("январ", 1), ("феврал", 2), ("март", 3), ("апрел", 4),
    ("ма[йяе]", 5), ("июн", 6), ("июл", 7), ("август", 8),
    ("сентябр", 9), ("октябр", 10), ("ноябр", 11), ("декабр", 12),
]

# Optional ordinal suffix: «-го», «-ого», «-его», «-ое» (с дефисом и без).
# Длинные альтернативы — раньше коротких, чтобы regex не съел только «го»
# когда дальше идёт «ого».
_DAY_ORDINAL_SUFFIX = r"(?:-?(?:ого|его|го|ое))?"

_MONTHS_ALT = "|".join(f"{stem}\\w*" for stem, _ in _MONTH_STEMS_RU)

# «16 апреля», «16-го апреля», «16ого апреля»
_DAY_MONTH_RE = re.compile(
    rf"\b(\d{{1,2}}){_DAY_ORDINAL_SUFFIX}\s+({_MONTHS_ALT})\b"
)

# «16 числа», «16-го числа», «16ого числа», «16го числа».
_DAY_OF_MONTH_RE = re.compile(
    rf"\b(\d{{1,2}}){_DAY_ORDINAL_SUFFIX}\s+числа\b"
)

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def current_datetime_str(now: datetime | None = None) -> str:
    """Текущая дата и время сервера в читаемом русском формате.

    Используется в системных промптах LLM — чтобы модель понимала,
    что такое «сегодня» / «вчера» / «за последние 7 дней».
    """
    dt = now or datetime.now()
    weekday = _WEEKDAYS_RU[dt.weekday()]
    month = _MONTHS_GEN_RU[dt.month - 1]
    return (
        f"{weekday}, {dt.day} {month} {dt.year}, "
        f"{dt.strftime('%H:%M')} "
        f"(ISO: {dt.strftime('%Y-%m-%d %H:%M')})"
    )


def _resolve_day_in_current_month(day: int, today: date) -> date | None:
    """Number 1..31 → конкретная дата.

    Если такой день уже наступил в этом месяце — берём этот месяц,
    иначе предыдущий. Это покрывает естественную интерпретацию вопроса
    «что было 16-го?» — пользователь имеет в виду ближайшее прошлое.
    """
    if not 1 <= day <= 31:
        return None
    year, month = today.year, today.month
    candidate = _safe_date(year, month, day)
    if candidate is not None and candidate <= today:
        return candidate
    # День в этом месяце ещё не наступил (или 31 в коротком месяце) —
    # уходим в предыдущий месяц.
    if month == 1:
        year, month = year - 1, 12
    else:
        month -= 1
    return _safe_date(year, month, day)


def _safe_date(year: int, month: int, day: int) -> date | None:
    """date(...) с проверкой допустимости (например, 31 февраля → None)."""
    try:
        return date(year, month, day)
    except ValueError:
        return None


def resolve_time_range(time_range_entity: str | None) -> tuple[date, date]:
    """Преобразует строку time_range entity в конкретные даты.

    Args:
        time_range_entity: Строка вида «сегодня», «вчера», «за неделю», «за месяц»,
                           «за последние N дней», название месяца, или ISO-дата
                           «YYYY-MM-DD» (одиночный день — для конкретных запросов
                           вида «16-го числа»).

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

    # ISO-дата «YYYY-MM-DD» — одиночный день. Используется как label для
    # запросов вида «16-го числа», где extract_time_range_label уже
    # резолвит фразу в конкретную дату.
    if _ISO_DATE_RE.match(entity):
        try:
            d = date.fromisoformat(entity)
            return d, d
        except ValueError:
            pass

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


_NUMERIC_DAYS_PATTERNS: list[tuple[str, int]] = [
    # "3-4 дня" → берём меньшее число (консервативно, не расширяем окно
    # сверх того, что пользователь явно мог назвать).
    (r"\b(\d+)\s*[-–]\s*(\d+)\s+дн\w+\b", 0),  # 0 — маркер: взять меньшее
    (r"\bза\s+(\d+)\s+дн\w+\b", 1),
    (r"\bпоследни\w*\s+(\d+)\s+дн\w+\b", 1),
    (r"\b(\d+)\s+дн\w+\s+назад\b", 1),
]


def extract_time_range_label(text: str, today: date | None = None) -> str | None:
    """Извлечь «сырую» фразу time_range из произвольного текста.

    Возвращает нормализованный label, совместимый с resolve_time_range:
    «сегодня» / «вчера» / «за неделю» / «за месяц» / «за последние N дней» /
    название месяца / ISO-дата «YYYY-MM-DD» (одиночный день для запросов
    «16-го числа», «16 апреля»). None — если ничего не найдено.

    Нормализация поглощает формы вроде «3-4 дня» → «за последние 3 дня».
    """
    lower = text.lower()
    today = today or date.today()

    if re.search(r"\bсегодня\b", lower):
        return "сегодня"
    if re.search(r"\bвчера\b", lower):
        return "вчера"

    # Конкретный день месяца — приоритетнее общих фраз «за неделю», иначе
    # «за прошлую неделю 16 апреля» съест день. На практике точная дата
    # перевешивает, потому что запрос с конкретным числом — это запрос
    # точечных данных.
    iso = _try_match_specific_date(lower, today)
    if iso is not None:
        return iso

    if re.search(r"\bза неделю\b|\bна неделю\b|\bна прошл\w+ неделе\b|\bза последн\w+ неделю\b", lower):
        return "за неделю"
    if re.search(r"\bза месяц\b|\bна месяц\b|\bна прошл\w+ месяц\b|\bза последн\w+ месяц\b", lower):
        return "за месяц"

    # Числовые формы: «3-4 дня», «за 10 дней», «за последние 14 дней»
    m = re.search(r"\b(\d+)\s*[-–]\s*(\d+)\s+дн\w+\b", lower)
    if m:
        n = min(int(m.group(1)), int(m.group(2)))
        return f"за последние {n} дней"
    m = re.search(r"\bза\s+последни\w*\s+(\d+)\s+дн\w+\b", lower)
    if m:
        return f"за последние {int(m.group(1))} дней"
    m = re.search(r"\bза\s+(\d+)\s+дн\w+\b", lower)
    if m:
        return f"за последние {int(m.group(1))} дней"

    months = [
        (r"\bв январ\w+\b", "январь"),
        (r"\bв феврал\w+\b", "февраль"),
        (r"\bв март\w+\b", "март"),
        (r"\bв апрел\w+\b", "апрель"),
        (r"\bв ма[йе]\w*\b", "май"),
        (r"\bв июн\w+\b", "июнь"),
        (r"\bв июл\w+\b", "июль"),
        (r"\bв август\w*\b", "август"),
        (r"\bв сентябр\w+\b", "сентябрь"),
        (r"\bв октябр\w+\b", "октябрь"),
        (r"\bв ноябр\w+\b", "ноябрь"),
        (r"\bв декабр\w+\b", "декабрь"),
    ]
    for pattern, name in months:
        if re.search(pattern, lower):
            return name

    return None


# Корни месяцев (родительный падеж) → номер месяца.
_MONTH_STEM_TO_NUM: dict[str, int] = {}
for _stem, _num in _MONTH_STEMS_RU:
    if _stem == "ма[йяе]":
        for _v in ("май", "мая", "мае"):
            _MONTH_STEM_TO_NUM[_v] = 5
    else:
        _MONTH_STEM_TO_NUM[_stem] = _num


def _try_match_specific_date(lower: str, today: date) -> str | None:
    """«16 апреля», «16-го числа» → ISO-строка YYYY-MM-DD; иначе None."""
    # Сначала «N <месяц>» — он несёт больше информации (явный месяц).
    m = _DAY_MONTH_RE.search(lower)
    if m:
        day = int(m.group(1))
        month_word = m.group(2)
        month = _month_num_from_word(month_word)
        if month is not None:
            year = today.year
            candidate = _safe_date(year, month, day)
            if candidate is None:
                return None
            # Если месяц ещё не наступил в этом году — берём прошлый год.
            if candidate > today:
                candidate = _safe_date(year - 1, month, day)
                if candidate is None:
                    return None
            return candidate.isoformat()

    # «N числа» — без явного месяца, текущий месяц.
    m = _DAY_OF_MONTH_RE.search(lower)
    if m:
        day = int(m.group(1))
        resolved = _resolve_day_in_current_month(day, today)
        if resolved is not None:
            return resolved.isoformat()

    return None


def _month_num_from_word(word: str) -> int | None:
    """«апреля» / «мая» / «январе» → номер месяца 1..12."""
    for stem, num in _MONTH_STEM_TO_NUM.items():
        if word.startswith(stem):
            return num
    return None


def build_time_range(label: str | None) -> "TimeRange | None":
    """Построить TimeRange из нормализованного label.

    Возвращает None, если label пустой (чтобы SlotState мог отличить
    «пользователь не указал период» от «указал и это 7 дней по умолчанию»).
    """
    if not label:
        return None
    # Импорт локальный — чтобы избежать циклического импорта на module load.
    from app.tools.schemas import TimeRange

    date_from, date_to = resolve_time_range(label)
    return TimeRange(date_from=date_from, date_to=date_to, label=label)
