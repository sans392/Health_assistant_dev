"""Pydantic-схемы аргументов tool'ов и связанных enum'ов.

Задаёт строгий контракт между intent detection / router / planner и tool_executor:
— enum'ы для metric / sport_type / body_part / intensity
— TimeRange с нормализацией строки в конкретные даты
— по одной модели args на каждый tool

Валидация args выполняется в ToolExecutor перед вызовом tool'а. Невалидный
запрос приводит к ToolResult(success=False, error=...) вместо тихого сбоя
в недрах SQL/LLM.
"""

from __future__ import annotations

import types
from datetime import date, timedelta
from enum import Enum
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enum'ы (значения совпадают с теми, что исторически использует pipeline —
# intent_detection выдаёт их как строки, поэтому трогать значения нельзя.)
# ---------------------------------------------------------------------------


class MetricEnum(str, Enum):
    HEART_RATE = "heart_rate"
    HRV = "hrv"
    WEIGHT = "вес"
    CALORIES = "калории"
    STEPS = "шаги"
    HEIGHT = "рост"
    DISTANCE = "дистанция"
    DURATION = "время"
    PACE = "темп"
    CADENCE = "cadence"
    SLEEP = "сон"
    RECOVERY = "recovery"
    STRAIN = "strain"
    RPE = "rpe"


class SportTypeEnum(str, Enum):
    RUNNING = "running"
    CYCLING = "cycling"
    SWIMMING = "swimming"
    GYM = "gym"
    YOGA = "yoga"
    FOOTBALL = "football"
    BASKETBALL = "basketball"
    TENNIS = "tennis"
    SKIING = "skiing"
    WALKING = "walking"


class BodyPartEnum(str, Enum):
    BACK = "спина"
    KNEE = "колено"
    LOWER_BACK = "поясница"
    SHOULDERS = "плечи"
    NECK = "шея"
    HIP = "бедро"
    ANKLE = "лодыжка"
    WRIST = "запястье"
    ELBOW = "локоть"
    CALF = "икра"


class IntensityEnum(str, Enum):
    EASY = "легко"
    MODERATE = "умеренно"
    HARD = "тяжело"
    STRONG = "сильно"


class AnalysisType(str, Enum):
    """Тип анализа, если запрос требует не только выборку данных.

    Зарезервировано под refactor intent-таксономии (issue «Intent taxonomy refactor»):
    data_retrieval + data_analysis → единый data_query с ортогональным флагом.
    """

    NONE = "none"               # только выборка
    NORM_CHECK = "norm_check"   # «это нормально?»
    TREND = "trend"             # динамика во времени
    BREAKDOWN = "breakdown"     # детализация / где проседаю
    COMPARE = "compare"         # сравнение периодов


# ---------------------------------------------------------------------------
# TimeRange
# ---------------------------------------------------------------------------


class TimeRange(BaseModel):
    """Нормализованное временное окно.

    Создавать через TimeRange.from_label(label) или явно передавая даты.
    """

    date_from: date
    date_to: date
    label: str | None = None  # исходная фраза пользователя («за неделю», «январь»)

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _check_order(self) -> "TimeRange":
        if self.date_from > self.date_to:
            raise ValueError(
                f"date_from ({self.date_from}) must be <= date_to ({self.date_to})"
            )
        return self

    @property
    def days(self) -> int:
        """Длина окна в днях (включая оба конца)."""
        return (self.date_to - self.date_from).days + 1

    def as_tuple(self) -> tuple[date, date]:
        return self.date_from, self.date_to

    @classmethod
    def last_n_days(cls, n: int, today: date | None = None) -> "TimeRange":
        """Последние N дней, включая сегодня."""
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        ref = today or date.today()
        return cls(
            date_from=ref - timedelta(days=n - 1),
            date_to=ref,
            label=f"за последние {n} дней",
        )


# ---------------------------------------------------------------------------
# Args-модели по одному tool'у
# ---------------------------------------------------------------------------


class ToolArgsBase(BaseModel):
    """Базовый класс args. Все tool-args подразумевают user_id."""

    user_id: str = Field(min_length=1)

    model_config = {"extra": "forbid"}


class GetActivitiesArgs(ToolArgsBase):
    tool: Literal["get_activities"] = "get_activities"
    date_from: date
    date_to: date
    sport_type: SportTypeEnum | None = None

    @model_validator(mode="after")
    def _check_dates(self) -> "GetActivitiesArgs":
        if self.date_from > self.date_to:
            raise ValueError("date_from must be <= date_to")
        return self


class GetActivitiesBySportArgs(ToolArgsBase):
    tool: Literal["get_activities_by_sport"] = "get_activities_by_sport"
    sport_type: SportTypeEnum
    days: int = Field(default=30, ge=1, le=365)
    limit: int | None = Field(default=None, ge=1, le=1000)


class GetDailyFactsArgs(ToolArgsBase):
    tool: Literal["get_daily_facts"] = "get_daily_facts"
    date_from: date
    date_to: date
    metrics: list[MetricEnum] | None = None

    @model_validator(mode="after")
    def _check_dates(self) -> "GetDailyFactsArgs":
        if self.date_from > self.date_to:
            raise ValueError("date_from must be <= date_to")
        return self


class GetUserProfileArgs(ToolArgsBase):
    tool: Literal["get_user_profile"] = "get_user_profile"


class RagRetrieveArgs(BaseModel):
    """RAG retrieval — не требует user_id, но требует query_text."""

    tool: Literal["rag_retrieve"] = "rag_retrieve"
    query_text: str = Field(min_length=1)
    category: str | None = None
    sport_type: SportTypeEnum | None = None
    top_k: int = Field(default=5, ge=1, le=50)

    model_config = {"extra": "forbid"}


class ComputeRecoveryArgs(ToolArgsBase):
    tool: Literal["compute_recovery"] = "compute_recovery"
    window_days: int = Field(default=14, ge=3, le=60)


class ComputeStrainArgs(ToolArgsBase):
    tool: Literal["compute_strain"] = "compute_strain"
    reference_date: date


class CheckOvertrainingArgs(ToolArgsBase):
    tool: Literal["check_overtraining"] = "check_overtraining"
    window_days: int = Field(default=14, ge=3, le=60)


class LogActivityArgs(ToolArgsBase):
    tool: Literal["log_activity"] = "log_activity"
    sport_type: SportTypeEnum
    duration: int = Field(ge=0, description="длительность в секундах")
    calories: int = Field(default=0, ge=0)
    distance: float | None = Field(default=None, ge=0)
    notes: str | None = Field(default=None, max_length=500)


class UpdateProfileArgs(ToolArgsBase):
    tool: Literal["update_profile"] = "update_profile"
    field: Literal[
        "name", "age", "weight_kg", "height_cm", "gender",
        "max_heart_rate", "resting_heart_rate", "training_goals",
        "experience_level", "injuries", "chronic_conditions", "preferred_sports",
    ]
    value: Any = None


# ---------------------------------------------------------------------------
# Реестр: tool_name → args-модель. Используется ToolExecutor'ом для валидации.
# ---------------------------------------------------------------------------


TOOL_ARGS_REGISTRY: dict[str, type[BaseModel]] = {
    "get_activities": GetActivitiesArgs,
    "get_activities_by_sport": GetActivitiesBySportArgs,
    "get_daily_facts": GetDailyFactsArgs,
    "get_user_profile": GetUserProfileArgs,
    "rag_retrieve": RagRetrieveArgs,
    "compute_recovery": ComputeRecoveryArgs,
    "compute_strain": ComputeStrainArgs,
    "check_overtraining": CheckOvertrainingArgs,
    "log_activity": LogActivityArgs,
    "update_profile": UpdateProfileArgs,
}


def validate_tool_args(tool_name: str, raw_args: dict[str, Any]) -> BaseModel:
    """Валидирует raw dict через соответствующую Pydantic-модель.

    Raises:
        KeyError: если tool_name не зарегистрирован
        pydantic.ValidationError: если args не проходят валидацию

    Returns:
        Экземпляр args-модели, готовый к передаче в tool.
    """
    model_cls = TOOL_ARGS_REGISTRY[tool_name]
    return model_cls.model_validate(raw_args)


# ---------------------------------------------------------------------------
# Compact tool signature for LLM prompts
# ---------------------------------------------------------------------------

# Поля, которые planner НЕ передаёт сам — заполняются ToolExecutor / pipeline.
# Скрываем их из сигнатуры, чтобы не сбивать LLM.
_PROMPT_HIDDEN_FIELDS: frozenset[str] = frozenset({"user_id", "tool"})


def _render_field_type(annotation: Any) -> str:
    """Компактное отображение типа поля для prompt-сигнатуры."""
    origin = get_origin(annotation)

    # Optional / Union → отдаём первое не-None значение и помечаем поле как опциональное.
    if origin in (Union, types.UnionType):
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return _render_field_type(non_none[0])
        return "|".join(_render_field_type(a) for a in non_none)

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return "|".join(member.value for member in annotation)

    if annotation is date:
        return "YYYY-MM-DD"
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    if annotation is str:
        return "str"
    if annotation is bool:
        return "bool"

    if origin in (list, tuple):
        inner_args = get_args(annotation)
        inner = _render_field_type(inner_args[0]) if inner_args else "any"
        return f"list[{inner}]"

    if origin is Literal:
        return "|".join(repr(v) for v in get_args(annotation))

    return getattr(annotation, "__name__", str(annotation))


def _is_optional_field(field: Any) -> bool:
    """Поле считается опциональным, если у него есть default или Union с None."""
    annotation = field.annotation
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType) and type(None) in get_args(annotation):
        return True
    # Pydantic v2: PydanticUndefined означает «default не задан».
    from pydantic_core import PydanticUndefined
    if field.default is not PydanticUndefined:
        return True
    if getattr(field, "default_factory", None) is not None:
        return True
    return False


def tool_to_prompt_signature(tool_name: str) -> str:
    """Компактная сигнатура tool'а для system-prompt планировщика.

    Генерируется из Pydantic-схемы, чтобы описание не разъезжалось с реальным
    контрактом. Скрытые поля (`user_id`, `tool`) не показываем — их заполняет
    pipeline, а не LLM.

    Пример:
        get_activities(date_from: YYYY-MM-DD, date_to: YYYY-MM-DD,
                       sport_type?: running|cycling|...)
    """
    model_cls = TOOL_ARGS_REGISTRY[tool_name]
    parts: list[str] = []
    for name, field in model_cls.model_fields.items():
        if name in _PROMPT_HIDDEN_FIELDS:
            continue
        rendered = _render_field_type(field.annotation)
        suffix = "?" if _is_optional_field(field) else ""
        parts.append(f"{name}{suffix}: {rendered}")
    return f"{tool_name}({', '.join(parts)})"
