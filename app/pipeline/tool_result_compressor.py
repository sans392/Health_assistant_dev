"""Schema-aware сжатие tool-результатов для Planner Agent (Issue #58).

Заменяет посимвольное обрезание (`_MAX_TOOL_RESULT_CHARS`) в `planner.py`
на структурированное сжатие, учитывающее природу данных каждого tool:

- list-результаты (`get_activities`, `get_daily_facts`) → summary + top-N
  наиболее значимых записей (по ключам калорий / HR / аномалии метрик).
- dict-результаты (`compute_recovery`, `check_overtraining`, `get_user_profile`)
  → pass-through (они уже компактны).
- RAG-результаты (`rag_retrieve`) → полный текст top-1, остальные — title+snippet.

Возвращает `CompressedResult` c флагом `compressed=True`, когда сжатие реально
применено, чтобы планировщик знал, что может запросить уточнение/больше данных.

Entry-point: `compress_for_planner(tool_name, raw_result) -> CompressedResult`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from app.services.data_processing.summary_builder import (
    build_activity_summary,
    build_metric_summary,
)

# ---------------------------------------------------------------------------
# Константы сжатия
# ---------------------------------------------------------------------------

# Количество записей-образцов в sample для list-tools.
_LIST_SAMPLE_SIZE = 5

# Сколько символов текста оставляем в сжатых RAG-чанках (кроме top-1).
_RAG_SNIPPET_CHARS = 150

# Порог, при котором list-результат считается «длинным» и нуждается в сжатии.
# Для коротких выборок (≤ sample_size) нет смысла делать summary — отдаём as is.
_LIST_COMPRESS_THRESHOLD = _LIST_SAMPLE_SIZE

# Какие метрики daily_facts использовать для выбора «самых аномальных» записей.
# Порядок важен: первая найденная ненулевая метрика используется для сортировки.
_DAILY_FACT_SORT_METRICS = (
    "hrv_rmssd_milli",
    "resting_heart_rate",
    "recovery_score",
    "strain_score",
    "steps",
)

# Известные tools, для которых применяем compress logic.
_LIST_TOOLS = {"get_activities", "get_daily_facts"}
_DICT_TOOLS = {
    "compute_recovery",
    "check_overtraining",
    "get_user_profile",
}
_RAG_TOOLS = {"rag_retrieve"}


# ---------------------------------------------------------------------------
# DTO
# ---------------------------------------------------------------------------


@dataclass
class CompressedResult:
    """Результат сжатия tool-вывода для подачи обратно планировщику.

    payload — сериализуемая структура, попадающая в сообщение LLM.
    Для list-tools: `{"summary": {...}, "sample": [...], "total_count": N}`.
    Для dict/rag/коротких list — исходный объект (list/dict/None).

    Флаг `compressed=True` означает, что произошло реальное сжатие и
    `full_count` > `shown`. Планировщик может использовать это, чтобы попросить
    уточнение или итерацию с другим фильтром.
    """

    payload: Any
    compressed: bool = False
    full_count: int | None = None
    shown: int | None = None

    def to_message_payload(self) -> Any:
        """Сериализуемый формат для подстановки в сообщение планировщика.

        Если было сжатие — оборачиваем в dict с флагом `compressed`.
        Иначе — отдаём payload как есть (pass-through).
        """
        if not self.compressed:
            return self.payload
        wrapped: dict[str, Any] = {
            "tool_result": self.payload,
            "compressed": True,
        }
        if self.full_count is not None:
            wrapped["full_count"] = self.full_count
        if self.shown is not None:
            wrapped["shown"] = self.shown
        return wrapped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compress_for_planner(tool_name: str, raw_result: Any) -> CompressedResult:
    """Сжать сырой tool-результат для подачи в сообщение планировщика.

    Args:
        tool_name: Имя tool'а из планировщика (get_activities, rag_retrieve, ...).
        raw_result: Сырые данные tool'а (list | dict | None).

    Returns:
        CompressedResult с payload'ом и флагами сжатия.
    """
    if raw_result is None:
        return CompressedResult(payload=None)

    if tool_name in _RAG_TOOLS:
        return _compress_rag(raw_result)

    if tool_name == "get_activities":
        return _compress_activities(raw_result)

    if tool_name == "get_daily_facts":
        return _compress_daily_facts(raw_result)

    if tool_name in _DICT_TOOLS:
        # Фиксированные поля — pass-through.
        return CompressedResult(payload=raw_result)

    # Неизвестный tool: применяем generic list/dict эвристику.
    if isinstance(raw_result, list):
        return _compress_generic_list(raw_result)
    return CompressedResult(payload=raw_result)


# ---------------------------------------------------------------------------
# List tools: activities / daily_facts
# ---------------------------------------------------------------------------


def _compress_activities(raw: Any) -> CompressedResult:
    """Сжать список активностей (get_activities)."""
    if not isinstance(raw, list):
        return CompressedResult(payload=raw)

    total = len(raw)
    if total <= _LIST_COMPRESS_THRESHOLD:
        return CompressedResult(payload=raw)

    summary_obj = build_activity_summary(raw)
    summary_dict = asdict(summary_obj)
    # Убираем избыточные поля most_intense.sport_type, совпадающие с by_sport_counts.
    # Оставляем как есть — они нужны планировщику для принятия решений.

    # Выбираем top-N по интенсивности (по той же функции, что и most_intense).
    sample = sorted(raw, key=_activity_intensity_score, reverse=True)[:_LIST_SAMPLE_SIZE]

    payload = {
        "summary": summary_dict,
        "sample": sample,
        "total_count": total,
    }
    return CompressedResult(
        payload=payload,
        compressed=True,
        full_count=total,
        shown=len(sample),
    )


def _activity_intensity_score(act: dict) -> float:
    """Скор интенсивности — согласован с summary_builder._activity_intensity."""
    if not isinstance(act, dict):
        return 0.0
    calories = float(act.get("calories") or 0)
    duration = float(act.get("duration_seconds") or 0)
    distance = float(act.get("distance_meters") or 0)
    return calories + duration * 0.05 + distance * 0.001


def _compress_daily_facts(raw: Any) -> CompressedResult:
    """Сжать список дневных фактов (get_daily_facts)."""
    if not isinstance(raw, list):
        return CompressedResult(payload=raw)

    total = len(raw)
    if total <= _LIST_COMPRESS_THRESHOLD:
        return CompressedResult(payload=raw)

    # Собираем summary по каждой «значимой» метрике, у которой есть хотя бы
    # одно ненулевое значение в выборке.
    metrics_summary: dict[str, dict[str, Any]] = {}
    for metric in _DAILY_FACT_SORT_METRICS:
        if not any(_has_value(f, metric) for f in raw):
            continue
        ms = build_metric_summary(raw, metric)
        if ms.count == 0:
            continue
        metrics_summary[metric] = {
            "mean": ms.mean,
            "min": ms.min_value,
            "max": ms.max_value,
            "latest_value": ms.latest_value,
            "latest_date": ms.latest_date,
            "baseline_mean": ms.baseline_mean,
            "delta_pct": ms.delta_pct,
            "anomaly_flags": ms.anomaly_flags,
        }

    # Sample: берём записи с наибольшим отклонением по первой доступной метрике.
    sort_metric = _choose_sort_metric(raw)
    if sort_metric is not None:
        sample = _top_anomalous_facts(raw, sort_metric, _LIST_SAMPLE_SIZE)
    else:
        # Нет подходящих метрик — берём последние N по дате.
        sample = sorted(
            raw,
            key=lambda f: f.get("iso_date") or "",
            reverse=True,
        )[:_LIST_SAMPLE_SIZE]

    payload = {
        "summary": {
            "total_days": total,
            "metrics": metrics_summary,
        },
        "sample": sample,
        "total_count": total,
    }
    return CompressedResult(
        payload=payload,
        compressed=True,
        full_count=total,
        shown=len(sample),
    )


def _has_value(fact: dict, metric: str) -> bool:
    """Проверить, что в fact есть числовое значение метрики."""
    if not isinstance(fact, dict):
        return False
    value = fact.get(metric)
    if value is None:
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _choose_sort_metric(facts: list[dict]) -> str | None:
    """Первая метрика из приоритетного списка, по которой в facts есть данные."""
    for metric in _DAILY_FACT_SORT_METRICS:
        if any(_has_value(f, metric) for f in facts):
            return metric
    return None


def _top_anomalous_facts(
    facts: list[dict],
    metric: str,
    n: int,
) -> list[dict]:
    """Выбрать n фактов с максимальным |value − mean| по метрике."""
    values = [float(f[metric]) for f in facts if _has_value(f, metric)]
    if not values:
        return facts[:n]
    mean = sum(values) / len(values)

    def _deviation(fact: dict) -> float:
        if not _has_value(fact, metric):
            return -1.0  # facts без метрики уходят вниз
        return abs(float(fact[metric]) - mean)

    return sorted(facts, key=_deviation, reverse=True)[:n]


# ---------------------------------------------------------------------------
# RAG: rag_retrieve
# ---------------------------------------------------------------------------


def _compress_rag(raw: Any) -> CompressedResult:
    """Сжать результат rag_retrieve: top-1 полностью, остальные — title+snippet."""
    if not isinstance(raw, list):
        return CompressedResult(payload=raw)

    total = len(raw)
    if total <= 1:
        return CompressedResult(payload=raw)

    compressed_chunks: list[dict[str, Any]] = []
    for idx, chunk in enumerate(raw):
        if not isinstance(chunk, dict):
            compressed_chunks.append(chunk)  # fallback — как есть
            continue
        if idx == 0:
            # top-1 — оставляем целиком
            compressed_chunks.append(chunk)
            continue
        text = chunk.get("text") or ""
        snippet = text[:_RAG_SNIPPET_CHARS]
        if len(text) > _RAG_SNIPPET_CHARS:
            snippet += "…"
        compressed_chunks.append({
            "title": _chunk_title(chunk),
            "snippet": snippet,
            "category": chunk.get("category"),
            "source": chunk.get("source"),
            "score": chunk.get("score"),
        })

    really_trimmed = any(
        isinstance(c, dict) and "snippet" in c for c in compressed_chunks
    )
    return CompressedResult(
        payload=compressed_chunks,
        compressed=really_trimmed,
        full_count=total,
        shown=total,
    )


def _chunk_title(chunk: dict) -> str:
    """Выбрать «заголовок» RAG-чанка. У нас нет явного title — составляем из source/category."""
    source = chunk.get("source") or ""
    category = chunk.get("category") or ""
    if source and category:
        return f"{source} [{category}]"
    return source or category or "chunk"


# ---------------------------------------------------------------------------
# Generic fallback (неизвестные list-tools)
# ---------------------------------------------------------------------------


def _compress_generic_list(raw: list) -> CompressedResult:
    """Fallback-сжатие для неизвестных list-tools: просто берём первые N."""
    total = len(raw)
    if total <= _LIST_COMPRESS_THRESHOLD:
        return CompressedResult(payload=raw)
    sample = raw[:_LIST_SAMPLE_SIZE]
    payload = {
        "sample": sample,
        "total_count": total,
    }
    return CompressedResult(
        payload=payload,
        compressed=True,
        full_count=total,
        shown=len(sample),
    )
