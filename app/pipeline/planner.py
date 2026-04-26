"""Planner Agent — LLM в цикле с tool-calls (Phase 2, Issue #29).

Базовая версия: JSON function-calling через промпт (не native Ollama tool-use).
Модель: роль `planner` (heavy) из LLMRegistry.

Цикл:
  1. Сформировать промпт с историей tool-calls
  2. Вызвать LLM (роль planner)
  3. Парсить JSON
  4. Если final_answer: true → выходим, передаём контекст в Response Generator
  5. Иначе — исполнить tool_calls, добавить результаты в историю
  6. При parse error — 1 retry с инструкцией «верни валидный JSON»
  7. По достижении max_iterations → форсированный выход
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from pydantic import ValidationError

from app.pipeline.tool_result_compressor import compress_for_planner
from app.services.llm_registry import llm_registry
from app.services.tool_call_logger import tool_call_logger
from app.tools.db_tools import get_activities, get_daily_facts, get_user_profile
from app.tools.rag_retrieve import rag_retrieve
from app.tools.schemas import (
    GetActivitiesArgs,
    GetDailyFactsArgs,
    tool_to_prompt_signature,
    validate_tool_args,
)
from app.tools.time_utils import current_datetime_str

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = 5
_MAX_TOOL_CALLS_PER_ITER = 3
_TIMEOUT_TOTAL = 360.0  # секунды


def _parse_iso_date(value: Any) -> date | None:
    """Безопасный парсер ISO-даты YYYY-MM-DD из произвольного значения."""
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None

# Описания tools — компактные, человеческие. Сигнатуры подставляются автоматически
# из Pydantic-схем (см. tool_to_prompt_signature), чтобы они не разъезжались с
# реальным контрактом ToolExecutor'а.
_TOOL_DESCRIPTIONS: list[tuple[str, str]] = [
    ("get_activities", "тренировки пользователя за интервал дат"),
    ("get_daily_facts", "дневные метрики здоровья (HRV, ЧСС, сон, шаги) за интервал дат"),
    ("get_user_profile", "профиль пользователя"),
    ("compute_recovery", "recovery score за окно window_days"),
    ("check_overtraining", "признаки перетренированности за окно window_days"),
    (
        "rag_retrieve",
        "знания из базы. Категории: physiology_norms | training_principles | "
        "recovery_science | sport_specific | nutrition_basics",
    ),
]


def _build_tools_description() -> str:
    lines = []
    for name, desc in _TOOL_DESCRIPTIONS:
        lines.append(f"- {tool_to_prompt_signature(name)}: {desc}")
    lines.append("")
    lines.append(
        "Аргументы date_from / date_to передавай в формате YYYY-MM-DD. "
        "Опциональные поля (помечены `?`) можно опускать."
    )
    lines.append(
        "Примеры: точечный день — date_from=date_to=YYYY-MM-DD; "
        "интервал — задавай оба конца явно."
    )
    lines.append(
        "У get_activities / get_daily_facts есть whitelisted-фильтры по полям "
        "(min_/max_distance_meters, min_/max_duration_seconds, min_/max_calories, "
        "min_/max_avg_heart_rate, min_/max_steps, min_/max_recovery_score, "
        "min_/max_hrv_rmssd_milli, ...). Используй их, чтобы сразу сузить "
        "выборку под запрос — например, «длинные пробежки» → "
        "sport_type=running + min_distance_meters=10000."
    )
    lines.append("")
    lines.append(
        "Формат результатов tools:\n"
        "- Длинные list-результаты (get_activities, get_daily_facts при >5 записей) "
        "приходят сжатыми: {\"tool_result\": {\"summary\": {...}, \"sample\": [...], "
        "\"total_count\": N}, \"compressed\": true, \"full_count\": N, \"shown\": K}. "
        "Поле \"summary\" содержит агрегаты и baseline-сравнение, \"sample\" — top-K "
        "самых значимых записей. Используй summary для общих выводов; при "
        "необходимости деталей вызови tool повторно с более узким интервалом "
        "(date_from / date_to ближе к интересующему дню) или конкретным sport_type.\n"
        "- dict-результаты (compute_recovery, check_overtraining, get_user_profile) "
        "приходят без сжатия.\n"
        "- rag_retrieve: top-1 чанк целиком, остальные — title+snippet "
        "(поле \"snippet\")."
    )
    return "\n".join(lines)


_TOOLS_DESCRIPTION = _build_tools_description()

_SYSTEM_PROMPT_TEMPLATE = """\
Ты — планировщик фитнес-ассистента. Для ответа пользователю вызывай tools.
Отвечай строго в JSON формате (без текста вне JSON):
{{
  "thought": "твои рассуждения",
  "tool_calls": [
    {{"tool": "имя_tool", "args": {{...}}}}
  ]
}}
или, если готов дать финальный ответ:
{{
  "thought": "обоснование финального ответа",
  "final_answer": true
}}

Текущая дата и время сервера: {current_datetime}

Доступные tools:
{tools}

Контекст пользователя:
{user_context}

Максимум {max_tool_calls} tool-вызовов за итерацию. Максимум {max_iterations} итераций."""


@dataclass
class PlannerResult:
    """Результат работы PlannerAgent."""

    tool_results: dict[str, Any] = field(default_factory=dict)
    tool_calls_history: list[dict] = field(default_factory=list)
    iterations: int = 0
    total_tool_calls: int = 0
    timeout_hit: bool = False
    error: str | None = None
    # Сырые результаты по итерациям (для логов/отладки). В tool_results
    # лежит дедуплицированное merged-представление для Response Generator.
    raw_iter_results: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return not self.timeout_hit and self.error is None


class PlannerAgent:
    """LLM-планировщик в цикле tool-calls (роль: planner).

    Вызывает heavy-модель, разбирает JSON-ответы,
    исполняет tool-calls, накапливает контекст до final_answer.
    """

    def __init__(self) -> None:
        self._max_iterations = _MAX_ITERATIONS
        self._max_tool_calls_per_iter = _MAX_TOOL_CALLS_PER_ITER
        self._timeout = _TIMEOUT_TOTAL

    async def plan(
        self,
        query: str,
        user_id: str,
        user_context: str,
        entities: dict,
        db: AsyncSession,
        request_id: str | None = None,
    ) -> PlannerResult:
        """Запустить цикл планирования.

        Args:
            query: Запрос пользователя.
            user_id: ID пользователя.
            user_context: Текстовый контекст (профиль + история сессии).
            entities: Сущности из IntentResult (sport_type и т.д.).
            db: Async DB session.
            request_id: ID запроса для логирования.

        Returns:
            PlannerResult с собранными tool results.
        """
        result = PlannerResult()
        start_time = time.monotonic()
        llm_client = llm_registry.get_client("planner")
        sport_type: str | None = entities.get("sport_type")

        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            tools=_TOOLS_DESCRIPTION,
            user_context=user_context,
            current_datetime=current_datetime_str(),
            max_tool_calls=self._max_tool_calls_per_iter,
            max_iterations=self._max_iterations,
        )

        # История диалога в chat-формате:
        # user → исходный запрос, далее assistant (JSON планер) ↔ user (tool results).
        messages: list[dict[str, str]] = [
            {"role": "user", "content": query},
        ]

        for iteration in range(1, self._max_iterations + 1):
            elapsed = time.monotonic() - start_time
            if elapsed >= self._timeout:
                logger.warning(
                    "PlannerAgent: таймаут до итерации %d | elapsed=%.1fs | user=%s",
                    iteration, elapsed, user_id,
                )
                result.timeout_hit = True
                break

            logger.info(
                "PlannerAgent: итерация %d/%d | elapsed=%.1fs | user=%s",
                iteration, self._max_iterations, elapsed, user_id,
            )

            remaining = max(self._timeout - (time.monotonic() - start_time), 5.0)
            try:
                llm_response = await asyncio.wait_for(
                    llm_client.chat(
                        messages=messages,
                        system_prompt=system_prompt,
                        temperature=0.3,
                        format="json",
                    ),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                logger.warning("PlannerAgent: таймаут LLM на итерации %d", iteration)
                result.timeout_hit = True
                break

            result.iterations = iteration
            raw_content = llm_response.content
            parsed = self._parse_response(raw_content)

            if parsed is None:
                logger.warning("PlannerAgent: JSON parse error на итерации %d, retry", iteration)
                retry_messages = messages + [
                    {"role": "assistant", "content": raw_content or ""},
                    {
                        "role": "user",
                        "content": (
                            "Твой предыдущий ответ не был валидным JSON. "
                            "Верни ТОЛЬКО валидный JSON по схеме, без текста вне JSON."
                        ),
                    },
                ]
                remaining = max(self._timeout - (time.monotonic() - start_time), 5.0)
                try:
                    llm_response = await asyncio.wait_for(
                        llm_client.chat(
                            messages=retry_messages,
                            system_prompt=system_prompt,
                            temperature=0.1,
                            format="json",
                        ),
                        timeout=remaining,
                    )
                    raw_content = llm_response.content
                    parsed = self._parse_response(raw_content)
                except asyncio.TimeoutError:
                    result.timeout_hit = True
                    break

                if parsed is None:
                    logger.error("PlannerAgent: повторная ошибка парсинга на итерации %d", iteration)
                    result.error = f"JSON parse error на итерации {iteration}"
                    break

            # Ответ планировщика фиксируем в истории в компактной JSON-форме,
            # чтобы модель видела свои предыдущие решения и не зацикливалась.
            assistant_json = self._serialize_assistant_turn(parsed)
            messages.append({"role": "assistant", "content": assistant_json})

            if parsed.get("final_answer"):
                logger.info("PlannerAgent: final_answer на итерации %d", iteration)
                break

            tool_calls = (parsed.get("tool_calls") or [])[:self._max_tool_calls_per_iter]
            if not tool_calls:
                logger.info("PlannerAgent: пустой tool_calls на итерации %d — завершаем", iteration)
                break

            tool_results_payload: list[dict] = []
            for tc in tool_calls:
                tool_name = tc.get("tool", "")
                tool_args = tc.get("args", {})
                result.tool_calls_history.append({
                    "iteration": iteration, "tool": tool_name, "args": tool_args,
                })

                tool_start_ms = time.monotonic() * 1000
                try:
                    tool_data = await self._execute_tool(
                        tool_name=tool_name, args=tool_args,
                        user_id=user_id, query=query,
                        sport_type=sport_type, db=db,
                    )
                    result.raw_iter_results[f"{tool_name}_iter{iteration}"] = tool_data
                    result.total_tool_calls += 1
                    compressed = compress_for_planner(tool_name, tool_data)
                    tool_results_payload.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result": compressed.to_message_payload(),
                    })
                    logger.info("PlannerAgent: tool '%s' выполнен | iter=%d", tool_name, iteration)
                    tool_call_logger.record(
                        name=tool_name,
                        source="planner",
                        args=tool_args,
                        result=tool_data,
                        success=tool_data is not None,
                        error=None if tool_data is not None else "нет данных / неизвестный tool",
                        duration_ms=int(time.monotonic() * 1000 - tool_start_ms),
                        iteration=iteration,
                    )
                except Exception as exc:
                    logger.error("PlannerAgent: tool '%s' ошибка: %s", tool_name, exc)
                    tool_results_payload.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "error": str(exc),
                    })
                    tool_call_logger.record(
                        name=tool_name,
                        source="planner",
                        args=tool_args,
                        result=None,
                        success=False,
                        error=str(exc),
                        duration_ms=int(time.monotonic() * 1000 - tool_start_ms),
                        iteration=iteration,
                    )

            # Результаты tools отправляем как следующее user-сообщение.
            user_turn = json.dumps(
                {"tool_results": tool_results_payload},
                ensure_ascii=False,
                default=str,
            )
            messages.append({"role": "user", "content": user_turn})

        # Дедуплицированный view для Response Generator: одна запись на tool,
        # списки мёржатся по естественному ключу (id / iso_date / started_at).
        result.tool_results = self._merge_iter_results(
            raw_results=result.raw_iter_results,
            history=result.tool_calls_history,
        )

        logger.info(
            "PlannerAgent: завершён | iterations=%d tool_calls=%d merged_tools=%d "
            "timeout=%s error=%s | user=%s",
            result.iterations, result.total_tool_calls,
            len(result.tool_results), result.timeout_hit, result.error, user_id,
        )
        return result

    def _parse_response(self, content: str) -> dict | None:
        """Разобрать JSON-ответ LLM. Возвращает None при ошибке парсинга."""
        content = content.strip()
        # Убираем markdown-блок если есть
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        # Пробуем найти JSON в тексте
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass
        return None

    def _summarize(self, data: Any) -> str:
        """Краткое резюме данных для вставки в историю итераций."""
        if data is None:
            return "нет данных"
        if isinstance(data, list):
            return f"[{len(data)} записей]"
        if isinstance(data, dict):
            keys = list(data.keys())[:4]
            return "{" + ", ".join(str(k) for k in keys) + ", ...}"
        return str(data)[:100]

    def _serialize_assistant_turn(self, parsed: dict) -> str:
        """Сериализовать разобранный JSON-ответ планера для истории чата."""
        try:
            return json.dumps(parsed, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(parsed)

    @staticmethod
    def _row_key(row: Any) -> str | None:
        """Естественный ключ записи для дедупликации списков tool-результатов."""
        if not isinstance(row, dict):
            return None
        for k in ("id", "iso_date", "started_at", "date"):
            v = row.get(k)
            if v is not None:
                return f"{k}={v}"
        return None

    def _merge_iter_results(
        self,
        raw_results: dict[str, Any],
        history: list[dict],
    ) -> dict[str, Any]:
        """Свернуть результаты по итерациям в одну запись на tool.

        Для списков dict'ов делаем дедуп по id / iso_date / started_at.
        Для скаляров и dict'ов — берём последнее значение.
        """
        merged: dict[str, Any] = {}
        for entry in history:
            tool = entry.get("tool", "")
            iter_n = entry.get("iteration")
            if not tool or iter_n is None:
                continue
            key = f"{tool}_iter{iter_n}"
            if key not in raw_results:
                continue
            data = raw_results[key]

            if tool not in merged:
                # Для списков кладём копию, чтобы не мутировать raw_iter_results
                merged[tool] = list(data) if isinstance(data, list) else data
                continue

            existing = merged[tool]
            if isinstance(existing, list) and isinstance(data, list):
                seen_keys: set[str] = set()
                for row in existing:
                    k = self._row_key(row)
                    if k is not None:
                        seen_keys.add(k)
                for row in data:
                    k = self._row_key(row)
                    if k is None:
                        # Нет естественного ключа — добавляем как есть
                        existing.append(row)
                    elif k not in seen_keys:
                        existing.append(row)
                        seen_keys.add(k)
            else:
                # Несписковые tools (get_user_profile, compute_recovery и т.д.) —
                # перезаписываем последним значением.
                merged[tool] = data
        return merged

    @staticmethod
    def _validate_db_args(
        tool_name: str,
        planner_args: dict,
        user_id: str,
        date_from: date,
        date_to: date,
        sport_type: str | None,
    ) -> dict | None:
        """Собрать kwargs для get_activities / get_daily_facts из planner-args.

        Игнорирует поля даты/days в planner_args (обработаны выше через
        _resolve_date_window) и неизвестные поля (Pydantic с extra='forbid'
        упадёт). Возвращает kwargs, готовый для распаковки в tool, либо None
        при ошибке валидации (с логом).

        sport_type из intent (sticky-контекст) применяется только если
        planner явно его не передал — иначе приоритет за planner-args.
        """
        # Поля, относящиеся к дате — уже обработаны _resolve_date_window.
        # tool/user_id планер не передаёт; всё что планер прислал не из
        # whitelisted-схемы — отфильтрует Pydantic с extra='forbid'.
        date_keys = {"date_from", "date_to", "days"}
        raw: dict[str, Any] = {
            k: v for k, v in (planner_args or {}).items()
            if k not in date_keys and v is not None and v != ""
        }
        raw["user_id"] = user_id
        raw["date_from"] = date_from
        raw["date_to"] = date_to
        if tool_name == "get_activities" and sport_type and "sport_type" not in raw:
            raw["sport_type"] = sport_type

        try:
            args_model = validate_tool_args(tool_name, raw)
        except ValidationError as exc:
            first = exc.errors()[0] if exc.errors() else {}
            msg = f"{first.get('loc', ('?',))[-1]}: {first.get('msg', str(exc))}"
            logger.warning(
                "PlannerAgent: args для %s не прошли валидацию: %s | raw=%r",
                tool_name, msg, raw,
            )
            return None

        # model_dump убирает None'ы и разворачивает enum'ы в их str-значения
        # (use_enum_values нет — но get_activities/get_daily_facts принимают
        # str, поэтому конвертируем явно).
        kwargs: dict[str, Any] = {}
        if isinstance(args_model, GetActivitiesArgs):
            kwargs["user_id"] = args_model.user_id
            kwargs["date_from"] = args_model.date_from
            kwargs["date_to"] = args_model.date_to
            if args_model.sport_type is not None:
                kwargs["sport_type"] = args_model.sport_type.value
            if args_model.sport_types:
                kwargs["sport_types"] = [s.value for s in args_model.sport_types]
            for f in (
                "min_distance_meters", "max_distance_meters",
                "min_duration_seconds", "max_duration_seconds",
                "min_calories", "max_calories",
                "min_avg_heart_rate", "max_avg_heart_rate",
                "min_avg_speed", "max_avg_speed",
                "min_elevation_meters", "max_elevation_meters",
                "title_contains",
            ):
                v = getattr(args_model, f)
                if v is not None:
                    kwargs[f] = v
            return kwargs

        if isinstance(args_model, GetDailyFactsArgs):
            kwargs["user_id"] = args_model.user_id
            kwargs["date_from"] = args_model.date_from
            kwargs["date_to"] = args_model.date_to
            if args_model.metrics:
                kwargs["metrics"] = [m.value for m in args_model.metrics]
            for f in (
                "min_steps", "max_steps",
                "min_calories_kcal", "max_calories_kcal",
                "min_recovery_score", "max_recovery_score",
                "min_hrv_rmssd_milli", "max_hrv_rmssd_milli",
                "min_resting_heart_rate", "max_resting_heart_rate",
                "min_sleep_total_in_bed_milli", "max_sleep_total_in_bed_milli",
                "min_water_liters", "max_water_liters",
                "min_spo2_percentage", "max_spo2_percentage",
            ):
                v = getattr(args_model, f)
                if v is not None:
                    kwargs[f] = v
            return kwargs

        return None

    @staticmethod
    def _resolve_date_window(args: dict, today: date) -> tuple[date, date]:
        """Вытащить date_from / date_to из args планировщика.

        Поддерживает три формы (по убыванию приоритета):
        — `date_from` + `date_to` (новая, основная, ISO YYYY-MM-DD);
        — только `date_from` (одиночный день);
        — `days: N` (legacy fallback — окно последних N дней до today).
        Если ничего не передано — последние 7 дней.
        """
        raw_from = args.get("date_from")
        raw_to = args.get("date_to")

        df = _parse_iso_date(raw_from)
        dt = _parse_iso_date(raw_to)

        if df is not None and dt is not None:
            if df > dt:
                df, dt = dt, df
            return df, dt
        if df is not None:
            return df, df
        if dt is not None:
            return dt, dt

        days_raw = args.get("days")
        if days_raw is not None:
            try:
                days = max(1, int(days_raw))
            except (TypeError, ValueError):
                days = 7
        else:
            days = 7
        return today - timedelta(days=days - 1), today

    async def _execute_tool(
        self,
        tool_name: str,
        args: dict,
        user_id: str,
        query: str,
        sport_type: str | None,
        db: AsyncSession,
    ) -> Any:
        """Выполнить tool-вызов планировщика."""
        import dataclasses

        today = date.today()
        date_from, date_to = self._resolve_date_window(args, today)

        if tool_name == "get_user_profile":
            res = await get_user_profile(db=db, user_id=user_id)
            return res.data

        if tool_name == "get_activities":
            kwargs = self._validate_db_args(
                tool_name="get_activities",
                planner_args=args,
                user_id=user_id,
                date_from=date_from,
                date_to=date_to,
                sport_type=sport_type,
            )
            if kwargs is None:
                return None
            res = await get_activities(db=db, **kwargs)
            return res.data

        if tool_name == "get_daily_facts":
            kwargs = self._validate_db_args(
                tool_name="get_daily_facts",
                planner_args=args,
                user_id=user_id,
                date_from=date_from,
                date_to=date_to,
                sport_type=None,
            )
            if kwargs is None:
                return None
            res = await get_daily_facts(db=db, **kwargs)
            return res.data

        if tool_name == "compute_recovery":
            from app.services.data_processing.recovery_score import compute_recovery_score
            facts_res = await get_daily_facts(
                db=db, user_id=user_id,
                date_from=today - timedelta(days=13), date_to=today,
            )
            acts_res = await get_activities(
                db=db, user_id=user_id,
                date_from=today - timedelta(days=27), date_to=today,
            )
            r = compute_recovery_score(
                daily_facts=facts_res.data or [],
                activities=acts_res.data or [],
            )
            return dataclasses.asdict(r)

        if tool_name == "check_overtraining":
            from app.services.data_processing.overtraining_detection import detect_overtraining
            facts_res = await get_daily_facts(
                db=db, user_id=user_id,
                date_from=today - timedelta(days=13), date_to=today,
            )
            acts_res = await get_activities(
                db=db, user_id=user_id,
                date_from=today - timedelta(days=27), date_to=today,
            )
            r = detect_overtraining(
                daily_facts=facts_res.data or [],
                activities=acts_res.data or [],
            )
            return dataclasses.asdict(r)

        if tool_name == "rag_retrieve":
            category = args.get("category")
            top_k = int(args.get("top_k", 3))
            res = await rag_retrieve(
                query=query,
                category=category,
                sport_type=sport_type,
                top_k=top_k,
            )
            return res.data

        logger.warning("PlannerAgent: неизвестный tool '%s'", tool_name)
        return None


# Глобальный синглтон
planner_agent = PlannerAgent()
