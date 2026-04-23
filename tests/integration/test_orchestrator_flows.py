"""Интеграционные тесты Orchestrator v2 — все 4 маршрута + safety_block (Issue #37).

Сценарии:
  1. fast_path      — "Привет" → fast_direct_answer, 1 LLM-вызов (response)
  2. tool_simple    — "Покажи мои тренировки за неделю" → tool_simple, tools called
  3. template_plan  — "Составь план на неделю" → weekly_training_plan
  4. planner_loop   — "Составь индивидуальный план тренировок" → planner, 2 итерации
  5. safety_block   — "У меня сильная боль в груди" → blocked, 0 LLM-вызовов

Каждый тест проверяет:
  * корректный итоговый ответ (совпадает с mock LLM ответом / redirect-message)
  * PipelineResult содержит ожидаемый route / stage_trace / llm_role_usage
  * llm_calls table содержит ожидаемое число записей
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.models.llm_call import LLMCall
from app.models.pending_clarification import PendingClarification
from app.pipeline.orchestrator import pipeline_orchestrator

from tests.integration.conftest import drain_background_tasks


pytestmark = pytest.mark.asyncio


async def _count_llm_calls(db) -> int:
    res = await db.execute(select(LLMCall))
    return len(res.scalars().all())


async def _fetch_llm_calls(db) -> list[LLMCall]:
    res = await db.execute(select(LLMCall))
    return list(res.scalars().all())


# ---------------------------------------------------------------------------
# 1. fast_path
# ---------------------------------------------------------------------------


async def test_fast_path_greeting(mock_ollama, mock_chroma, test_db) -> None:
    mock_ollama.set("response", "Привет! Как твои тренировки?")

    result = await pipeline_orchestrator.process_query(
        user_id="user-test",
        session_id="sess-fast",
        raw_query="Привет",
        db=test_db,
    )

    assert result.response_text == "Привет! Как твои тренировки?"
    assert result.route == "fast_direct_answer"
    assert result.fast_path is True
    assert result.blocked is False
    assert result.intent == "general_chat"
    assert result.tools_called == []
    assert result.modules_used == []
    assert result.llm_calls_count == 1
    assert result.request_id  # сгенерирован uuid

    # stage_trace содержит обязательные стадии
    stages = [s["stage"] for s in result.stage_trace]
    assert "context_build" in stages
    assert "intent_stage1" in stages
    assert "safety" in stages
    assert "routing" in stages
    assert "response_gen" in stages

    # llm_role_usage: ровно 1 вызов роли response
    assert result.llm_role_usage == {"response": 1}

    # llm_calls записаны в БД с тем же request_id
    await drain_background_tasks()
    await test_db.commit()
    calls = await _fetch_llm_calls(test_db)
    assert len(calls) == 1
    assert calls[0].role == "response"
    assert calls[0].request_id == result.request_id


# ---------------------------------------------------------------------------
# 2. tool_simple
# ---------------------------------------------------------------------------


async def test_tool_simple_data_query(mock_ollama, mock_chroma, test_db) -> None:
    mock_ollama.set("response", "За прошлую неделю у тебя 7 пробежек по 45 минут.")

    result = await pipeline_orchestrator.process_query(
        user_id="user-test",
        session_id="sess-tool",
        raw_query="Покажи мои тренировки за неделю",
        db=test_db,
    )

    assert result.response_text == "За прошлую неделю у тебя 7 пробежек по 45 минут."
    assert result.route == "tool_simple"
    assert result.fast_path is False
    assert result.blocked is False
    assert result.intent == "data_query"
    # tool_simple для data_query зовёт get_activities + get_daily_facts
    assert "get_activities" in result.tools_called
    assert "get_daily_facts" in result.tools_called
    assert result.llm_calls_count == 1

    stages = [s["stage"] for s in result.stage_trace]
    assert "tool_simple" in stages
    assert "response_gen" in stages

    assert result.llm_role_usage == {"response": 1}

    await drain_background_tasks()
    await test_db.commit()
    assert await _count_llm_calls(test_db) == 1


# ---------------------------------------------------------------------------
# 3. template_plan
# ---------------------------------------------------------------------------


async def test_template_plan_weekly(mock_ollama, mock_chroma, test_db) -> None:
    # Для intent=plan_request финальный response_generator использует роль "planner"
    mock_ollama.set("planner", "Недельный план: Пн — интервалы, Вт — восстановление, ...")

    result = await pipeline_orchestrator.process_query(
        user_id="user-test",
        session_id="sess-template",
        raw_query="Составь план на неделю",
        db=test_db,
    )

    assert result.route == "template_plan"
    assert result.template_id == "weekly_training_plan"
    assert result.intent == "plan_request"
    assert result.response_text.startswith("Недельный план:")
    assert result.llm_calls_count == 1

    # tools, вызванные внутри template — подмножество шагов шаблона
    # (get_user_profile, get_activities, compute_training_load, rag_retrieve x2)
    assert "get_activities" in result.tools_called
    assert "compute_training_load" in result.tools_called

    stages = [s["stage"] for s in result.stage_trace]
    assert "template_plan" in stages

    # Response-роль используется для финального ответа (intent=plan_request → planner)
    # См. response_generator._select_role: plan_request → planner
    assert result.llm_role_usage.get("planner", 0) == 1

    await drain_background_tasks()
    await test_db.commit()
    calls = await _fetch_llm_calls(test_db)
    assert len(calls) == 1
    assert calls[0].role == "planner"


# ---------------------------------------------------------------------------
# 4. planner_loop
# ---------------------------------------------------------------------------


async def test_planner_loop(mock_ollama, mock_chroma, test_db) -> None:
    # 2 итерации planner: 1-я — вызов tool, 2-я — final_answer, 3-й вызов (response) — финал.
    mock_ollama.set_sequence("planner", [
        '{"thought":"Сначала посмотрю историю тренировок","tool_calls":[{"tool":"get_activities","args":{"days":14}}]}',
        '{"thought":"Достаточно данных, готов ответить","final_answer":true}',
        "Индивидуальный план: понедельник — интервалы, среда — длинный бег.",
    ])

    result = await pipeline_orchestrator.process_query(
        user_id="user-test",
        session_id="sess-planner",
        # time_range задан явно — иначе orchestrator сначала уточнит период
        # (Issue #57: clarification loop для plan_request без time_range).
        raw_query="Составь индивидуальный план тренировок на месяц",
        db=test_db,
    )

    assert result.route == "planner"
    assert result.intent == "plan_request"
    assert result.blocked is False
    assert result.response_text.startswith("Индивидуальный план")
    # 2 итерации planner-loop + 1 финальный response_generator
    assert result.llm_calls_count == 3
    # tool вызывался хотя бы один раз внутри planner-loop
    assert "get_activities" in result.tools_called

    stages = [s["stage"] for s in result.stage_trace]
    assert "planner" in stages
    assert "response_gen" in stages

    # 2 вызова планировщика, 1 финальный генератор (роль planner для plan_request)
    assert result.llm_role_usage.get("planner", 0) == 3

    await drain_background_tasks()
    await test_db.commit()
    calls = await _fetch_llm_calls(test_db)
    assert len(calls) == 3
    assert all(c.role == "planner" for c in calls)


# ---------------------------------------------------------------------------
# 5. safety_block
# ---------------------------------------------------------------------------


async def test_safety_block_chest_pain(mock_ollama, mock_chroma, test_db) -> None:
    # Не задаём response — блокировка должна сработать до LLM
    result = await pipeline_orchestrator.process_query(
        user_id="user-test",
        session_id="sess-safety",
        raw_query="У меня сильная боль в груди",
        db=test_db,
    )

    assert result.blocked is True
    assert result.route == "blocked"
    assert result.safety_level == "high_priority"
    # redirect-message содержит упоминание врача
    assert "врачу" in result.response_text or "скорую" in result.response_text
    # LLM НЕ вызывался
    assert result.llm_calls_count == 0
    assert mock_ollama.calls.get("response", 0) == 0
    assert mock_ollama.calls.get("planner", 0) == 0

    stages = [s["stage"] for s in result.stage_trace]
    assert "safety" in stages
    # response_gen НЕ вызывался при blocked
    assert "response_gen" not in stages

    await drain_background_tasks()
    await test_db.commit()
    assert await _count_llm_calls(test_db) == 0
    assert result.llm_role_usage == {}


# ---------------------------------------------------------------------------
# 6. HRV today — structured summary + baseline annotation (Issue #56)
# ---------------------------------------------------------------------------


async def test_hrv_today_prompt_has_baseline_pct(mock_ollama, mock_chroma, test_db) -> None:
    """Запрос про показатели недели должен получить system-промпт с «% от базы»:
    response_generator не сериализует голый JSON, а строит structured summary
    с baseline-сравнением (Issue #56). Синтетические daily_facts в test_db
    содержат растущий HRV — baseline отличается от latest, значит в промпте
    должно появиться «% от базы»."""
    mock_ollama.set("response", "Твои показатели недели в норме.")

    result = await pipeline_orchestrator.process_query(
        user_id="user-test",
        session_id="sess-hrv",
        raw_query="Проанализируй мои показатели за неделю",
        db=test_db,
    )

    assert result.blocked is False
    assert result.intent == "data_query"
    # route может быть tool_simple (get_daily_facts) — структура передаётся в RG

    # В system-промпте должно быть упоминание % от базы (baseline comparison)
    system_prompts: list[str] = []
    for recorded in mock_ollama.prompts:
        sp = recorded.get("system_prompt")
        if sp:
            system_prompts.append(sp)

    combined = "\n".join(system_prompts)
    # Главный acceptance criterion: «% от базы» присутствует
    assert "% от базы" in combined, (
        "Ожидаем, что response_generator добавит baseline-сравнение "
        f"в system prompt, но получили:\n{combined[:2000]}"
    )
    # Голый JSON-дамп не должен уходить в промпт
    assert '"iso_date"' not in combined


# ---------------------------------------------------------------------------
# 7. Clarification loop (Issue #57) — два user-turn сценарий
# ---------------------------------------------------------------------------


async def test_clarification_two_turn_plan_request(
    mock_ollama, mock_chroma, test_db,
) -> None:
    """Первый ход: «составь план» без time_range → оркестратор задаёт
    уточняющий вопрос и сохраняет pending. Второй ход: «на неделю для бега»
    дозаполняет time_range и sport_type → нормальный flow генерации плана."""
    session_id = "sess-clarif"

    # --- Turn 1: pending должен сохраниться, вопрос задан ---
    result1 = await pipeline_orchestrator.process_query(
        user_id="user-test",
        session_id=session_id,
        raw_query="Составь план",
        db=test_db,
    )

    assert result1.route == "clarification"
    assert result1.intent == "plan_request"
    assert result1.blocked is False
    assert result1.llm_calls_count == 0
    # Вопрос про период — priority slot time_range
    assert "период" in result1.response_text.lower()
    # Никакой LLM не дёргали — ни response, ни planner
    assert mock_ollama.calls.get("response", 0) == 0
    assert mock_ollama.calls.get("planner", 0) == 0

    stages = [s["stage"] for s in result1.stage_trace]
    assert "clarification_requested" in stages

    # Pending зафиксирован в БД для этой сессии
    await test_db.commit()
    await test_db.rollback()  # сброс кеша identity map
    res = await test_db.execute(
        select(PendingClarification).where(
            PendingClarification.session_id == session_id
        )
    )
    pending = res.scalar_one_or_none()
    assert pending is not None
    assert pending.intent == "plan_request"
    assert "time_range" in pending.missing_slots

    # --- Turn 2: ответ пользователя дозаполняет слоты, план генерируется ---
    mock_ollama.set("planner", "Недельный план для бега: Пн — лёгкий бег, Ср — интервалы, ...")

    result2 = await pipeline_orchestrator.process_query(
        user_id="user-test",
        session_id=session_id,
        raw_query="за неделю для бега",
        db=test_db,
    )

    assert result2.route != "clarification"
    assert result2.intent == "plan_request"
    assert result2.blocked is False
    assert result2.response_text.startswith("Недельный план")

    # Resume → Context Builder, intent_stage1 пропускаем (уже есть pending)
    stages2 = [s["stage"] for s in result2.stage_trace]
    assert "intent_stage1" not in stages2
    assert "routing" in stages2
    assert "response_gen" in stages2

    # Pending очищен
    await test_db.commit()
    await test_db.rollback()
    res2 = await test_db.execute(
        select(PendingClarification).where(
            PendingClarification.session_id == session_id
        )
    )
    assert res2.scalar_one_or_none() is None
