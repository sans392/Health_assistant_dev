"""Тесты для PlannerAgent (Issue #29)."""

import asyncio
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.pipeline.planner import PlannerAgent, PlannerResult, _TOOLS_DESCRIPTION


def _make_db() -> AsyncMock:
    return AsyncMock()


def _make_llm_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.content = content
    resp.model = "qwen2.5:32b"
    resp.prompt_length = 100
    resp.response_length = len(content)
    resp.duration_ms = 500.0
    return resp


class TestPlannerJsonParsing:
    """Тесты парсинга JSON-ответов LLM."""

    def test_parse_valid_json_with_tool_calls(self) -> None:
        agent = PlannerAgent()
        content = '{"thought": "нужны данные", "tool_calls": [{"tool": "get_activities", "args": {"days": 7}}]}'
        parsed = agent._parse_response(content)
        assert parsed is not None
        assert parsed["thought"] == "нужны данные"
        assert len(parsed["tool_calls"]) == 1

    def test_parse_valid_json_final_answer(self) -> None:
        agent = PlannerAgent()
        content = '{"thought": "готово", "final_answer": true}'
        parsed = agent._parse_response(content)
        assert parsed is not None
        assert parsed.get("final_answer") is True

    def test_parse_json_in_markdown_block(self) -> None:
        agent = PlannerAgent()
        content = '```json\n{"thought": "test", "final_answer": true}\n```'
        parsed = agent._parse_response(content)
        assert parsed is not None
        assert parsed.get("final_answer") is True

    def test_parse_json_with_surrounding_text(self) -> None:
        agent = PlannerAgent()
        content = 'Вот мой ответ: {"thought": "ok", "final_answer": true} и всё.'
        parsed = agent._parse_response(content)
        assert parsed is not None

    def test_parse_invalid_json_returns_none(self) -> None:
        agent = PlannerAgent()
        parsed = agent._parse_response("это не json вообще")
        assert parsed is None


class TestPlannerToolsDescription:
    """system-prompt планировщика должен описывать инструменты по схеме."""

    def test_describes_get_activities_with_date_range(self) -> None:
        # У старой версии было `args: {"days": int}` — оставлять нельзя,
        # иначе LLM просит несуществующий параметр.
        assert "get_activities(" in _TOOLS_DESCRIPTION
        assert "date_from: YYYY-MM-DD" in _TOOLS_DESCRIPTION
        assert "date_to: YYYY-MM-DD" in _TOOLS_DESCRIPTION

    def test_describes_get_daily_facts_with_date_range(self) -> None:
        assert "get_daily_facts(" in _TOOLS_DESCRIPTION
        # Хотя бы один из аргументов date_* должен попасть в описание facts.
        # Грубая проверка — описание не пустое и упоминает оба tool'а с YYYY-MM-DD.
        idx = _TOOLS_DESCRIPTION.find("get_daily_facts(")
        snippet = _TOOLS_DESCRIPTION[idx:idx + 200]
        assert "YYYY-MM-DD" in snippet


class TestPlannerResolveDateWindow:
    """Парсер аргументов даты в _execute_tool: новые ISO + legacy days."""

    def test_explicit_date_from_to(self) -> None:
        df, dt = PlannerAgent._resolve_date_window(
            {"date_from": "2026-04-16", "date_to": "2026-04-16"},
            today=date(2026, 4, 25),
        )
        assert df == date(2026, 4, 16)
        assert dt == date(2026, 4, 16)

    def test_explicit_range(self) -> None:
        df, dt = PlannerAgent._resolve_date_window(
            {"date_from": "2026-04-10", "date_to": "2026-04-20"},
            today=date(2026, 4, 25),
        )
        assert df == date(2026, 4, 10)
        assert dt == date(2026, 4, 20)

    def test_only_date_from_collapses_to_single_day(self) -> None:
        df, dt = PlannerAgent._resolve_date_window(
            {"date_from": "2026-04-16"},
            today=date(2026, 4, 25),
        )
        assert df == dt == date(2026, 4, 16)

    def test_legacy_days_fallback(self) -> None:
        today = date(2026, 4, 25)
        df, dt = PlannerAgent._resolve_date_window({"days": 14}, today=today)
        assert dt == today
        assert df == today - timedelta(days=13)

    def test_no_args_defaults_to_7_days(self) -> None:
        today = date(2026, 4, 25)
        df, dt = PlannerAgent._resolve_date_window({}, today=today)
        assert dt == today
        assert df == today - timedelta(days=6)

    def test_inverted_dates_swapped(self) -> None:
        df, dt = PlannerAgent._resolve_date_window(
            {"date_from": "2026-04-20", "date_to": "2026-04-10"},
            today=date(2026, 4, 25),
        )
        assert df < dt


class TestPlannerSummarize:
    """Тесты метода _summarize."""

    def test_summarize_list(self) -> None:
        agent = PlannerAgent()
        result = agent._summarize([1, 2, 3, 4, 5])
        assert "5" in result

    def test_summarize_dict(self) -> None:
        agent = PlannerAgent()
        result = agent._summarize({"key1": "val1", "key2": "val2"})
        assert "{" in result

    def test_summarize_none(self) -> None:
        agent = PlannerAgent()
        result = agent._summarize(None)
        assert "нет" in result.lower()


@pytest.mark.asyncio
class TestPlannerPlanMethod:
    """Тесты метода plan() с mock LLM."""

    async def test_plan_reaches_final_answer(self) -> None:
        agent = PlannerAgent()
        db = _make_db()

        # Первая итерация: запрашиваем tool
        # Вторая итерация: final_answer
        responses = [
            '{"thought": "нужны данные о тренировках", "tool_calls": [{"tool": "get_activities", "args": {"days": 7}}]}',
            '{"thought": "данные получены, готов ответить", "final_answer": true}',
        ]
        call_count = 0

        async def mock_chat(messages, system_prompt=None, temperature=0.3, format=None):
            nonlocal call_count
            resp = _make_llm_response(responses[min(call_count, len(responses) - 1)])
            call_count += 1
            return resp

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=mock_chat)

        # Мокаем tool execution
        async def mock_execute_tool(tool_name, args, user_id, query, sport_type, db):
            return [{"activity": "run", "duration": 3600}]

        agent._execute_tool = mock_execute_tool

        with patch("app.pipeline.planner.llm_registry") as mock_registry:
            mock_registry.get_client.return_value = mock_client
            result = await agent.plan(
                query="покажи мои тренировки за неделю",
                user_id="u1",
                user_context="Профиль: тест",
                entities={},
                db=db,
            )

        assert isinstance(result, PlannerResult)
        assert result.iterations >= 1
        assert result.total_tool_calls == 1
        assert result.timeout_hit is False

    async def test_plan_uses_chat_api_with_json_format(self) -> None:
        """Планировщик вызывает /api/chat c format='json'."""
        agent = PlannerAgent()
        db = _make_db()

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_llm_response(
            '{"thought":"ok","final_answer":true}'
        ))

        with patch("app.pipeline.planner.llm_registry") as mock_registry:
            mock_registry.get_client.return_value = mock_client
            await agent.plan(
                query="тест",
                user_id="u1",
                user_context="ctx",
                entities={},
                db=db,
            )

        assert mock_client.chat.call_count == 1
        kwargs = mock_client.chat.call_args.kwargs
        assert kwargs["format"] == "json"
        # messages: первый — user с исходным запросом
        messages = kwargs["messages"]
        assert messages[0] == {"role": "user", "content": "тест"}
        # system_prompt передан
        assert kwargs.get("system_prompt")

    async def test_plan_stops_at_max_iterations(self) -> None:
        agent = PlannerAgent()
        agent._max_iterations = 3
        db = _make_db()

        # Всегда возвращаем tool_calls, никогда final_answer
        response = '{"thought": "ещё нужны данные", "tool_calls": [{"tool": "get_user_profile", "args": {}}]}'

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_llm_response(response))

        async def mock_execute_tool(tool_name, args, user_id, query, sport_type, db):
            return {"name": "Тест"}

        agent._execute_tool = mock_execute_tool

        with patch("app.pipeline.planner.llm_registry") as mock_registry:
            mock_registry.get_client.return_value = mock_client
            result = await agent.plan(
                query="помоги с тренировками",
                user_id="u1",
                user_context="контекст",
                entities={},
                db=db,
            )

        assert result.iterations == 3  # дошли до max_iterations
        assert result.timeout_hit is False

    async def test_plan_parse_error_retry(self) -> None:
        agent = PlannerAgent()
        db = _make_db()

        responses = [
            "это не json",           # parse error → retry
            '{"thought": "ok", "final_answer": true}',  # retry ответ
        ]
        call_count = 0

        async def mock_chat(messages, system_prompt=None, temperature=0.3, format=None):
            nonlocal call_count
            idx = min(call_count, len(responses) - 1)
            call_count += 1
            return _make_llm_response(responses[idx])

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=mock_chat)

        with patch("app.pipeline.planner.llm_registry") as mock_registry:
            mock_registry.get_client.return_value = mock_client
            result = await agent.plan(
                query="тест",
                user_id="u1",
                user_context="контекст",
                entities={},
                db=db,
            )

        # Второй вызов (retry) должен был вернуть final_answer
        assert mock_client.chat.call_count >= 2
        assert result.error is None or result.iterations >= 1

    async def test_tool_results_are_deduplicated_across_iterations(self) -> None:
        """Повторные вызовы одного и того же tool мёржатся в одну запись.

        Без этого Response Generator получает 5 копий daily_facts и
        раздувает system prompt дублями.
        """
        agent = PlannerAgent()
        agent._max_iterations = 3
        db = _make_db()

        response = '{"thought":"ещё","tool_calls":[{"tool":"get_daily_facts","args":{"days":7}}]}'
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_llm_response(response))

        # Возвращаем пересекающиеся наборы: один общий ключ, один уникальный
        # на итерацию — ожидаем, что в merged будет дедуп по iso_date.
        call_idx = 0

        async def mock_execute_tool(tool_name, args, user_id, query, sport_type, db):
            nonlocal call_idx
            call_idx += 1
            return [
                {"iso_date": "2026-04-13", "steps": 8099},
                {"iso_date": f"2026-04-{13 + call_idx}", "steps": 1000 * call_idx},
            ]

        agent._execute_tool = mock_execute_tool

        with patch("app.pipeline.planner.llm_registry") as mock_registry:
            mock_registry.get_client.return_value = mock_client
            result = await agent.plan(
                query="сколько шагов",
                user_id="u1",
                user_context="ctx",
                entities={},
                db=db,
            )

        assert result.iterations == 3
        assert result.total_tool_calls == 3
        # raw_iter_results хранит все три итерации отдельно
        assert len(result.raw_iter_results) == 3
        # tool_results — одна запись на tool, дедуп по iso_date
        assert set(result.tool_results.keys()) == {"get_daily_facts"}
        merged = result.tool_results["get_daily_facts"]
        dates = sorted(r["iso_date"] for r in merged)
        assert dates == ["2026-04-13", "2026-04-14", "2026-04-15", "2026-04-16"]

    async def test_tool_data_included_in_next_iteration_messages(self) -> None:
        """Сырые данные tool-результата попадают в messages следующей итерации.

        Без этого LLM не видит ранее полученные данные и зацикливается
        на повторных вызовах того же tool.
        """
        agent = PlannerAgent()
        agent._max_iterations = 2
        db = _make_db()

        messages_seen: list[list[dict]] = []

        async def mock_chat(messages, system_prompt=None, temperature=0.3, format=None):
            # Сохраняем снимок списка messages на момент вызова.
            messages_seen.append([dict(m) for m in messages])
            if len(messages_seen) == 1:
                return _make_llm_response(
                    '{"thought":"нужны шаги","tool_calls":'
                    '[{"tool":"get_daily_facts","args":{"days":1}}]}'
                )
            return _make_llm_response('{"thought":"ok","final_answer":true}')

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=mock_chat)

        async def mock_execute_tool(tool_name, args, user_id, query, sport_type, db):
            return [{"iso_date": "2026-04-18", "steps": 12345}]

        agent._execute_tool = mock_execute_tool

        with patch("app.pipeline.planner.llm_registry") as mock_registry:
            mock_registry.get_client.return_value = mock_client
            await agent.plan(
                query="сколько шагов сегодня",
                user_id="u1",
                user_context="ctx",
                entities={},
                db=db,
            )

        assert len(messages_seen) >= 2
        # Во второй итерации история должна содержать assistant-ответ планера
        # и user-сообщение с результатами tools.
        second_iter_contents = "\n".join(m["content"] for m in messages_seen[1])
        assert "12345" in second_iter_contents
        assert "2026-04-18" in second_iter_contents
        # Есть assistant-ход планера в истории второй итерации
        assert any(m["role"] == "assistant" for m in messages_seen[1])

    async def test_plan_compresses_long_tool_result_before_sending_to_llm(self) -> None:
        """Issue #58: длинные list-результаты сжимаются перед подачей планировщику.

        Убеждаемся, что во второй итерации:
        - в истории НЕТ сырых 50 записей (raw blob превратился бы в raw text);
        - присутствует поле `compressed: true` и `full_count: 50`;
        - присутствует блок summary;
        - присутствует sample ограниченного размера (5).
        """
        agent = PlannerAgent()
        agent._max_iterations = 2
        db = _make_db()

        messages_seen: list[list[dict]] = []

        async def mock_chat(messages, system_prompt=None, temperature=0.3, format=None):
            messages_seen.append([dict(m) for m in messages])
            if len(messages_seen) == 1:
                return _make_llm_response(
                    '{"thought":"нужны активности","tool_calls":'
                    '[{"tool":"get_activities","args":{"days":60}}]}'
                )
            return _make_llm_response('{"thought":"ок","final_answer":true}')

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=mock_chat)

        async def mock_execute_tool(tool_name, args, user_id, query, sport_type, db):
            # Возвращаем 50 активностей с уникальными id и различной интенсивностью.
            return [
                {
                    "id": i,
                    "user_id": "u1",
                    "title": f"run-{i}",
                    "sport_type": "running",
                    "distance_meters": 5000 + i,
                    "duration_seconds": 1800,
                    "start_time": f"2026-03-{1 + (i % 28):02d}T10:00:00",
                    "end_time": f"2026-03-{1 + (i % 28):02d}T10:30:00",
                    "avg_speed": 3.0,
                    "calories": 100 + i * 5,
                    "avg_heart_rate": 140,
                    "source": "seed",
                }
                for i in range(50)
            ]

        agent._execute_tool = mock_execute_tool

        with patch("app.pipeline.planner.llm_registry") as mock_registry:
            mock_registry.get_client.return_value = mock_client
            result = await agent.plan(
                query="разбери мою нагрузку за 2 месяца",
                user_id="u1",
                user_context="ctx",
                entities={},
                db=db,
            )

        # Планер всё ещё видит все 50 записей в merged tool_results (для response).
        assert len(result.tool_results["get_activities"]) == 50

        # А в истории сообщений LLM — сжатая версия.
        assert len(messages_seen) >= 2
        second_iter_user_turns = [
            m["content"] for m in messages_seen[1] if m["role"] == "user"
        ]
        # Последнее user-сообщение — результаты tool.
        tool_turn_text = second_iter_user_turns[-1]
        assert '"compressed": true' in tool_turn_text
        assert '"full_count": 50' in tool_turn_text
        assert '"shown": 5' in tool_turn_text
        assert '"summary"' in tool_turn_text
        assert '"sample"' in tool_turn_text
        # Финальный ответ планера (на второй итерации) успешно получен.
        assert result.error is None

    async def test_execute_tool_passes_sport_type_from_planner_args(self) -> None:
        """Регрессия: planner присылает sport_type в args — он должен дойти до tool.

        До правки `_execute_tool` ловил только date_from/to и роняла фильтры.
        """
        agent = PlannerAgent()
        captured: dict = {}

        async def fake_get_activities(**kwargs):
            captured.update(kwargs)
            from app.tools.db_tools import ToolResult
            return ToolResult(
                tool_name="get_activities", success=True, data=[], error=None,
            )

        with patch("app.pipeline.planner.get_activities", new=fake_get_activities):
            await agent._execute_tool(
                tool_name="get_activities",
                args={
                    "date_from": "2026-04-19",
                    "date_to": "2026-04-25",
                    "sport_type": "running",
                    "min_distance_meters": 8000,
                },
                user_id="u1",
                query="покажи длинные пробежки",
                sport_type=None,
                db=_make_db(),
            )

        assert captured["sport_type"] == "running"
        assert captured["min_distance_meters"] == 8000
        assert captured["date_from"] == date(2026, 4, 19)
        assert captured["date_to"] == date(2026, 4, 25)

    async def test_execute_tool_falls_back_to_intent_sport_type(self) -> None:
        """Если planner не передал sport_type — берём из intent (sticky)."""
        agent = PlannerAgent()
        captured: dict = {}

        async def fake_get_activities(**kwargs):
            captured.update(kwargs)
            from app.tools.db_tools import ToolResult
            return ToolResult(
                tool_name="get_activities", success=True, data=[], error=None,
            )

        with patch("app.pipeline.planner.get_activities", new=fake_get_activities):
            await agent._execute_tool(
                tool_name="get_activities",
                args={"date_from": "2026-04-19", "date_to": "2026-04-25"},
                user_id="u1", query="покажи пробежки",
                sport_type="running",
                db=_make_db(),
            )

        assert captured["sport_type"] == "running"

    async def test_execute_tool_planner_sport_type_overrides_intent(self) -> None:
        """Если planner явно передал sport_type — он приоритетнее intent."""
        agent = PlannerAgent()
        captured: dict = {}

        async def fake_get_activities(**kwargs):
            captured.update(kwargs)
            from app.tools.db_tools import ToolResult
            return ToolResult(
                tool_name="get_activities", success=True, data=[], error=None,
            )

        with patch("app.pipeline.planner.get_activities", new=fake_get_activities):
            await agent._execute_tool(
                tool_name="get_activities",
                args={
                    "date_from": "2026-04-19", "date_to": "2026-04-25",
                    "sport_type": "cycling",
                },
                user_id="u1", query="вело",
                sport_type="running",  # intent sticky
                db=_make_db(),
            )

        assert captured["sport_type"] == "cycling"

    async def test_execute_tool_get_daily_facts_passes_filters(self) -> None:
        agent = PlannerAgent()
        captured: dict = {}

        async def fake_get_daily_facts(**kwargs):
            captured.update(kwargs)
            from app.tools.db_tools import ToolResult
            return ToolResult(
                tool_name="get_daily_facts", success=True, data=[], error=None,
            )

        with patch("app.pipeline.planner.get_daily_facts", new=fake_get_daily_facts):
            await agent._execute_tool(
                tool_name="get_daily_facts",
                args={
                    "date_from": "2026-04-19", "date_to": "2026-04-25",
                    "min_steps": 10000, "max_recovery_score": 90,
                },
                user_id="u1", query="дни с 10к шагов",
                sport_type=None,
                db=_make_db(),
            )

        assert captured["min_steps"] == 10000
        assert captured["max_recovery_score"] == 90

    async def test_execute_tool_invalid_args_returns_none(self) -> None:
        """Невалидные args (например, недопустимый sport_type) → None + WARN."""
        agent = PlannerAgent()

        with patch(
            "app.pipeline.planner.get_activities",
            new=AsyncMock(),
        ) as mock_call:
            data = await agent._execute_tool(
                tool_name="get_activities",
                args={
                    "date_from": "2026-04-19", "date_to": "2026-04-25",
                    "sport_type": "квиддич",
                },
                user_id="u1", query="?", sport_type=None,
                db=_make_db(),
            )

        assert data is None
        mock_call.assert_not_called()

    async def test_plan_timeout_sets_flag(self) -> None:
        agent = PlannerAgent()
        agent._timeout = 0.0001  # Ультра-малый таймаут

        db = _make_db()

        # LLM возвращает tool_calls (НЕ final_answer), чтобы цикл перешёл
        # на итерацию 2, где проверка elapsed >= self._timeout выставит
        # timeout_hit. Если бы ответ был final_answer — планер успешно
        # завершился бы на итерации 1 без срабатывания таймаута.
        async def slow_chat(messages, system_prompt=None, temperature=0.3, format=None):
            await asyncio.sleep(0.05)
            return _make_llm_response(
                '{"thought":"нужны данные","tool_calls":[{"tool":"get_user_profile","args":{}}]}'
            )

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=slow_chat)

        with patch("app.pipeline.planner.llm_registry") as mock_registry:
            mock_registry.get_client.return_value = mock_client
            # Подменяем tool execution, чтобы не ходить в реальные tools
            with patch.object(agent, "_execute_tool", AsyncMock(return_value={"ok": True})):
                result = await agent.plan(
                    query="тест",
                    user_id="u1",
                    user_context="контекст",
                    entities={},
                    db=db,
                )

        assert result.timeout_hit is True
