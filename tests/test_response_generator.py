"""Тесты ResponseGenerator v2 (app/pipeline/response_generator.py, Issue #30)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.pipeline.context_builder import EnrichedQuery
from app.pipeline.response_generator import (
    GeneratorResult,
    ResponseGenerator,
    _SAFETY_WARNING_SUFFIX,
    _build_context_system,
    _extract_rag_chunks,
    _format_conversation_history,
    _format_rag_block,
    _format_semantic_block,
    _format_structured_result,
    _format_user_profile,
    _select_role,
    _strip_presented_keys,
)
from app.services.llm_service import LLMResponse


def _make_llm_response(content: str = "Тестовый ответ") -> LLMResponse:
    return LLMResponse(
        content=content,
        model="qwen2.5:14b",
        prompt_length=100,
        response_length=len(content),
        duration_ms=500.0,
    )


def _make_enriched_query(
    normalized_text: str = "покажи мои тренировки за неделю",
    user_profile: dict | None = None,
    conversation_history: list | None = None,
    semantic_context: list | None = None,
    knowledge_context: list | None = None,
) -> EnrichedQuery:
    return EnrichedQuery(
        raw_text=normalized_text,
        normalized_text=normalized_text,
        user_profile=user_profile or {
            "name": "Тест", "age": 30, "weight_kg": 75.0, "height_cm": 180.0,
            "experience_level": "intermediate", "training_goals": ["похудение"],
            "preferred_sports": ["running"],
        },
        conversation_history=conversation_history or [],
        semantic_context=semantic_context or [],
        knowledge_context=knowledge_context or [],
        metadata={"session_id": "s-1", "user_id": "u-1"},
    )


class TestSelectRole:
    """Тесты выбора роли LLM по intent."""

    def test_plan_request_uses_planner_role(self) -> None:
        assert _select_role("plan_request") == "planner"

    def test_health_concern_uses_response_role(self) -> None:
        assert _select_role("health_concern") == "response"

    def test_data_query_uses_response_role(self) -> None:
        assert _select_role("data_query") == "response"

    def test_general_chat_uses_response_role(self) -> None:
        assert _select_role("general_chat") == "response"


class TestFormatHelpers:
    """Тесты вспомогательных функций форматирования."""

    def test_format_user_profile_none(self) -> None:
        assert "не задан" in _format_user_profile(None)

    def test_format_user_profile_with_data(self) -> None:
        profile = {
            "name": "Иван", "age": 25, "weight_kg": 80.0, "height_cm": 175.0,
            "experience_level": "beginner", "training_goals": ["сила"],
            "preferred_sports": ["gym"],
        }
        text = _format_user_profile(profile)
        assert "Иван" in text and "25" in text and "80" in text

    def test_format_conversation_history_empty(self) -> None:
        assert "пуст" in _format_conversation_history([]).lower()

    def test_format_conversation_history_truncates_to_5(self) -> None:
        history = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        text = _format_conversation_history(history)
        assert "msg 9" in text
        assert "msg 0" not in text

    def test_format_structured_result_none(self) -> None:
        assert "нет" in _format_structured_result(None).lower()

    def test_format_structured_result_with_data(self) -> None:
        data = {"total_activities": 5, "total_calories": 2000}
        text = _format_structured_result(data)
        assert "total_activities" in text and "5" in text


class TestRagFormatting:
    """Тесты форматирования RAG-контекста."""

    def test_extract_rag_chunks_from_structured_result(self) -> None:
        structured = {
            "rag_retrieve_recovery_science": [
                {"text": "сон важен", "category": "recovery_science", "confidence": "high", "score": 0.9}
            ],
            "compute_recovery": {"score": 72},
        }
        chunks = _extract_rag_chunks(structured)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "сон важен"

    def test_extract_rag_chunks_empty_result(self) -> None:
        assert _extract_rag_chunks(None) == []
        assert _extract_rag_chunks({}) == []

    def test_extract_rag_chunks_from_tool_data(self) -> None:
        """RAG-чанки внутри tool_data (маршрут tool_simple) тоже извлекаются."""
        structured = {
            "tool_data": {
                "rag_retrieve": [
                    {"text": "контент", "category": "training_principles",
                     "confidence": "high", "score": 0.8}
                ],
                "get_activities": [{"id": 1}],
            }
        }
        chunks = _extract_rag_chunks(structured)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "контент"

    def test_format_rag_block_empty(self) -> None:
        assert _format_rag_block([]) == ""

    def test_format_rag_block_with_chunks(self) -> None:
        chunks = [
            {"text": "тренируйся 3 раза в неделю", "category": "training_principles",
             "confidence": "high", "score": 0.9}
        ]
        block = _format_rag_block(chunks)
        assert "RAG" in block
        assert "training_principles" in block

    def test_format_semantic_block_empty(self) -> None:
        assert _format_semantic_block([]) == ""

    def test_format_semantic_block_with_data(self) -> None:
        semantic = [{"text": "прошлый ответ про тренировки", "score": 0.8, "timestamp": "2026-04-10"}]
        block = _format_semantic_block(semantic)
        assert "прошлые" in block.lower() or "прошлый" in block.lower()


class TestStripPresentedKeys:
    """Тесты фильтрации ключей, уже отрисованных в отдельных system-блоках."""

    def test_strips_rag_keys_when_strip_rag_true(self) -> None:
        """rag_retrieve* вырезаются, когда их уже показали в RAG-блоке."""
        structured = {
            "rag_retrieve_training_principles": [{"text": "x"}],
            "rag_retrieve": [{"text": "y"}],
            "compute_recovery": {"score": 72},
        }
        cleaned = _strip_presented_keys(structured, strip_rag=True)
        assert cleaned == {"compute_recovery": {"score": 72}}

    def test_keeps_rag_keys_when_strip_rag_false(self) -> None:
        """Если RAG-блок не сформирован — ключи остаются в JSON."""
        structured = {
            "rag_retrieve_training_principles": [{"text": "x"}],
            "compute_recovery": {"score": 72},
        }
        cleaned = _strip_presented_keys(structured, strip_rag=False)
        assert "rag_retrieve_training_principles" in cleaned
        assert "compute_recovery" in cleaned

    def test_handles_nested_tool_data(self) -> None:
        """Фильтрация работает внутри tool_data (структура tool_simple)."""
        structured = {
            "tool_data": {
                "rag_retrieve": [{"text": "x"}],
                "get_activities": [{"id": 1}],
            },
        }
        cleaned = _strip_presented_keys(structured, strip_rag=True)
        assert cleaned == {"tool_data": {"get_activities": [{"id": 1}]}}

    def test_drops_empty_tool_data(self) -> None:
        """Если tool_data опустел после фильтрации — сам ключ удаляем."""
        structured = {
            "tool_data": {
                "rag_retrieve": [{"text": "x"}],
            },
        }
        cleaned = _strip_presented_keys(structured, strip_rag=True)
        assert cleaned is None

    def test_returns_none_for_empty_input(self) -> None:
        assert _strip_presented_keys(None, strip_rag=True) is None
        assert _strip_presented_keys({}, strip_rag=True) == {} or \
               _strip_presented_keys({}, strip_rag=True) is None

    def test_build_context_system_avoids_rag_duplication(self) -> None:
        """Спец-блок RAG сформирован → rag_retrieve* НЕ должен появляться в JSON."""
        rag_chunks = [
            {"text": "sleep matters", "category": "recovery_science",
             "confidence": "high", "score": 0.9}
        ]
        structured = {
            "rag_retrieve_recovery_science": rag_chunks,
            "compute_recovery": {"score": 72},
        }
        system = _build_context_system(
            structured_result=structured,
            rag_chunks=rag_chunks,
            semantic_context=[],
        )
        assert system is not None
        assert "## Релевантные знания (RAG)" in system
        assert "## Результаты анализа" in system
        assert "compute_recovery" in system
        # Ключ RAG не должен попасть в блок «Результаты анализа»
        assert "rag_retrieve_recovery_science" not in system


@pytest.mark.asyncio
class TestResponseGenerator:
    """Тесты ResponseGenerator.generate v2."""

    async def test_fast_path_returns_result(self) -> None:
        generator = ResponseGenerator()
        enriched = _make_enriched_query()
        mock_llm = _make_llm_response("Отличный вопрос!")

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=mock_llm)

        with patch("app.pipeline.response_generator.llm_registry") as mock_reg:
            mock_reg.get_client.return_value = mock_client
            result = await generator.generate(
                enriched_query=enriched,
                route="fast_direct_answer",
                intent="general_chat",
            )

        assert isinstance(result, GeneratorResult)
        assert result.content == "Отличный вопрос!"
        assert result.route == "fast_direct_answer"

    async def test_standard_path_uses_correct_role_for_plan_request(self) -> None:
        generator = ResponseGenerator()
        enriched = _make_enriched_query()
        mock_llm = _make_llm_response("Вот ваш план.")

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=mock_llm)

        with patch("app.pipeline.response_generator.llm_registry") as mock_reg:
            mock_reg.get_client.return_value = mock_client
            await generator.generate(
                enriched_query=enriched,
                route="template_plan",
                structured_result={},
                intent="plan_request",
            )
            # Для plan_request должен запрашиваться "planner" клиент
            mock_reg.get_client.assert_called_with("planner")

    async def test_standard_path_uses_response_role_for_data_query(self) -> None:
        generator = ResponseGenerator()
        enriched = _make_enriched_query()
        mock_llm = _make_llm_response("Анализ данных.")

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=mock_llm)

        with patch("app.pipeline.response_generator.llm_registry") as mock_reg:
            mock_reg.get_client.return_value = mock_client
            await generator.generate(
                enriched_query=enriched,
                route="tool_simple",
                intent="data_query",
            )
            mock_reg.get_client.assert_called_with("response")

    async def test_rag_chunks_in_system_block_for_health_concern(self) -> None:
        """RAG-чанки добавляются в отдельный system-блок для health_concern."""
        generator = ResponseGenerator()
        enriched = _make_enriched_query()
        structured = {
            "rag_retrieve_recovery_science": [
                {"text": "отдыхай между тренировками", "category": "recovery_science",
                 "confidence": "high", "score": 0.9}
            ]
        }

        captured: list[dict] = []

        async def mock_chat(messages, system_prompt=None, system_prompts=None,
                            temperature=0.5, max_tokens=600):
            captured.append({
                "messages": messages,
                "system_prompts": system_prompts or [],
            })
            return _make_llm_response("Ответ про восстановление")

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=mock_chat)

        with patch("app.pipeline.response_generator.llm_registry") as mock_reg:
            mock_reg.get_client.return_value = mock_client
            await generator.generate(
                enriched_query=enriched,
                route="planner",
                structured_result=structured,
                intent="health_concern",
            )

        assert len(captured) == 1
        combined_system = "\n".join(captured[0]["system_prompts"])
        assert "recovery_science" in combined_system

    async def test_semantic_context_in_system_block(self) -> None:
        """Semantic context добавляется в system-блок."""
        generator = ResponseGenerator()
        enriched = _make_enriched_query(
            semantic_context=[{"text": "прошлый ответ о тренировках", "score": 0.8, "timestamp": "x"}]
        )

        captured: list[dict] = []

        async def mock_chat(messages, system_prompt=None, system_prompts=None,
                            temperature=0.5, max_tokens=600):
            captured.append({"system_prompts": system_prompts or []})
            return _make_llm_response("Ответ")

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=mock_chat)

        with patch("app.pipeline.response_generator.llm_registry") as mock_reg:
            mock_reg.get_client.return_value = mock_client
            await generator.generate(
                enriched_query=enriched,
                route="tool_simple",
                intent="data_query",
            )

        combined_system = "\n".join(captured[0]["system_prompts"])
        assert "прошлый ответ" in combined_system

    async def test_standard_path_splits_system_into_three_blocks(self) -> None:
        """Standard path должен формировать 3 отдельных system-сообщения:
        базовый промпт, профиль, контекст (tools/RAG/semantic)."""
        generator = ResponseGenerator()
        enriched = _make_enriched_query()
        structured = {"compute_recovery": {"score": 72}}

        captured: list[list[str]] = []

        async def mock_chat(messages, system_prompt=None, system_prompts=None,
                            temperature=0.5, max_tokens=600):
            captured.append(list(system_prompts or []))
            return _make_llm_response("ok")

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=mock_chat)

        with patch("app.pipeline.response_generator.llm_registry") as mock_reg:
            mock_reg.get_client.return_value = mock_client
            await generator.generate(
                enriched_query=enriched,
                route="tool_simple",
                structured_result=structured,
                intent="data_query",
            )

        assert len(captured) == 1
        system_prompts = captured[0]
        assert len(system_prompts) == 3
        assert "фитнес-ассистент" in system_prompts[0].lower()
        assert "пользователе" in system_prompts[1].lower() or "профиль" in system_prompts[1].lower()
        assert "compute_recovery" in system_prompts[2]

    async def test_conversation_history_passed_as_user_assistant_messages(self) -> None:
        """История диалога должна передаваться как role=user/assistant, а не в system."""
        generator = ResponseGenerator()
        enriched = _make_enriched_query(
            conversation_history=[
                {"role": "user", "content": "Привет", "timestamp": "t1"},
                {"role": "assistant", "content": "Здравствуй!", "timestamp": "t2"},
            ],
            normalized_text="как мои дела",
        )

        captured: list[list[dict]] = []

        async def mock_chat(messages, system_prompt=None, system_prompts=None,
                            temperature=0.7, max_tokens=300):
            captured.append([dict(m) for m in messages])
            return _make_llm_response("Норм")

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=mock_chat)

        with patch("app.pipeline.response_generator.llm_registry") as mock_reg:
            mock_reg.get_client.return_value = mock_client
            await generator.generate(
                enriched_query=enriched,
                route="fast_direct_answer",
                intent="general_chat",
            )

        messages = captured[0]
        # Ожидаем: user="Привет", assistant="Здравствуй!", user="как мои дела"
        assert messages[0] == {"role": "user", "content": "Привет"}
        assert messages[1] == {"role": "assistant", "content": "Здравствуй!"}
        assert messages[-1] == {"role": "user", "content": "как мои дела"}

    async def test_safety_warning_appended_for_medium(self) -> None:
        generator = ResponseGenerator()
        enriched = _make_enriched_query()

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_llm_response("Понял."))

        with patch("app.pipeline.response_generator.llm_registry") as mock_reg:
            mock_reg.get_client.return_value = mock_client
            result = await generator.generate(
                enriched_query=enriched,
                route="fast_direct_answer",
                safety_level="medium_priority",
                intent="general_chat",
            )

        assert _SAFETY_WARNING_SUFFIX in result.content

    async def test_no_safety_warning_for_ok(self) -> None:
        generator = ResponseGenerator()
        enriched = _make_enriched_query()

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_llm_response("Всё хорошо."))

        with patch("app.pipeline.response_generator.llm_registry") as mock_reg:
            mock_reg.get_client.return_value = mock_client
            result = await generator.generate(
                enriched_query=enriched,
                route="fast_direct_answer",
                safety_level="ok",
                intent="general_chat",
            )

        assert _SAFETY_WARNING_SUFFIX not in result.content

    async def test_streaming_callback_invoked(self) -> None:
        """on_token callback вызывается при streaming."""
        generator = ResponseGenerator()
        enriched = _make_enriched_query()
        tokens_received: list[str] = []

        async def mock_chat_stream(messages, system_prompt=None, system_prompts=None,
                                   temperature=0.7, max_tokens=300, on_token=None):
            tokens = ["Привет", " мир", "!"]
            for t in tokens:
                if on_token:
                    on_token(t)
            return _make_llm_response("Привет мир!")

        mock_client = MagicMock()
        mock_client.chat_stream = AsyncMock(side_effect=mock_chat_stream)

        def collect_token(token: str) -> None:
            tokens_received.append(token)

        with patch("app.pipeline.response_generator.llm_registry") as mock_reg:
            mock_reg.get_client.return_value = mock_client
            result = await generator.generate(
                enriched_query=enriched,
                route="fast_direct_answer",
                intent="general_chat",
                on_token=collect_token,
            )

        assert len(tokens_received) == 3
        assert result.content == "Привет мир!"

    async def test_fallback_on_llm_error(self) -> None:
        generator = ResponseGenerator()
        enriched = _make_enriched_query()

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=RuntimeError("Ollama timeout"))

        with patch("app.pipeline.response_generator.llm_registry") as mock_reg:
            mock_reg.get_client.return_value = mock_client
            result = await generator.generate(
                enriched_query=enriched,
                route="fast_direct_answer",
                intent="general_chat",
            )

        assert "не могу сгенерировать" in result.content

    async def test_llm_response_stored_in_result(self) -> None:
        generator = ResponseGenerator()
        enriched = _make_enriched_query()
        mock_llm = _make_llm_response("Ответ LLM.")

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=mock_llm)

        with patch("app.pipeline.response_generator.llm_registry") as mock_reg:
            mock_reg.get_client.return_value = mock_client
            result = await generator.generate(
                enriched_query=enriched,
                route="fast_direct_answer",
                intent="general_chat",
            )

        assert result.llm_response.model == "qwen2.5:14b"
        assert result.llm_response.duration_ms == 500.0
