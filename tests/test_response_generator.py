"""Тесты ResponseGenerator v2 (app/pipeline/response_generator.py, Issue #30)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.pipeline.context_builder import EnrichedQuery
from app.pipeline.response_generator import (
    GeneratorResult,
    ResponseGenerator,
    _SAFETY_WARNING_SUFFIX,
    _extract_rag_chunks,
    _format_conversation_history,
    _format_rag_block,
    _format_semantic_block,
    _format_structured_result,
    _format_user_profile,
    _select_role,
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

    def test_data_analysis_uses_response_role(self) -> None:
        assert _select_role("data_analysis") == "response"

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


@pytest.mark.asyncio
class TestResponseGenerator:
    """Тесты ResponseGenerator.generate v2."""

    async def test_fast_path_returns_result(self) -> None:
        generator = ResponseGenerator()
        enriched = _make_enriched_query()
        mock_llm = _make_llm_response("Отличный вопрос!")

        mock_client = MagicMock()
        mock_client.generate = AsyncMock(return_value=mock_llm)

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
        mock_client.generate = AsyncMock(return_value=mock_llm)

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

    async def test_standard_path_uses_response_role_for_data_analysis(self) -> None:
        generator = ResponseGenerator()
        enriched = _make_enriched_query()
        mock_llm = _make_llm_response("Анализ данных.")

        mock_client = MagicMock()
        mock_client.generate = AsyncMock(return_value=mock_llm)

        with patch("app.pipeline.response_generator.llm_registry") as mock_reg:
            mock_reg.get_client.return_value = mock_client
            await generator.generate(
                enriched_query=enriched,
                route="tool_simple",
                intent="data_analysis",
            )
            mock_reg.get_client.assert_called_with("response")

    async def test_rag_chunks_in_prompt_for_health_concern(self) -> None:
        """RAG-чанки добавляются в промпт для health_concern."""
        generator = ResponseGenerator()
        enriched = _make_enriched_query()
        structured = {
            "rag_retrieve_recovery_science": [
                {"text": "отдыхай между тренировками", "category": "recovery_science",
                 "confidence": "high", "score": 0.9}
            ]
        }

        captured_prompts: list[str] = []

        async def mock_generate(prompt, system_prompt=None, temperature=0.5, max_tokens=600):
            captured_prompts.append(system_prompt or "")
            return _make_llm_response("Ответ про восстановление")

        mock_client = MagicMock()
        mock_client.generate = AsyncMock(side_effect=mock_generate)

        with patch("app.pipeline.response_generator.llm_registry") as mock_reg:
            mock_reg.get_client.return_value = mock_client
            result = await generator.generate(
                enriched_query=enriched,
                route="planner",
                structured_result=structured,
                intent="health_concern",
            )

        assert len(captured_prompts) == 1
        assert "recovery_science" in captured_prompts[0]

    async def test_semantic_context_in_prompt(self) -> None:
        """Semantic context добавляется в промпт."""
        generator = ResponseGenerator()
        enriched = _make_enriched_query(
            semantic_context=[{"text": "прошлый ответ о тренировках", "score": 0.8, "timestamp": "x"}]
        )

        captured_prompts: list[str] = []

        async def mock_generate(prompt, system_prompt=None, temperature=0.5, max_tokens=600):
            captured_prompts.append(system_prompt or "")
            return _make_llm_response("Ответ")

        mock_client = MagicMock()
        mock_client.generate = AsyncMock(side_effect=mock_generate)

        with patch("app.pipeline.response_generator.llm_registry") as mock_reg:
            mock_reg.get_client.return_value = mock_client
            await generator.generate(
                enriched_query=enriched,
                route="tool_simple",
                intent="data_analysis",
            )

        assert "прошлый ответ" in captured_prompts[0]

    async def test_safety_warning_appended_for_medium(self) -> None:
        generator = ResponseGenerator()
        enriched = _make_enriched_query()

        mock_client = MagicMock()
        mock_client.generate = AsyncMock(return_value=_make_llm_response("Понял."))

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
        mock_client.generate = AsyncMock(return_value=_make_llm_response("Всё хорошо."))

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

        async def mock_generate_stream(prompt, system_prompt=None, temperature=0.7,
                                       max_tokens=300, on_token=None):
            tokens = ["Привет", " мир", "!"]
            for t in tokens:
                if on_token:
                    on_token(t)
            return _make_llm_response("Привет мир!")

        mock_client = MagicMock()
        mock_client.generate_stream = AsyncMock(side_effect=mock_generate_stream)

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
        mock_client.generate = AsyncMock(side_effect=RuntimeError("Ollama timeout"))

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
        mock_client.generate = AsyncMock(return_value=mock_llm)

        with patch("app.pipeline.response_generator.llm_registry") as mock_reg:
            mock_reg.get_client.return_value = mock_client
            result = await generator.generate(
                enriched_query=enriched,
                route="fast_direct_answer",
                intent="general_chat",
            )

        assert result.llm_response.model == "qwen2.5:14b"
        assert result.llm_response.duration_ms == 500.0
