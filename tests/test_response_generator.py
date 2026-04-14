"""Тесты ResponseGenerator (app/pipeline/response_generator.py).

Используют мок Ollama — без реального HTTP.
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.pipeline.context_builder import EnrichedQuery
from app.pipeline.response_generator import (
    GeneratorResult,
    ResponseGenerator,
    _SAFETY_WARNING_SUFFIX,
    _format_conversation_history,
    _format_structured_result,
    _format_user_profile,
)
from app.services.llm_service import LLMResponse


def _make_llm_response(content: str = "Тестовый ответ") -> LLMResponse:
    return LLMResponse(
        content=content,
        model="qwen2.5:7b",
        prompt_length=100,
        response_length=len(content),
        duration_ms=500.0,
    )


def _make_enriched_query(
    normalized_text: str = "покажи мои тренировки за неделю",
    user_profile: dict | None = None,
    conversation_history: list | None = None,
) -> EnrichedQuery:
    return EnrichedQuery(
        raw_text=normalized_text,
        normalized_text=normalized_text,
        user_profile=user_profile or {
            "name": "Тест",
            "age": 30,
            "weight_kg": 75.0,
            "height_cm": 180.0,
            "experience_level": "intermediate",
            "training_goals": ["похудение"],
            "preferred_sports": ["running"],
        },
        conversation_history=conversation_history or [],
        metadata={"session_id": "s-1", "user_id": "u-1"},
    )


class TestFormatHelpers:
    """Тесты вспомогательных функций форматирования."""

    def test_format_user_profile_none(self) -> None:
        text = _format_user_profile(None)
        assert "не задан" in text

    def test_format_user_profile_with_data(self) -> None:
        profile = {
            "name": "Иван", "age": 25, "weight_kg": 80.0, "height_cm": 175.0,
            "experience_level": "beginner", "training_goals": ["сила"],
            "preferred_sports": ["gym"],
        }
        text = _format_user_profile(profile)
        assert "Иван" in text
        assert "25" in text
        assert "80" in text

    def test_format_conversation_history_empty(self) -> None:
        text = _format_conversation_history([])
        assert "пуст" in text.lower()

    def test_format_conversation_history_with_messages(self) -> None:
        history = [
            {"role": "user", "content": "Привет"},
            {"role": "assistant", "content": "Здравствуй!"},
        ]
        text = _format_conversation_history(history)
        assert "Привет" in text
        assert "Здравствуй" in text

    def test_format_conversation_history_truncates_to_5(self) -> None:
        history = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        text = _format_conversation_history(history)
        # Последние 5 сообщений
        assert "msg 9" in text
        assert "msg 0" not in text

    def test_format_structured_result_none(self) -> None:
        text = _format_structured_result(None)
        assert "нет" in text.lower()

    def test_format_structured_result_with_data(self) -> None:
        data = {"total_activities": 5, "total_calories": 2000}
        text = _format_structured_result(data)
        assert "total_activities" in text
        assert "5" in text


@pytest.mark.asyncio
class TestResponseGenerator:
    """Тесты ResponseGenerator.generate."""

    async def test_fast_path_returns_result(self) -> None:
        generator = ResponseGenerator()
        enriched = _make_enriched_query()
        mock_llm = _make_llm_response("Отличный вопрос!")

        with patch("app.pipeline.response_generator.ollama_client") as mock_client:
            mock_client.generate = AsyncMock(return_value=mock_llm)
            result = await generator.generate(
                enriched_query=enriched,
                route="fast_direct_answer",
                safety_level="ok",
            )

        assert isinstance(result, GeneratorResult)
        assert result.content == "Отличный вопрос!"
        assert result.route == "fast_direct_answer"

    async def test_standard_path_returns_result(self) -> None:
        generator = ResponseGenerator()
        enriched = _make_enriched_query()
        mock_llm = _make_llm_response("За неделю: 5 тренировок, 2500 kcal.")

        with patch("app.pipeline.response_generator.ollama_client") as mock_client:
            mock_client.generate = AsyncMock(return_value=mock_llm)
            result = await generator.generate(
                enriched_query=enriched,
                route="tool_simple",
                structured_result={"total_activities": 5, "total_calories": 2500},
                safety_level="ok",
            )

        assert result.route == "tool_simple"
        assert "5" in result.content or "тренировок" in result.content

    async def test_safety_warning_appended_for_medium(self) -> None:
        generator = ResponseGenerator()
        enriched = _make_enriched_query()
        mock_llm = _make_llm_response("Понял ваш запрос.")

        with patch("app.pipeline.response_generator.ollama_client") as mock_client:
            mock_client.generate = AsyncMock(return_value=mock_llm)
            result = await generator.generate(
                enriched_query=enriched,
                route="health_concern",
                safety_level="medium_priority",
            )

        assert _SAFETY_WARNING_SUFFIX in result.content

    async def test_no_safety_warning_for_ok(self) -> None:
        generator = ResponseGenerator()
        enriched = _make_enriched_query()
        mock_llm = _make_llm_response("Всё хорошо.")

        with patch("app.pipeline.response_generator.ollama_client") as mock_client:
            mock_client.generate = AsyncMock(return_value=mock_llm)
            result = await generator.generate(
                enriched_query=enriched,
                route="fast_direct_answer",
                safety_level="ok",
            )

        assert _SAFETY_WARNING_SUFFIX not in result.content

    async def test_llm_response_stored_in_result(self) -> None:
        generator = ResponseGenerator()
        enriched = _make_enriched_query()
        mock_llm = _make_llm_response("Ответ LLM.")

        with patch("app.pipeline.response_generator.ollama_client") as mock_client:
            mock_client.generate = AsyncMock(return_value=mock_llm)
            result = await generator.generate(
                enriched_query=enriched,
                route="fast_direct_answer",
            )

        assert result.llm_response.model == "qwen2.5:7b"
        assert result.llm_response.duration_ms == 500.0

    async def test_no_profile_handled_gracefully(self) -> None:
        generator = ResponseGenerator()
        enriched = _make_enriched_query(user_profile=None)
        mock_llm = _make_llm_response("Ответ без профиля.")

        with patch("app.pipeline.response_generator.ollama_client") as mock_client:
            mock_client.generate = AsyncMock(return_value=mock_llm)
            result = await generator.generate(
                enriched_query=enriched,
                route="fast_direct_answer",
            )

        # Не должно падать
        assert result.content == "Ответ без профиля."
