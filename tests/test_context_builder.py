"""Тесты для модуля ContextBuilder."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.pipeline.context_builder import ContextBuilder, EnrichedQuery, _normalize_text


class TestNormalizeText:
    """Тесты нормализации текста."""

    def test_lowercase(self) -> None:
        assert _normalize_text("ПРИВЕТ") == "привет"

    def test_strip_whitespace(self) -> None:
        assert _normalize_text("  привет  ") == "привет"

    def test_collapse_spaces(self, ) -> None:
        assert _normalize_text("привет   мир") == "привет мир"

    def test_preserves_original(self) -> None:
        original = "Покажи Тренировки"
        normalized = _normalize_text(original)
        assert original != normalized
        assert normalized == "покажи тренировки"


class TestContextBuilder:
    """Тесты ContextBuilder с mock БД."""

    @pytest.fixture
    def builder(self) -> ContextBuilder:
        return ContextBuilder()

    @pytest.fixture
    def mock_db(self) -> AsyncMock:
        return AsyncMock()

    def _make_message(self, role: str, content: str, index: int) -> MagicMock:
        msg = MagicMock()
        msg.role = role
        msg.content = content
        msg.order_index = index
        msg.created_at = datetime(2026, 1, 1, 12, 0, 0)
        return msg

    def _make_profile(self) -> MagicMock:
        profile = MagicMock()
        profile.user_id = "user-1"
        profile.name = "Иван"
        profile.age = 30
        profile.weight_kg = 75.0
        profile.height_cm = 180.0
        profile.gender = "male"
        profile.max_heart_rate = 190
        profile.resting_heart_rate = 60
        profile.training_goals = ["похудение"]
        profile.experience_level = "intermediate"
        profile.injuries = []
        profile.chronic_conditions = []
        profile.preferred_sports = ["бег", "велосипед"]
        return profile

    @pytest.mark.asyncio
    async def test_build_returns_enriched_query(
        self, builder: ContextBuilder, mock_db: AsyncMock
    ) -> None:
        # История: 2 сообщения
        msg1 = self._make_message("user", "Привет", 0)
        msg2 = self._make_message("assistant", "Здравствуйте!", 1)

        execute_result = MagicMock()
        execute_result.scalars.return_value.all.return_value = [msg2, msg1]

        # Профиль
        profile_result = MagicMock()
        profile_result.scalar_one_or_none.return_value = self._make_profile()

        mock_db.execute = AsyncMock(side_effect=[execute_result, profile_result])

        result = await builder.build(
            query="Покажи тренировки",
            session_id="session-1",
            user_id="user-1",
            db=mock_db,
        )

        assert isinstance(result, EnrichedQuery)
        assert result.raw_text == "Покажи тренировки"
        assert result.normalized_text == "покажи тренировки"

    @pytest.mark.asyncio
    async def test_build_with_empty_history(
        self, builder: ContextBuilder, mock_db: AsyncMock
    ) -> None:
        execute_result = MagicMock()
        execute_result.scalars.return_value.all.return_value = []

        profile_result = MagicMock()
        profile_result.scalar_one_or_none.return_value = None

        mock_db.execute = AsyncMock(side_effect=[execute_result, profile_result])

        result = await builder.build(
            query="Привет",
            session_id="session-empty",
            user_id="user-unknown",
            db=mock_db,
        )

        assert result.conversation_history == []
        assert result.user_profile is None

    @pytest.mark.asyncio
    async def test_build_metadata(
        self, builder: ContextBuilder, mock_db: AsyncMock
    ) -> None:
        execute_result = MagicMock()
        execute_result.scalars.return_value.all.return_value = []

        profile_result = MagicMock()
        profile_result.scalar_one_or_none.return_value = None

        mock_db.execute = AsyncMock(side_effect=[execute_result, profile_result])

        result = await builder.build(
            query="Тест",
            session_id="sess-42",
            user_id="user-42",
            db=mock_db,
        )

        assert result.metadata["session_id"] == "sess-42"
        assert result.metadata["user_id"] == "user-42"
        assert "timestamp" in result.metadata

    @pytest.mark.asyncio
    async def test_semantic_and_knowledge_context_empty(
        self, builder: ContextBuilder, mock_db: AsyncMock
    ) -> None:
        """В MVP semantic_context и knowledge_context всегда пустые."""
        execute_result = MagicMock()
        execute_result.scalars.return_value.all.return_value = []

        profile_result = MagicMock()
        profile_result.scalar_one_or_none.return_value = None

        mock_db.execute = AsyncMock(side_effect=[execute_result, profile_result])

        result = await builder.build("Тест", "s1", "u1", mock_db)

        assert result.semantic_context == []
        assert result.knowledge_context == []

    @pytest.mark.asyncio
    async def test_history_order_chronological(
        self, builder: ContextBuilder, mock_db: AsyncMock
    ) -> None:
        """История сообщений должна быть в хронологическом порядке."""
        msg1 = self._make_message("user", "первое", 0)
        msg2 = self._make_message("assistant", "второе", 1)
        # execute возвращает в порядке desc, builder разворачивает
        execute_result = MagicMock()
        execute_result.scalars.return_value.all.return_value = [msg2, msg1]

        profile_result = MagicMock()
        profile_result.scalar_one_or_none.return_value = None

        mock_db.execute = AsyncMock(side_effect=[execute_result, profile_result])

        result = await builder.build("Что дальше?", "s1", "u1", mock_db)

        assert result.conversation_history[0]["content"] == "первое"
        assert result.conversation_history[1]["content"] == "второе"

    @pytest.mark.asyncio
    async def test_profile_fields_present(
        self, builder: ContextBuilder, mock_db: AsyncMock
    ) -> None:
        execute_result = MagicMock()
        execute_result.scalars.return_value.all.return_value = []

        profile_result = MagicMock()
        profile_result.scalar_one_or_none.return_value = self._make_profile()

        mock_db.execute = AsyncMock(side_effect=[execute_result, profile_result])

        result = await builder.build("Тест", "s1", "user-1", mock_db)

        profile = result.user_profile
        assert profile is not None
        assert profile["age"] == 30
        assert profile["weight_kg"] == 75.0
        assert profile["experience_level"] == "intermediate"
        assert "preferred_sports" in profile
