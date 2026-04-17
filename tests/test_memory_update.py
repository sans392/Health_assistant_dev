"""Тесты MemoryUpdater — rule-based fact extraction и semantic memory."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.pipeline.memory_update import MemoryUpdater


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile(
    injuries: list | None = None,
    training_goals: list | None = None,
    preferred_sports: list | None = None,
) -> MagicMock:
    p = MagicMock()
    p.injuries = injuries or []
    p.training_goals = training_goals or []
    p.preferred_sports = preferred_sports or []
    p.updated_at = None
    return p


# ---------------------------------------------------------------------------
# Long-term: injuries
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_injury_added_when_body_part_in_entities() -> None:
    """body_part из entities добавляется в profile.injuries."""
    updater = MemoryUpdater()
    profile = _make_profile()

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = profile
    mock_db.execute = AsyncMock(return_value=mock_result)
    mock_db.commit = AsyncMock()

    mock_session_ctx = MagicMock()
    mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_db)
    mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("app.pipeline.memory_update.AsyncSessionLocal", return_value=mock_session_ctx):
        await updater._update_long_term(
            user_id="u1",
            entities={"body_part": "колено"},
        )

    assert any(i.get("body_part") == "колено" for i in profile.injuries)
    mock_db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_injury_not_duplicated() -> None:
    """Существующая травма не добавляется повторно."""
    updater = MemoryUpdater()
    profile = _make_profile(injuries=[{"body_part": "колено", "status": "active"}])

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = profile
    mock_db.execute = AsyncMock(return_value=mock_result)
    mock_db.commit = AsyncMock()

    mock_session_ctx = MagicMock()
    mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_db)
    mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("app.pipeline.memory_update.AsyncSessionLocal", return_value=mock_session_ctx):
        await updater._update_long_term(
            user_id="u1",
            entities={"body_part": "колено"},
        )

    # commit не должен вызываться — ничего не изменилось
    mock_db.commit.assert_not_awaited()


# ---------------------------------------------------------------------------
# Long-term: goals
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_goal_added_when_goal_in_entities() -> None:
    """goal из entities добавляется в profile.training_goals."""
    updater = MemoryUpdater()
    profile = _make_profile()

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = profile
    mock_db.execute = AsyncMock(return_value=mock_result)
    mock_db.commit = AsyncMock()

    mock_session_ctx = MagicMock()
    mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_db)
    mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("app.pipeline.memory_update.AsyncSessionLocal", return_value=mock_session_ctx):
        await updater._update_long_term(
            user_id="u1",
            entities={"goal": "похудеть"},
        )

    assert "похудеть" in profile.training_goals
    mock_db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_goal_not_duplicated() -> None:
    """Существующая цель не добавляется повторно."""
    updater = MemoryUpdater()
    profile = _make_profile(training_goals=["похудеть"])

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = profile
    mock_db.execute = AsyncMock(return_value=mock_result)
    mock_db.commit = AsyncMock()

    mock_session_ctx = MagicMock()
    mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_db)
    mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("app.pipeline.memory_update.AsyncSessionLocal", return_value=mock_session_ctx):
        await updater._update_long_term(
            user_id="u1",
            entities={"goal": "похудеть"},
        )

    mock_db.commit.assert_not_awaited()


# ---------------------------------------------------------------------------
# Long-term: sports
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sport_added_when_sport_type_in_entities() -> None:
    """sport_type из entities добавляется в profile.preferred_sports."""
    updater = MemoryUpdater()
    profile = _make_profile()

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = profile
    mock_db.execute = AsyncMock(return_value=mock_result)
    mock_db.commit = AsyncMock()

    mock_session_ctx = MagicMock()
    mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_db)
    mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("app.pipeline.memory_update.AsyncSessionLocal", return_value=mock_session_ctx):
        await updater._update_long_term(
            user_id="u1",
            entities={"sport_type": "running"},
        )

    assert "running" in profile.preferred_sports
    mock_db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_sport_not_duplicated() -> None:
    """Существующий вид спорта не добавляется повторно."""
    updater = MemoryUpdater()
    profile = _make_profile(preferred_sports=["running"])

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = profile
    mock_db.execute = AsyncMock(return_value=mock_result)
    mock_db.commit = AsyncMock()

    mock_session_ctx = MagicMock()
    mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_db)
    mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("app.pipeline.memory_update.AsyncSessionLocal", return_value=mock_session_ctx):
        await updater._update_long_term(
            user_id="u1",
            entities={"sport_type": "running"},
        )

    mock_db.commit.assert_not_awaited()


# ---------------------------------------------------------------------------
# Long-term: no entities — skip DB
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_long_term_skipped_when_no_relevant_entities() -> None:
    """Если нет body_part/goal/sport_type — DB не вызывается."""
    updater = MemoryUpdater()

    with patch("app.pipeline.memory_update.AsyncSessionLocal") as mock_local:
        await updater._update_long_term(
            user_id="u1",
            entities={"time_range": "за неделю", "metric": "калории"},
        )
        mock_local.assert_not_called()


# ---------------------------------------------------------------------------
# Long-term: missing profile — no error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_long_term_no_error_when_profile_missing() -> None:
    """Если профиль не найден — метод завершается без ошибки."""
    updater = MemoryUpdater()

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db.execute = AsyncMock(return_value=mock_result)
    mock_db.commit = AsyncMock()

    mock_session_ctx = MagicMock()
    mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_db)
    mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("app.pipeline.memory_update.AsyncSessionLocal", return_value=mock_session_ctx):
        await updater._update_long_term(
            user_id="u_missing",
            entities={"body_part": "спина"},
        )

    mock_db.commit.assert_not_awaited()


# ---------------------------------------------------------------------------
# Semantic memory called
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_semantic_remember_called() -> None:
    """_update_semantic вызывает semantic_memory.remember."""
    updater = MemoryUpdater()

    with patch("app.pipeline.memory_update.semantic_memory") as mock_sm:
        mock_sm.remember = AsyncMock()
        await updater._update_semantic(
            user_id="u1",
            request_id="req-123",
            query="Как мой прогресс?",
            response="За неделю вы тренировались 4 раза.",
        )

    mock_sm.remember.assert_awaited_once_with(
        user_id="u1",
        request_id="req-123",
        query="Как мой прогресс?",
        response="За неделю вы тренировались 4 раза.",
    )


@pytest.mark.asyncio
async def test_semantic_error_does_not_propagate() -> None:
    """Ошибка в semantic_memory.remember не пробрасывается наружу."""
    updater = MemoryUpdater()

    with patch("app.pipeline.memory_update.semantic_memory") as mock_sm:
        mock_sm.remember = AsyncMock(side_effect=RuntimeError("chroma down"))
        # Не должно бросить исключение
        await updater._update_semantic(
            user_id="u1",
            request_id=None,
            query="q",
            response="r",
        )


# ---------------------------------------------------------------------------
# update() end-to-end: errors don't break caller
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_does_not_raise_on_db_error() -> None:
    """Ошибка в update() поглощается, не влияет на caller."""
    updater = MemoryUpdater()

    mock_session_ctx = MagicMock()
    mock_session_ctx.__aenter__ = AsyncMock(side_effect=RuntimeError("db error"))
    mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("app.pipeline.memory_update.AsyncSessionLocal", return_value=mock_session_ctx):
        with patch("app.pipeline.memory_update.semantic_memory") as mock_sm:
            mock_sm.remember = AsyncMock()
            # Не должно бросить исключение
            await updater.update(
                user_id="u1",
                session_id="s1",
                request_id="req-1",
                query="q",
                response="r",
                intent="health_concern",
                entities={"body_part": "колено"},
            )
