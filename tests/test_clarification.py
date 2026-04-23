"""Тесты для app/pipeline/clarification.py — Issue #57."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db import Base
from app.models.pending_clarification import PendingClarification
from app.pipeline.clarification import (
    PENDING_TTL,
    build_clarification_question,
    clear_pending,
    needs_clarification,
    resume_from_clarification,
    save_pending,
)
from app.pipeline.intent_detection import IntentResult
from app.pipeline.slot_state import SlotState, slot_state_from_entities
from app.tools.schemas import AnalysisType, SportTypeEnum, TimeRange


# ---------------------------------------------------------------------------
# needs_clarification
# ---------------------------------------------------------------------------


class TestNeedsClarification:
    def test_plan_request_without_time_range_needs_clarification(self) -> None:
        slots = slot_state_from_entities({}, raw_query="Составь план")
        missing = needs_clarification("plan_request", slots)
        assert missing == ["time_range"]

    def test_plan_request_with_time_range_does_not_need(self) -> None:
        slots = slot_state_from_entities(
            {"time_range": "за неделю"}, raw_query="Составь план на неделю"
        )
        missing = needs_clarification("plan_request", slots)
        assert missing == []

    def test_log_activity_without_sport_needs_clarification(self) -> None:
        slots = slot_state_from_entities({}, raw_query="Запиши тренировку")
        missing = needs_clarification("log_activity", slots)
        assert missing == ["sport_types"]

    def test_log_activity_with_sport_does_not_need(self) -> None:
        slots = slot_state_from_entities({"sport_type": "running"})
        missing = needs_clarification("log_activity", slots)
        assert missing == []

    def test_data_query_default_does_not_need(self) -> None:
        """Большинство data_query — дефолт «за неделю» достаточен."""
        slots = slot_state_from_entities({}, raw_query="Покажи тренировки")
        missing = needs_clarification("data_query", slots)
        assert missing == []

    def test_data_query_compare_without_time_range_needs_clarification(self) -> None:
        """COMPARE без time_range бессмыслен — сравнивать нечего."""
        slots = SlotState(analysis_type=AnalysisType.COMPARE)
        missing = needs_clarification("data_query", slots)
        assert missing == ["time_range"]

    def test_data_query_compare_with_time_range_ok(self) -> None:
        slots = slot_state_from_entities(
            {"time_range": "за неделю", "analysis_type": "compare"}
        )
        missing = needs_clarification("data_query", slots)
        assert missing == []

    def test_unknown_intent_does_not_need(self) -> None:
        slots = SlotState()
        assert needs_clarification("general_chat", slots) == []
        assert needs_clarification("reference_question", slots) == []
        assert needs_clarification("capability_question", slots) == []


# ---------------------------------------------------------------------------
# build_clarification_question
# ---------------------------------------------------------------------------


class TestBuildClarificationQuestion:
    def test_time_range_priority(self) -> None:
        """time_range всегда спрашивается первым."""
        q = build_clarification_question(["sport_types", "time_range"])
        assert "период" in q.lower()

    def test_sport_types_question(self) -> None:
        q = build_clarification_question(["sport_types"])
        assert "актив" in q.lower() or "спорт" in q.lower() or "бег" in q.lower()

    def test_single_slot_time_range(self) -> None:
        q = build_clarification_question(["time_range"])
        assert "период" in q.lower()
        # Один вопрос — не должно быть перечисления нескольких тем
        assert q.count("?") <= 2

    def test_empty_missing_returns_default(self) -> None:
        q = build_clarification_question([])
        assert q  # непустой

    def test_one_question_at_a_time(self) -> None:
        """Issue #57 risk: вопрос должен быть один за раз."""
        q = build_clarification_question(
            ["time_range", "sport_types", "metrics", "body_parts"]
        )
        # Не спрашиваем «ответь на 3 вопроса»
        assert "период" in q.lower()
        assert "спорт" not in q.lower() and "актив" not in q.lower()


# ---------------------------------------------------------------------------
# save_pending / resume_from_clarification (требуют БД)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db_session(tmp_path) -> AsyncSession:
    db_path = tmp_path / "clarif.db"
    url = f"sqlite+aiosqlite:///{db_path}"
    engine = create_async_engine(url, echo=False)
    session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session = session_factory()
    try:
        yield session
    finally:
        await session.close()
        await engine.dispose()


def _make_intent_result(
    intent: str,
    entities: dict | None = None,
    raw_query: str = "",
) -> IntentResult:
    entities = dict(entities or {})
    return IntentResult(
        intent=intent,
        confidence=0.9,
        entities=entities,
        raw_query=raw_query,
        llm_used=False,
        slots=slot_state_from_entities(entities, raw_query=raw_query),
    )


class TestSaveAndResume:
    @pytest.mark.asyncio
    async def test_save_then_resume_with_time_range(self, db_session: AsyncSession) -> None:
        """Пользователь написал «составь план», мы сохранили pending. В следующем
        сообщении «на неделю» заполнен time_range — resume должен вернуть
        полный IntentResult с intent=plan_request."""
        original = _make_intent_result("plan_request", {}, raw_query="Составь план")
        await save_pending(db_session, "sess-1", original, missing=["time_range"])

        resumed = await resume_from_clarification(
            db_session, "sess-1", "за неделю для бега"
        )
        assert resumed is not None
        assert resumed.intent == "plan_request"
        assert resumed.slots.time_range is not None
        assert resumed.slots.time_range.label == "за неделю"
        assert resumed.slots.sport_type == SportTypeEnum.RUNNING
        # raw_query должен содержать оба сообщения — keyword router ищет маркеры
        assert "Составь план" in resumed.raw_query
        assert "за неделю" in resumed.raw_query

    @pytest.mark.asyncio
    async def test_pending_cleared_after_resume(self, db_session: AsyncSession) -> None:
        original = _make_intent_result("plan_request", {}, raw_query="Составь план")
        await save_pending(db_session, "sess-2", original, missing=["time_range"])

        await resume_from_clarification(db_session, "sess-2", "за неделю")
        # Запись удалена после успешного resume
        res = await db_session.execute(
            select(PendingClarification).where(
                PendingClarification.session_id == "sess-2"
            )
        )
        assert res.scalar_one_or_none() is None

    @pytest.mark.asyncio
    async def test_resume_returns_none_when_no_pending(self, db_session: AsyncSession) -> None:
        resumed = await resume_from_clarification(db_session, "nonexistent", "за неделю")
        assert resumed is None

    @pytest.mark.asyncio
    async def test_expired_pending_returns_none(self, db_session: AsyncSession) -> None:
        """Pending старше TTL должен игнорироваться."""
        # Создаём запись с просроченным expires_at вручную
        past = datetime.utcnow() - PENDING_TTL - timedelta(seconds=60)
        db_session.add(PendingClarification(
            session_id="sess-expired",
            intent="plan_request",
            original_query="Составь план",
            filled_slots={},
            missing_slots=["time_range"],
            created_at=past,
            expires_at=past + timedelta(seconds=10),  # уже в прошлом
        ))
        await db_session.commit()

        resumed = await resume_from_clarification(
            db_session, "sess-expired", "за неделю"
        )
        assert resumed is None
        # И запись очищена
        res = await db_session.execute(
            select(PendingClarification).where(
                PendingClarification.session_id == "sess-expired"
            )
        )
        assert res.scalar_one_or_none() is None

    @pytest.mark.asyncio
    async def test_resume_returns_none_if_nothing_covered(
        self, db_session: AsyncSession,
    ) -> None:
        """Если новое сообщение не покрывает ни одного недостающего слота —
        return None, вызывающий код обработает его как новый запрос."""
        original = _make_intent_result("plan_request", {}, raw_query="Составь план")
        await save_pending(db_session, "sess-3", original, missing=["time_range"])

        resumed = await resume_from_clarification(
            db_session, "sess-3", "спасибо"  # ни периода, ни спорта
        )
        assert resumed is None
        # Pending не удалён — пользователь может ещё ответить
        res = await db_session.execute(
            select(PendingClarification).where(
                PendingClarification.session_id == "sess-3"
            )
        )
        assert res.scalar_one_or_none() is not None

    @pytest.mark.asyncio
    async def test_save_pending_replaces_previous(self, db_session: AsyncSession) -> None:
        """Повторный save_pending для той же сессии — replace, а не дубль."""
        first = _make_intent_result("plan_request", {}, raw_query="первый")
        await save_pending(db_session, "sess-4", first, missing=["time_range"])

        second = _make_intent_result("log_activity", {}, raw_query="второй")
        await save_pending(db_session, "sess-4", second, missing=["sport_types"])

        res = await db_session.execute(
            select(PendingClarification).where(
                PendingClarification.session_id == "sess-4"
            )
        )
        rows = res.scalars().all()
        assert len(rows) == 1
        assert rows[0].intent == "log_activity"
        assert rows[0].original_query == "второй"

    @pytest.mark.asyncio
    async def test_clear_pending_idempotent(self, db_session: AsyncSession) -> None:
        await clear_pending(db_session, "never-existed")  # не должен падать
        original = _make_intent_result("plan_request", {}, raw_query="план")
        await save_pending(db_session, "sess-5", original, missing=["time_range"])
        await clear_pending(db_session, "sess-5")
        await clear_pending(db_session, "sess-5")  # повторный — idempotent
        res = await db_session.execute(
            select(PendingClarification).where(
                PendingClarification.session_id == "sess-5"
            )
        )
        assert res.scalar_one_or_none() is None
