"""Unit-тесты Admin Logs v2 + Profile Editor fix (Issue #34).

Проверяет:
- GET /api/admin/logs/{id}/llm-calls    — список LLM-вызовов
- GET /api/admin/logs/{id}/stage-trace  — stage trace
- POST /api/admin/profiles/{id}         — полное обновление с JSON-валидацией
- POST /api/admin/profiles/{id} с невалидным JSON → 400
"""

from __future__ import annotations

import base64
import os
import sys
import uuid
from datetime import datetime
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def async_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Асинхронная in-memory SQLite БД с полной схемой."""
    from app.db import Base
    import app.models  # noqa: F401

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session

    await engine.dispose()


@pytest_asyncio.fixture
async def client(async_db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """FastAPI test client с переопределённой зависимостью БД."""
    from app.main import app
    from app.db import get_db

    async def override_get_db():
        yield async_db_session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c

    app.dependency_overrides.clear()


def _auth_header() -> dict[str, str]:
    token = base64.b64encode(b"admin:admin").decode()
    return {"Authorization": f"Basic {token}"}


# ---------------------------------------------------------------------------
# Helpers — создание тестовых данных
# ---------------------------------------------------------------------------

async def _create_pipeline_log(
    session: AsyncSession,
    request_id: str | None = None,
    stage_trace: list | None = None,
    rag_chunks_used: list | None = None,
) -> str:
    from app.models.pipeline_log import PipelineLog

    rid = request_id or str(uuid.uuid4())
    log = PipelineLog(
        id=rid,
        user_id="u-test",
        session_id="s-test",
        raw_query="тестовый запрос",
        intent="data_retrieval",
        route="tool_simple",
        total_duration_ms=500,
        stage_trace=stage_trace or [],
        rag_chunks_used=rag_chunks_used or [],
    )
    session.add(log)
    await session.commit()
    return rid


async def _create_llm_call(session: AsyncSession, request_id: str, role: str = "response") -> None:
    from app.models.llm_call import LLMCall

    call = LLMCall(
        id=str(uuid.uuid4()),
        request_id=request_id,
        role=role,
        model="qwen2.5:7b",
        prompt="Тестовый промпт для LLM",
        response="Тестовый ответ LLM",
        prompt_length=30,
        response_length=20,
        duration_ms=120,
        iteration=None,
        timestamp=datetime.utcnow(),
    )
    session.add(call)
    await session.commit()


async def _create_profile(session: AsyncSession) -> str:
    from app.models.user_profile import UserProfile

    pid = str(uuid.uuid4())
    profile = UserProfile(
        id=pid,
        user_id=str(uuid.uuid4()),
        name="Тест Юзер",
        age=30,
        weight_kg=75.0,
        height_cm=180.0,
        gender="male",
        experience_level="intermediate",
        max_heart_rate=185,
        resting_heart_rate=55,
        training_goals=["сила"],
        injuries=[],
        chronic_conditions=[],
        preferred_sports=["running"],
    )
    session.add(profile)
    await session.commit()
    return pid


# ---------------------------------------------------------------------------
# Tests: GET /api/admin/logs/{id}/llm-calls
# ---------------------------------------------------------------------------

class TestLogLLMCalls:
    @pytest.mark.asyncio
    async def test_returns_empty_when_no_calls(self, client: AsyncClient, async_db_session: AsyncSession) -> None:
        rid = await _create_pipeline_log(async_db_session)
        resp = await client.get(f"/api/admin/logs/{rid}/llm-calls", headers=_auth_header())
        assert resp.status_code == 200
        data = resp.json()
        assert data["request_id"] == rid
        assert data["total"] == 0
        assert data["items"] == []

    @pytest.mark.asyncio
    async def test_returns_llm_calls(self, client: AsyncClient, async_db_session: AsyncSession) -> None:
        rid = await _create_pipeline_log(async_db_session)
        await _create_llm_call(async_db_session, rid, role="response")
        await _create_llm_call(async_db_session, rid, role="intent_llm")

        resp = await client.get(f"/api/admin/logs/{rid}/llm-calls", headers=_auth_header())
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        roles = {c["role"] for c in data["items"]}
        assert "response" in roles
        assert "intent_llm" in roles

    @pytest.mark.asyncio
    async def test_call_has_preview_fields(self, client: AsyncClient, async_db_session: AsyncSession) -> None:
        rid = await _create_pipeline_log(async_db_session)
        await _create_llm_call(async_db_session, rid)

        resp = await client.get(f"/api/admin/logs/{rid}/llm-calls", headers=_auth_header())
        item = resp.json()["items"][0]
        assert "prompt_preview" in item
        assert "response_preview" in item
        assert "prompt" in item
        assert "response" in item
        assert "duration_ms" in item

    @pytest.mark.asyncio
    async def test_requires_auth(self, client: AsyncClient, async_db_session: AsyncSession) -> None:
        rid = await _create_pipeline_log(async_db_session)
        resp = await client.get(f"/api/admin/logs/{rid}/llm-calls")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Tests: GET /api/admin/logs/{id}/stage-trace
# ---------------------------------------------------------------------------

class TestLogStageTrace:
    @pytest.mark.asyncio
    async def test_empty_trace(self, client: AsyncClient, async_db_session: AsyncSession) -> None:
        rid = await _create_pipeline_log(async_db_session)
        resp = await client.get(f"/api/admin/logs/{rid}/stage-trace", headers=_auth_header())
        assert resp.status_code == 200
        data = resp.json()
        assert data["request_id"] == rid
        assert data["stages"] == []

    @pytest.mark.asyncio
    async def test_trace_with_waterfall_data(self, client: AsyncClient, async_db_session: AsyncSession) -> None:
        trace = [
            {"stage": "context_build", "start_ms": 0, "duration_ms": 100},
            {"stage": "intent_detection", "start_ms": 100, "duration_ms": 50},
            {"stage": "response_gen", "start_ms": 150, "duration_ms": 300},
        ]
        rid = await _create_pipeline_log(async_db_session, stage_trace=trace)
        resp = await client.get(f"/api/admin/logs/{rid}/stage-trace", headers=_auth_header())
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["stages"]) == 3
        # Проверяем, что waterfall поля добавлены
        first = data["stages"][0]
        assert "width_pct" in first
        assert "offset_pct" in first
        assert first["offset_pct"] == 0.0

    @pytest.mark.asyncio
    async def test_not_found(self, client: AsyncClient) -> None:
        resp = await client.get("/api/admin/logs/nonexistent-id/stage-trace", headers=_auth_header())
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Tests: POST /api/admin/profiles/{id}
# ---------------------------------------------------------------------------

class TestProfileUpdateFull:
    @pytest.mark.asyncio
    async def test_update_scalar_fields(self, client: AsyncClient, async_db_session: AsyncSession) -> None:
        pid = await _create_profile(async_db_session)
        payload = {
            "name": "Новое Имя",
            "age": 25,
            "weight_kg": 70.5,
            "height_cm": 175.0,
            "gender": "female",
            "experience_level": "advanced",
            "max_heart_rate": 195,
            "resting_heart_rate": 50,
        }
        resp = await client.post(
            f"/api/admin/profiles/{pid}",
            json=payload,
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Новое Имя"
        assert data["age"] == 25
        assert data["gender"] == "female"
        assert data["experience_level"] == "advanced"

    @pytest.mark.asyncio
    async def test_update_json_fields_as_strings(self, client: AsyncClient, async_db_session: AsyncSession) -> None:
        """JSON-поля переданные как строки парсятся и сохраняются."""
        pid = await _create_profile(async_db_session)
        payload = {
            "injuries": '["колено", "плечо"]',
            "chronic_conditions": '["гипертония"]',
            "preferred_sports": '["swimming", "cycling"]',
            "training_goals": '["выносливость"]',
        }
        resp = await client.post(
            f"/api/admin/profiles/{pid}",
            json=payload,
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["injuries"] == ["колено", "плечо"]
        assert data["chronic_conditions"] == ["гипертония"]
        assert data["preferred_sports"] == ["swimming", "cycling"]

    @pytest.mark.asyncio
    async def test_update_json_fields_as_lists(self, client: AsyncClient, async_db_session: AsyncSession) -> None:
        """JSON-поля переданные как list сохраняются без изменений."""
        pid = await _create_profile(async_db_session)
        payload = {"injuries": ["колено"], "preferred_sports": ["yoga"]}
        resp = await client.post(
            f"/api/admin/profiles/{pid}",
            json=payload,
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        assert resp.json()["injuries"] == ["колено"]

    @pytest.mark.asyncio
    async def test_invalid_json_injuries_returns_400(self, client: AsyncClient, async_db_session: AsyncSession) -> None:
        """Невалидный JSON в поле injuries → HTTP 400."""
        pid = await _create_profile(async_db_session)
        payload = {"injuries": "не json!"}
        resp = await client.post(
            f"/api/admin/profiles/{pid}",
            json=payload,
            headers=_auth_header(),
        )
        assert resp.status_code == 400
        assert "injuries" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_invalid_json_preferred_sports_returns_400(self, client: AsyncClient, async_db_session: AsyncSession) -> None:
        """Невалидный JSON в preferred_sports → HTTP 400."""
        pid = await _create_profile(async_db_session)
        payload = {"preferred_sports": "{not a list}"}
        resp = await client.post(
            f"/api/admin/profiles/{pid}",
            json=payload,
            headers=_auth_header(),
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_json_object_not_list_returns_400(self, client: AsyncClient, async_db_session: AsyncSession) -> None:
        """JSON-объект вместо массива → HTTP 400."""
        pid = await _create_profile(async_db_session)
        payload = {"injuries": '{"key": "value"}'}
        resp = await client.post(
            f"/api/admin/profiles/{pid}",
            json=payload,
            headers=_auth_header(),
        )
        assert resp.status_code == 400
        assert "массив" in resp.json()["detail"].lower() or "list" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_empty_string_json_field_saves_empty_list(self, client: AsyncClient, async_db_session: AsyncSession) -> None:
        """Пустая строка в JSON-поле → сохраняется как пустой список."""
        pid = await _create_profile(async_db_session)
        payload = {"injuries": ""}
        resp = await client.post(
            f"/api/admin/profiles/{pid}",
            json=payload,
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        assert resp.json()["injuries"] == []

    @pytest.mark.asyncio
    async def test_not_found_returns_404(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/admin/profiles/nonexistent",
            json={"name": "test"},
            headers=_auth_header(),
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_requires_auth(self, client: AsyncClient, async_db_session: AsyncSession) -> None:
        pid = await _create_profile(async_db_session)
        resp = await client.post(f"/api/admin/profiles/{pid}", json={"name": "x"})
        assert resp.status_code == 401
