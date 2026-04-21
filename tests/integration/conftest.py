"""Фикстуры для интеграционных тестов Orchestrator v2 (Issue #37).

Предоставляет:
- mock_ollama: подменяет OllamaClient.generate/generate_stream/chat/chat_stream/list_models,
  возвращает заранее заготовленные ответы per-role (поддерживает streaming).
- mock_chroma: отключает ChromaDB (vector_store.available=False), так что
  semantic_memory.recall / rag_retrieve возвращают пустые списки без сетевых вызовов.
- test_db: временная SQLite (tmp_path) с созданными таблицами и seed-данными.
  Перенаправляет app.db.AsyncSessionLocal на тестовую базу — чтобы фоновые
  memory_updater-таски работали с той же БД.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app import db as app_db
from app.db import Base
from app.models.activity import Activity
from app.models.daily_fact import DailyFact
from app.models.user_profile import UserProfile
from app.services.llm_call_logger import llm_call_logger
from app.services.llm_registry import llm_registry
from app.services.llm_service import LLMResponse, OllamaClient
from app.services.vector_store import vector_store


# ---------------------------------------------------------------------------
# Mock Ollama
# ---------------------------------------------------------------------------


@dataclass
class MockOllama:
    """Менеджер заготовленных ответов LLM per-role.

    Usage:
        mock_ollama.set("response", "Привет!")
        mock_ollama.set_sequence("planner", [
            '{"thought":"нужны данные","tool_calls":[{"tool":"get_activities","args":{}}]}',
            '{"thought":"готов","final_answer":true}',
        ])
    """

    responses: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    prompts: list[dict] = field(default_factory=list)

    def set(self, role: str, response: str) -> None:
        """Задать один фиксированный (sticky) ответ для роли."""
        self.responses[role] = [response]

    def set_sequence(self, role: str, responses: list[str]) -> None:
        """Задать последовательность ответов — каждый вызов потребляет очередной."""
        self.responses[role] = list(responses)

    def next_response(self, role: str) -> str:
        """Получить следующий ответ для роли; последний — «sticky»."""
        self.calls[role] += 1
        items = self.responses.get(role, [])
        if not items:
            return "mock-response"
        if len(items) == 1:
            return items[0]
        return items.pop(0)

    def record_prompt(self, role: str, prompt: str, system_prompt: str | None) -> None:
        self.prompts.append({
            "role": role,
            "prompt": prompt,
            "system_prompt": system_prompt,
        })

    def reset(self) -> None:
        self.responses.clear()
        self.calls.clear()
        self.prompts.clear()


@pytest.fixture
def mock_ollama(monkeypatch: pytest.MonkeyPatch) -> Iterator[MockOllama]:
    """Подменяет OllamaClient.* методы и сбрасывает кеш LLMRegistry."""
    mock = MockOllama()

    def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
        """Склеить messages в единую строку для логирования и трассировки промптов."""
        return "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in messages)

    async def fake_generate(
        self: OllamaClient,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        format: str | None = None,
    ) -> LLMResponse:
        role = self._role
        content = mock.next_response(role)
        mock.record_prompt(role, prompt, system_prompt)
        model = self._model or "mock-model"
        llm_call_logger.record(
            role=role,
            model=model,
            prompt=prompt[:4096] if prompt else None,
            response=content[:4096] if content else None,
            prompt_length=len(prompt or ""),
            response_length=len(content or ""),
            duration_ms=1,
        )
        return LLMResponse(
            content=content,
            model=model,
            prompt_length=len(prompt or ""),
            response_length=len(content or ""),
            duration_ms=1.0,
        )

    async def fake_generate_stream(
        self: OllamaClient,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        role = self._role
        content = mock.next_response(role)
        mock.record_prompt(role, prompt, system_prompt)
        model = self._model or "mock-model"
        if on_token is not None:
            for token in content:
                on_token(token)
        llm_call_logger.record(
            role=role,
            model=model,
            prompt=prompt[:4096] if prompt else None,
            response=content[:4096] if content else None,
            prompt_length=len(prompt or ""),
            response_length=len(content or ""),
            duration_ms=1,
        )
        return LLMResponse(
            content=content,
            model=model,
            prompt_length=len(prompt or ""),
            response_length=len(content or ""),
            duration_ms=1.0,
        )

    async def fake_chat(
        self: OllamaClient,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        system_prompts: list[str] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        format: str | None = None,
    ) -> LLMResponse:
        role = self._role
        content = mock.next_response(role)
        prompt_str = _messages_to_prompt(messages)
        sp_combined = (
            "\n".join(system_prompts) if system_prompts else (system_prompt or None)
        )
        mock.record_prompt(role, prompt_str, sp_combined)
        model = self._model or "mock-model"
        llm_call_logger.record(
            role=role,
            model=model,
            prompt=prompt_str[:4096] if prompt_str else None,
            response=content[:4096] if content else None,
            prompt_length=len(prompt_str),
            response_length=len(content or ""),
            duration_ms=1,
        )
        return LLMResponse(
            content=content,
            model=model,
            prompt_length=len(prompt_str),
            response_length=len(content or ""),
            duration_ms=1.0,
        )

    async def fake_chat_stream(
        self: OllamaClient,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        system_prompts: list[str] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        role = self._role
        content = mock.next_response(role)
        prompt_str = _messages_to_prompt(messages)
        sp_combined = (
            "\n".join(system_prompts) if system_prompts else (system_prompt or None)
        )
        mock.record_prompt(role, prompt_str, sp_combined)
        model = self._model or "mock-model"
        if on_token is not None:
            for token in content:
                on_token(token)
        llm_call_logger.record(
            role=role,
            model=model,
            prompt=prompt_str[:4096] if prompt_str else None,
            response=content[:4096] if content else None,
            prompt_length=len(prompt_str),
            response_length=len(content or ""),
            duration_ms=1,
        )
        return LLMResponse(
            content=content,
            model=model,
            prompt_length=len(prompt_str),
            response_length=len(content or ""),
            duration_ms=1.0,
        )

    async def fake_list_models(self: OllamaClient) -> list[str]:
        return ["mock-model"]

    async def fake_health_check(self: OllamaClient) -> dict:
        return {"available": True, "model": "mock-model", "model_loaded": True, "error": None}

    monkeypatch.setattr(OllamaClient, "generate", fake_generate)
    monkeypatch.setattr(OllamaClient, "generate_stream", fake_generate_stream)
    monkeypatch.setattr(OllamaClient, "chat", fake_chat)
    monkeypatch.setattr(OllamaClient, "chat_stream", fake_chat_stream)
    monkeypatch.setattr(OllamaClient, "list_models", fake_list_models)
    monkeypatch.setattr(OllamaClient, "health_check", fake_health_check)

    # Сбрасываем кеш клиентов LLMRegistry — чтобы новые клиенты создавались с mock-model
    llm_registry._clients.clear()
    llm_registry._overrides.clear()

    yield mock

    llm_registry._clients.clear()
    llm_registry._overrides.clear()


# ---------------------------------------------------------------------------
# Mock Chroma (graceful-degradation режим)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_chroma(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Отключает ChromaDB — semantic_memory и rag_retrieve возвращают пустые списки.

    Этого достаточно для интеграционных тестов: основной pipeline работает
    без Knowledge Base / Semantic Memory (все компоненты graceful-degrade).
    """
    original = vector_store._available
    vector_store._available = False
    yield
    vector_store._available = original


# ---------------------------------------------------------------------------
# Test DB (tmp SQLite + seed)
# ---------------------------------------------------------------------------


def _seed_profile(user_id: str) -> UserProfile:
    return UserProfile(
        user_id=user_id,
        name="Тест Тестович",
        age=30,
        weight_kg=75.0,
        height_cm=180.0,
        gender="male",
        max_heart_rate=190,
        resting_heart_rate=60,
        training_goals=["выносливость"],
        experience_level="intermediate",
        injuries=[],
        chronic_conditions=[],
        preferred_sports=["running"],
    )


def _seed_activities(user_id: str, count: int = 7) -> list[Activity]:
    """Создаёт несколько тренировок бегом в последние `count` дней."""
    today = datetime.utcnow().replace(hour=9, minute=0, second=0, microsecond=0)
    activities: list[Activity] = []
    for i in range(count):
        start = today - timedelta(days=i)
        end = start + timedelta(minutes=45)
        activities.append(Activity(
            user_id=user_id,
            title=f"Пробежка #{i}",
            sport_type="running",
            duration_seconds=45 * 60,
            distance_meters=7500.0,
            start_time=start,
            end_time=end,
            calories=420,
            avg_heart_rate=145,
            max_heart_rate=170,
            source="manual",
            is_primary=True,
            anomaly_flags=[],
            raw_title=f"Пробежка #{i}",
        ))
    return activities


def _seed_daily_facts(user_id: str, count: int = 7) -> list[DailyFact]:
    """Создаёт дневные метрики за последние `count` дней."""
    today = datetime.utcnow().date()
    facts: list[DailyFact] = []
    for i in range(count):
        iso_date = (today - timedelta(days=i)).isoformat()
        facts.append(DailyFact(
            user_id=user_id,
            iso_date=iso_date,
            steps=9000 + i * 200,
            calories_kcal=2200 + i * 50,
            recovery_score=65 + (i % 10),
            hrv_rmssd_milli=50.0 + i,
            resting_heart_rate=58 + (i % 4),
            spo2_percentage=97.0,
            skin_temp_celsius=36.6,
            sleep_total_in_bed_milli=7 * 3600 * 1000,
            water_liters=2.0,
            sources_json={},
            recovery_score_calculated=66 + (i % 10),
            strain_score=12.5,
            anomaly_flags=[],
        ))
    return facts


@pytest_asyncio.fixture
async def test_db(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> AsyncSession:
    """Временная SQLite-база с созданными таблицами и seed-данными.

    Перенаправляет `app.db.AsyncSessionLocal` на тестовую базу, чтобы фоновые
    задачи (memory_updater) тоже использовали тестовую БД.
    """
    db_path = tmp_path / "test.db"
    url = f"sqlite+aiosqlite:///{db_path}"

    engine = create_async_engine(url, echo=False)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Seed базовых данных под user-test
    async with session_factory() as seed_db:
        seed_db.add(_seed_profile("user-test"))
        for a in _seed_activities("user-test"):
            seed_db.add(a)
        for f in _seed_daily_facts("user-test"):
            seed_db.add(f)
        await seed_db.commit()

    # Подменяем глобальный AsyncSessionLocal — иначе фоновые таски пишут в prod-БД
    monkeypatch.setattr(app_db, "AsyncSessionLocal", session_factory)

    db = session_factory()
    try:
        yield db
    finally:
        await db.close()
        await engine.dispose()


# ---------------------------------------------------------------------------
# Вспомогательные утилиты
# ---------------------------------------------------------------------------


async def drain_background_tasks() -> None:
    """Даёт время фоновой memory_updater-таске завершиться.

    Несколько итераций asyncio.sleep(0) + небольшой sleep — этого достаточно
    для in-memory операций.
    """
    for _ in range(3):
        await asyncio.sleep(0)
    await asyncio.sleep(0.05)
