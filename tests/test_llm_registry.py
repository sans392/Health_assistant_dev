"""Unit-тесты для LLM Registry (Issue #21).

Покрывают: выбор модели по роли, override, fallback на базовую модель,
получение клиента, персистентность в БД (mock).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.llm_registry import LLMRegistry, ALL_ROLES, ROLE_TIMEOUTS
from app.services.llm_service import OllamaClient


# ------------------------------------------------------------------ #
# Вспомогательные fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def registry() -> LLMRegistry:
    """Свежий экземпляр LLMRegistry (не затрагивает глобальный синглтон)."""
    return LLMRegistry()


# ------------------------------------------------------------------ #
# get_model
# ------------------------------------------------------------------ #

class TestGetModel:
    def test_returns_base_model_when_no_env_or_override(self, registry: LLMRegistry):
        """Без env-переменных и overrides — возвращает базовую модель."""
        with patch("app.services.llm_registry.settings") as mock_settings:
            mock_settings.llm_response_model = ""
            mock_settings.ollama_model = "qwen2.5:7b"
            assert registry.get_model("response") == "qwen2.5:7b"

    def test_returns_env_model_when_set(self, registry: LLMRegistry):
        """Если env-переменная задана — возвращает её."""
        with patch("app.services.llm_registry.settings") as mock_settings:
            mock_settings.llm_response_model = "qwen2.5:14b"
            mock_settings.llm_intent_model = "qwen2.5:7b"
            mock_settings.llm_safety_model = ""
            mock_settings.llm_planner_model = ""
            mock_settings.ollama_model = "qwen2.5:7b"
            assert registry.get_model("response") == "qwen2.5:14b"

    def test_override_has_priority_over_env(self, registry: LLMRegistry):
        """Runtime override имеет приоритет над env-переменной."""
        with patch("app.services.llm_registry.settings") as mock_settings:
            mock_settings.llm_response_model = "qwen2.5:14b"
            mock_settings.ollama_model = "qwen2.5:7b"
            registry._overrides["response"] = "llama3.1:8b"
            assert registry.get_model("response") == "llama3.1:8b"


# ------------------------------------------------------------------ #
# set_model
# ------------------------------------------------------------------ #

class TestSetModel:
    def test_set_model_updates_override(self, registry: LLMRegistry):
        registry.set_model("response", "qwen2.5:14b")
        assert registry._overrides["response"] == "qwen2.5:14b"

    def test_set_model_clears_client_cache(self, registry: LLMRegistry):
        """После set_model кэш клиента для данной роли сбрасывается."""
        # Создать клиент
        _ = registry.get_client("response")
        assert "response" in registry._clients

        registry.set_model("response", "qwen2.5:14b")
        assert "response" not in registry._clients

    def test_set_model_unknown_role_raises(self, registry: LLMRegistry):
        with pytest.raises(ValueError, match="Неизвестная роль"):
            registry.set_model("unknown_role", "some-model")

    def test_all_roles_can_be_set(self, registry: LLMRegistry):
        for role in ALL_ROLES:
            registry.set_model(role, "test-model")
            assert registry.get_model(role) == "test-model"


# ------------------------------------------------------------------ #
# get_client
# ------------------------------------------------------------------ #

class TestGetClient:
    def test_returns_ollama_client(self, registry: LLMRegistry):
        with patch("app.services.llm_registry.settings") as mock_settings:
            mock_settings.llm_response_model = ""
            mock_settings.ollama_model = "qwen2.5:7b"
            mock_settings.ollama_host = "http://localhost:11434"
            mock_settings.ollama_timeout = 60
            client = registry.get_client("response")
        assert isinstance(client, OllamaClient)

    def test_client_cached(self, registry: LLMRegistry):
        """Один и тот же объект клиента при повторном вызове."""
        with patch("app.services.llm_registry.settings") as mock_settings:
            mock_settings.llm_response_model = ""
            mock_settings.ollama_model = "qwen2.5:7b"
            mock_settings.ollama_host = "http://localhost:11434"
            mock_settings.ollama_timeout = 60
            c1 = registry.get_client("response")
            c2 = registry.get_client("response")
        assert c1 is c2

    def test_planner_uses_longer_timeout(self, registry: LLMRegistry):
        """Роль planner получает таймаут 120s."""
        with patch("app.services.llm_registry.settings") as mock_settings:
            mock_settings.llm_planner_model = ""
            mock_settings.ollama_model = "qwen2.5:7b"
            mock_settings.ollama_host = "http://localhost:11434"
            mock_settings.ollama_timeout = 60
            client = registry.get_client("planner")
        assert client._timeout == ROLE_TIMEOUTS["planner"]
        assert ROLE_TIMEOUTS["planner"] == 120

    def test_all_roles_return_client(self, registry: LLMRegistry):
        with patch("app.services.llm_registry.settings") as mock_settings:
            mock_settings.llm_intent_model = ""
            mock_settings.llm_safety_model = ""
            mock_settings.llm_response_model = ""
            mock_settings.llm_planner_model = ""
            mock_settings.ollama_model = "qwen2.5:7b"
            mock_settings.ollama_host = "http://localhost:11434"
            mock_settings.ollama_timeout = 60
            for role in ALL_ROLES:
                client = registry.get_client(role)
                assert isinstance(client, OllamaClient)


# ------------------------------------------------------------------ #
# initialize / fallback
# ------------------------------------------------------------------ #

class TestFallback:
    @pytest.mark.asyncio
    async def test_fallback_when_model_unavailable(self, registry: LLMRegistry):
        """Если модель роли недоступна — применяется fallback на базовую."""
        with patch("app.services.llm_registry.settings") as mock_settings:
            mock_settings.llm_response_model = "qwen2.5:14b"
            mock_settings.llm_intent_model = ""
            mock_settings.llm_safety_model = ""
            mock_settings.llm_planner_model = ""
            mock_settings.ollama_model = "qwen2.5:7b"
            mock_settings.ollama_host = "http://localhost:11434"
            mock_settings.ollama_timeout = 60

            # Ollama возвращает только базовую модель
            mock_client = AsyncMock()
            mock_client.list_models = AsyncMock(return_value=["qwen2.5:7b"])

            with patch(
                "app.services.llm_registry.OllamaClient",
                return_value=mock_client,
            ):
                await registry.initialize()

            # response был qwen2.5:14b, но его нет → fallback на qwen2.5:7b
            assert registry.get_model("response") == "qwen2.5:7b"

    @pytest.mark.asyncio
    async def test_no_fallback_when_model_available(self, registry: LLMRegistry):
        """Если модель доступна — override не применяется."""
        with patch("app.services.llm_registry.settings") as mock_settings:
            mock_settings.llm_response_model = "qwen2.5:14b"
            mock_settings.llm_intent_model = ""
            mock_settings.llm_safety_model = ""
            mock_settings.llm_planner_model = ""
            mock_settings.ollama_model = "qwen2.5:7b"
            mock_settings.ollama_host = "http://localhost:11434"
            mock_settings.ollama_timeout = 60

            mock_client = AsyncMock()
            mock_client.list_models = AsyncMock(
                return_value=["qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b"]
            )

            with patch(
                "app.services.llm_registry.OllamaClient",
                return_value=mock_client,
            ):
                await registry.initialize()

            # response = qwen2.5:14b, доступен → остаётся как есть
            assert registry.get_model("response") == "qwen2.5:14b"

    @pytest.mark.asyncio
    async def test_initialize_handles_ollama_unavailable(self, registry: LLMRegistry):
        """Если Ollama недоступен — initialize() не падает."""
        with patch("app.services.llm_registry.settings") as mock_settings:
            mock_settings.llm_response_model = ""
            mock_settings.ollama_model = "qwen2.5:7b"
            mock_settings.ollama_host = "http://localhost:11434"
            mock_settings.ollama_timeout = 60

            mock_client = AsyncMock()
            mock_client.list_models = AsyncMock(side_effect=Exception("Connection refused"))

            with patch(
                "app.services.llm_registry.OllamaClient",
                return_value=mock_client,
            ):
                await registry.initialize()  # Не должно бросать исключение


# ------------------------------------------------------------------ #
# health_check
# ------------------------------------------------------------------ #

class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check_returns_all_roles(self, registry: LLMRegistry):
        with patch("app.services.llm_registry.settings") as mock_settings:
            mock_settings.llm_intent_model = ""
            mock_settings.llm_safety_model = ""
            mock_settings.llm_response_model = "qwen2.5:14b"
            mock_settings.llm_planner_model = ""
            mock_settings.ollama_model = "qwen2.5:7b"
            mock_settings.ollama_host = "http://localhost:11434"
            mock_settings.ollama_timeout = 60

            mock_client = AsyncMock()
            mock_client.list_models = AsyncMock(
                return_value=["qwen2.5:7b", "qwen2.5:14b"]
            )

            with patch(
                "app.services.llm_registry.OllamaClient",
                return_value=mock_client,
            ):
                result = await registry.health_check()

        assert set(result.keys()) == set(ALL_ROLES)
        for role, info in result.items():
            assert "model" in info
            assert "model_loaded" in info

    @pytest.mark.asyncio
    async def test_health_check_model_loaded_flag(self, registry: LLMRegistry):
        """model_loaded=True только если модель в списке доступных."""
        with patch("app.services.llm_registry.settings") as mock_settings:
            mock_settings.llm_intent_model = ""
            mock_settings.llm_safety_model = ""
            mock_settings.llm_response_model = "qwen2.5:14b"
            mock_settings.llm_planner_model = "qwen2.5:32b"
            mock_settings.ollama_model = "qwen2.5:7b"
            mock_settings.ollama_host = "http://localhost:11434"
            mock_settings.ollama_timeout = 60

            mock_client = AsyncMock()
            mock_client.list_models = AsyncMock(
                return_value=["qwen2.5:7b", "qwen2.5:14b"]
            )

            with patch(
                "app.services.llm_registry.OllamaClient",
                return_value=mock_client,
            ):
                result = await registry.health_check()

        assert result["response"]["model_loaded"] is True     # 14b доступна
        assert result["planner"]["model_loaded"] is False     # 32b недоступна
