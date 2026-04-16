"""Unit-тесты для EmbeddingService (Issue #22).

Покрывают: embed одной строки, батч, retry при таймауте,
ошибка при двух таймаутах, check_model_available.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from app.services.embedding_service import EmbeddingService


# ------------------------------------------------------------------ #
# Вспомогательные fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def service() -> EmbeddingService:
    return EmbeddingService(host="http://localhost:11434", model="nomic-embed-text")


FAKE_VECTOR = [0.1, 0.2, 0.3, 0.4]


def _make_mock_response(vector: list[float]) -> MagicMock:
    """Создать mock httpx.Response с эмбеддингом."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value={"embedding": vector})
    return mock_resp


# ------------------------------------------------------------------ #
# embed
# ------------------------------------------------------------------ #

class TestEmbed:
    @pytest.mark.asyncio
    async def test_embed_single_string(self, service: EmbeddingService):
        """embed('текст') возвращает список из одного вектора."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=_make_mock_response(FAKE_VECTOR))

        with patch.object(service, "_make_client", return_value=mock_client):
            result = await service.embed("привет мир")

        assert result == [FAKE_VECTOR]

    @pytest.mark.asyncio
    async def test_embed_list_of_strings(self, service: EmbeddingService):
        """embed(['a', 'b', 'c']) возвращает список из трёх векторов."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=_make_mock_response(FAKE_VECTOR))

        with patch.object(service, "_make_client", return_value=mock_client):
            result = await service.embed(["a", "b", "c"])

        assert len(result) == 3
        assert all(v == FAKE_VECTOR for v in result)

    @pytest.mark.asyncio
    async def test_embed_calls_correct_endpoint(self, service: EmbeddingService):
        """Запрос идёт на /api/embeddings с нужной моделью."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=_make_mock_response(FAKE_VECTOR))

        with patch.object(service, "_make_client", return_value=mock_client):
            await service.embed("test")

        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/api/embeddings"
        payload = call_args[1]["json"]
        assert payload["model"] == "nomic-embed-text"
        assert payload["prompt"] == "test"

    @pytest.mark.asyncio
    async def test_embed_empty_string(self, service: EmbeddingService):
        """embed('') возвращает вектор (пустая строка — допустимый ввод)."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=_make_mock_response(FAKE_VECTOR))

        with patch.object(service, "_make_client", return_value=mock_client):
            result = await service.embed("")

        assert result == [FAKE_VECTOR]


# ------------------------------------------------------------------ #
# Retry при таймауте
# ------------------------------------------------------------------ #

class TestRetry:
    @pytest.mark.asyncio
    async def test_retry_on_first_timeout(self, service: EmbeddingService):
        """При первом таймауте — retry, возвращает результат."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        # Первый вызов — таймаут, второй — успех
        mock_client.post = AsyncMock(
            side_effect=[
                httpx.TimeoutException("timeout"),
                _make_mock_response(FAKE_VECTOR),
            ]
        )

        with patch.object(service, "_make_client", return_value=mock_client):
            result = await service.embed("текст")

        assert result == [FAKE_VECTOR]
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_two_timeouts(self, service: EmbeddingService):
        """После двух таймаутов — raises ValueError."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(
            side_effect=httpx.TimeoutException("timeout")
        )

        with patch.object(service, "_make_client", return_value=mock_client):
            with pytest.raises(ValueError, match="таймаут"):
                await service.embed("текст")


# ------------------------------------------------------------------ #
# check_model_available
# ------------------------------------------------------------------ #

class TestCheckModelAvailable:
    @pytest.mark.asyncio
    async def test_returns_true_when_model_present(self, service: EmbeddingService):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value={
            "models": [{"name": "nomic-embed-text:latest"}]
        })
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch.object(service, "_make_client", return_value=mock_client):
            result = await service.check_model_available()

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_model_absent(self, service: EmbeddingService):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value={
            "models": [{"name": "qwen2.5:7b"}]
        })
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch.object(service, "_make_client", return_value=mock_client):
            result = await service.check_model_available()

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_network_error(self, service: EmbeddingService):
        """При сетевой ошибке возвращает False (не бросает)."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with patch.object(service, "_make_client", return_value=mock_client):
            result = await service.check_model_available()

        assert result is False
