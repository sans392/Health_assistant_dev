"""Unit-тесты для VectorStore (Issue #22).

Используют in-memory ChromaDB (chromadb.EphemeralClient) вместо PersistentClient,
чтобы тесты не зависели от файловой системы.
"""

import pytest
from unittest.mock import MagicMock, patch

from app.services.vector_store import (
    VectorStore,
    COLLECTION_KNOWLEDGE_BASE,
    COLLECTION_SEMANTIC_MEMORY,
    ALL_COLLECTIONS,
)


# ------------------------------------------------------------------ #
# Вспомогательные fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def in_memory_store(tmp_path) -> VectorStore:
    """VectorStore с изолированным PersistentClient в tmp_path.

    Использует tmp_path (уникальная директория на каждый тест) вместо
    EphemeralClient, т.к. EphemeralClient в chromadb >= 1.x делит состояние
    между инстансами в рамках одного Python-процесса.
    """
    try:
        import chromadb
    except ImportError:
        pytest.skip("chromadb не установлен")

    chroma_dir = str(tmp_path / "chroma")
    store = VectorStore(path=chroma_dir)
    store.initialize()
    return store


SAMPLE_IDS = ["chunk_1", "chunk_2"]
SAMPLE_DOCS = ["Текст первого чанка", "Текст второго чанка"]
SAMPLE_EMBEDDINGS = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
SAMPLE_META = [
    {"category": "training_principles", "source": "test"},
    {"category": "recovery_science", "source": "test"},
]


# ------------------------------------------------------------------ #
# initialize
# ------------------------------------------------------------------ #

class TestInitialize:
    def test_initialize_success(self):
        """Успешная инициализация с mock chromadb."""
        store = VectorStore(path="/tmp/test_chroma_init")

        mock_client = MagicMock()
        mock_coll = MagicMock()
        mock_client.get_or_create_collection = MagicMock(return_value=mock_coll)

        mock_chromadb = MagicMock()
        mock_chromadb.PersistentClient = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"chromadb": mock_chromadb}):
            store.initialize()

        assert store.available is True

    def test_initialize_graceful_on_error(self):
        """При ошибке инициализации available=False, исключение не бросается."""
        store = VectorStore(path="/tmp/test_chroma_error")

        mock_chromadb = MagicMock()
        mock_chromadb.PersistentClient = MagicMock(
            side_effect=Exception("ChromaDB недоступен")
        )

        with patch.dict("sys.modules", {"chromadb": mock_chromadb}):
            store.initialize()

        assert store.available is False

    def test_unavailable_raises_on_operations(self):
        """Если ChromaDB недоступен — операции бросают RuntimeError."""
        store = VectorStore()
        store._available = False

        with pytest.raises(RuntimeError, match="ChromaDB недоступен"):
            store.count(COLLECTION_KNOWLEDGE_BASE)


# ------------------------------------------------------------------ #
# add / count
# ------------------------------------------------------------------ #

class TestAddAndCount:
    def test_add_and_count(self, in_memory_store: VectorStore):
        """Добавление документов увеличивает count."""
        initial = in_memory_store.count(COLLECTION_KNOWLEDGE_BASE)

        in_memory_store.add(
            collection=COLLECTION_KNOWLEDGE_BASE,
            ids=SAMPLE_IDS,
            embeddings=SAMPLE_EMBEDDINGS,
            metadatas=SAMPLE_META,
            documents=SAMPLE_DOCS,
        )

        assert in_memory_store.count(COLLECTION_KNOWLEDGE_BASE) == initial + 2

    def test_semantic_memory_independent(self, in_memory_store: VectorStore):
        """Коллекции независимы."""
        in_memory_store.add(
            collection=COLLECTION_KNOWLEDGE_BASE,
            ids=["k1"],
            embeddings=[[0.1, 0.2, 0.3]],
            metadatas=[{"source": "test"}],
            documents=["знание"],
        )

        assert in_memory_store.count(COLLECTION_SEMANTIC_MEMORY) == 0
        assert in_memory_store.count(COLLECTION_KNOWLEDGE_BASE) == 1

    def test_unknown_collection_raises(self, in_memory_store: VectorStore):
        with pytest.raises(ValueError, match="Неизвестная коллекция"):
            in_memory_store.count("nonexistent_collection")


# ------------------------------------------------------------------ #
# query
# ------------------------------------------------------------------ #

class TestQuery:
    def test_query_returns_results(self, in_memory_store: VectorStore):
        """query возвращает ближайшие документы."""
        in_memory_store.add(
            collection=COLLECTION_KNOWLEDGE_BASE,
            ids=SAMPLE_IDS,
            embeddings=SAMPLE_EMBEDDINGS,
            metadatas=SAMPLE_META,
            documents=SAMPLE_DOCS,
        )

        result = in_memory_store.query(
            collection=COLLECTION_KNOWLEDGE_BASE,
            query_embedding=[0.1, 0.2, 0.3],
            n_results=1,
        )

        assert "ids" in result
        assert len(result["ids"][0]) == 1

    def test_query_with_where_filter(self, in_memory_store: VectorStore):
        """where-фильтр работает по метаданным."""
        in_memory_store.add(
            collection=COLLECTION_KNOWLEDGE_BASE,
            ids=SAMPLE_IDS,
            embeddings=SAMPLE_EMBEDDINGS,
            metadatas=SAMPLE_META,
            documents=SAMPLE_DOCS,
        )

        result = in_memory_store.query(
            collection=COLLECTION_KNOWLEDGE_BASE,
            query_embedding=[0.1, 0.2, 0.3],
            n_results=2,
            where={"category": "training_principles"},
        )

        # Только документы с category=training_principles
        returned_ids = result["ids"][0]
        assert "chunk_1" in returned_ids
        assert "chunk_2" not in returned_ids

    def test_query_n_results_respected(self, in_memory_store: VectorStore):
        """n_results ограничивает количество результатов."""
        in_memory_store.add(
            collection=COLLECTION_KNOWLEDGE_BASE,
            ids=SAMPLE_IDS,
            embeddings=SAMPLE_EMBEDDINGS,
            metadatas=SAMPLE_META,
            documents=SAMPLE_DOCS,
        )

        result = in_memory_store.query(
            collection=COLLECTION_KNOWLEDGE_BASE,
            query_embedding=[0.1, 0.2, 0.3],
            n_results=1,
        )
        assert len(result["ids"][0]) == 1


# ------------------------------------------------------------------ #
# delete
# ------------------------------------------------------------------ #

class TestDelete:
    def test_delete_by_ids(self, in_memory_store: VectorStore):
        """Удаление по id уменьшает count."""
        in_memory_store.add(
            collection=COLLECTION_KNOWLEDGE_BASE,
            ids=SAMPLE_IDS,
            embeddings=SAMPLE_EMBEDDINGS,
            metadatas=SAMPLE_META,
            documents=SAMPLE_DOCS,
        )

        in_memory_store.delete(
            collection=COLLECTION_KNOWLEDGE_BASE,
            ids=["chunk_1"],
        )

        assert in_memory_store.count(COLLECTION_KNOWLEDGE_BASE) == 1

    def test_delete_all(self, in_memory_store: VectorStore):
        """delete(ids=None) очищает всю коллекцию."""
        in_memory_store.add(
            collection=COLLECTION_KNOWLEDGE_BASE,
            ids=SAMPLE_IDS,
            embeddings=SAMPLE_EMBEDDINGS,
            metadatas=SAMPLE_META,
            documents=SAMPLE_DOCS,
        )

        in_memory_store.delete(collection=COLLECTION_KNOWLEDGE_BASE)

        assert in_memory_store.count(COLLECTION_KNOWLEDGE_BASE) == 0


# ------------------------------------------------------------------ #
# health_check
# ------------------------------------------------------------------ #

class TestHealthCheck:
    def test_health_check_available(self, in_memory_store: VectorStore):
        result = in_memory_store.health_check()
        assert result["available"] is True
        assert COLLECTION_KNOWLEDGE_BASE in result["collections"]
        assert COLLECTION_SEMANTIC_MEMORY in result["collections"]

    def test_health_check_unavailable(self):
        store = VectorStore()
        store._available = False
        result = store.health_check()
        assert result["available"] is False
        assert result["collections"] == {}

    def test_health_check_shows_counts(self, in_memory_store: VectorStore):
        """health_check отражает актуальное количество документов."""
        in_memory_store.add(
            collection=COLLECTION_KNOWLEDGE_BASE,
            ids=["k1"],
            embeddings=[[0.1, 0.2, 0.3]],
            metadatas=[{"source": "test"}],
            documents=["doc"],
        )
        result = in_memory_store.health_check()
        assert result["collections"][COLLECTION_KNOWLEDGE_BASE] == 1
        assert result["collections"][COLLECTION_SEMANTIC_MEMORY] == 0
