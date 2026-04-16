"""ChromaDB vector store wrapper (Phase 2, Issue #22).

Persistent embedded ChromaDB с двумя коллекциями:
- knowledge_base   : RAG-чанки (статические знания)
- semantic_memory  : эмбеддинги Q/A из диалогов (semantic memory v1)

Инициализируется при старте через vector_store.initialize().
Graceful degradation: если ChromaDB недоступен — /health это отражает,
но приложение продолжает работу без RAG/semantic memory.
"""

import logging
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

# Имена коллекций
COLLECTION_KNOWLEDGE_BASE = "knowledge_base"
COLLECTION_SEMANTIC_MEMORY = "semantic_memory"
ALL_COLLECTIONS = [COLLECTION_KNOWLEDGE_BASE, COLLECTION_SEMANTIC_MEMORY]


class VectorStore:
    """Обёртка над chromadb.PersistentClient.

    Все операции (add/query/count/delete) работают только если ChromaDB доступен.
    При недоступности возбуждает RuntimeError с понятным сообщением.
    """

    def __init__(self, path: str | None = None) -> None:
        self._path = path or settings.chroma_path
        self._client: Any = None
        self._collections: dict[str, Any] = {}
        self._available = False

    def initialize(self) -> None:
        """Инициализировать ChromaDB клиент и коллекции.

        Вызывается при старте приложения (lifespan).
        Graceful: при ошибке логирует и продолжает работу без ChromaDB.
        """
        try:
            import chromadb  # type: ignore[import-untyped]

            self._client = chromadb.PersistentClient(path=self._path)
            for name in ALL_COLLECTIONS:
                self._collections[name] = self._client.get_or_create_collection(name)
            self._available = True
            logger.info("ChromaDB инициализирован: path=%s", self._path)
        except Exception as exc:
            logger.error("ChromaDB инициализация не удалась: %s", exc)
            self._available = False

    @property
    def available(self) -> bool:
        """True если ChromaDB доступен."""
        return self._available

    def _get_collection(self, name: str) -> Any:
        if not self._available:
            raise RuntimeError("ChromaDB недоступен — инициализация не прошла")
        if name not in self._collections:
            raise ValueError(
                f"Неизвестная коллекция: {name!r}. Допустимые: {ALL_COLLECTIONS}"
            )
        return self._collections[name]

    # ------------------------------------------------------------------
    # CRUD операции
    # ------------------------------------------------------------------

    def add(
        self,
        collection: str,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        documents: list[str],
    ) -> None:
        """Добавить документы в коллекцию.

        Args:
            collection: Имя коллекции (COLLECTION_*).
            ids: Уникальные идентификаторы документов.
            embeddings: Векторы (размерность должна совпадать с существующими).
            metadatas: Метаданные для каждого документа.
            documents: Тексты документов.
        """
        coll = self._get_collection(collection)
        coll.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )
        logger.debug(
            "VectorStore.add: collection=%s добавлено=%d", collection, len(ids)
        )

    def query(
        self,
        collection: str,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Semantic search в коллекции.

        Args:
            collection: Имя коллекции.
            query_embedding: Вектор запроса.
            n_results: Количество ближайших результатов.
            where: Фильтр по метаданным (ChromaDB where-синтаксис).

        Returns:
            ChromaDB query result с ключами ids, documents, metadatas, distances.
        """
        coll = self._get_collection(collection)
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where
        return coll.query(**kwargs)

    def count(self, collection: str) -> int:
        """Количество документов в коллекции."""
        coll = self._get_collection(collection)
        return coll.count()

    def delete(
        self,
        collection: str,
        ids: list[str] | None = None,
    ) -> None:
        """Удалить документы из коллекции.

        Args:
            collection: Имя коллекции.
            ids: Список id для удаления. Если None — очищает всю коллекцию.
        """
        coll = self._get_collection(collection)
        if ids is None:
            # Очистить всю коллекцию
            existing = coll.get(include=[])
            all_ids: list[str] = existing.get("ids", [])
            if all_ids:
                coll.delete(ids=all_ids)
        else:
            coll.delete(ids=ids)
        logger.debug("VectorStore.delete: collection=%s", collection)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        """Проверить статус ChromaDB.

        Returns:
            Словарь с ключами available и collections (name → count).
        """
        if not self._available:
            return {"available": False, "collections": {}}

        try:
            collections_info: dict[str, Any] = {}
            for name in ALL_COLLECTIONS:
                try:
                    collections_info[name] = self._collections[name].count()
                except Exception as exc:
                    collections_info[name] = f"error: {exc}"
            return {"available": True, "collections": collections_info}
        except Exception as exc:
            logger.error("ChromaDB health_check error: %s", exc)
            return {"available": False, "error": str(exc), "collections": {}}


# Глобальный синглтон
vector_store = VectorStore()
