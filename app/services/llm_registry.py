"""LLM Registry — мульти-модельный роутинг по ролям (Phase 2, Issue #21).

Каждая роль в пайплайне может использовать свою модель Ollama.
Конфиг: .env переменные + runtime overrides в памяти + персистентность в SQLite.

Роли:
- intent_llm : классификация intent при low-confidence (lightweight ~7b)
- safety_llm : зарезервировано под Safety v2, сейчас не вызывается
- response   : основной генератор ответов (primary ~14b)
- planner    : сложное планирование (heavy ~32b)
"""

import logging
from typing import Any

from app.config import settings
from app.services.llm_service import OllamaClient

logger = logging.getLogger(__name__)

# Таймаут (сек) для каждой роли
ROLE_TIMEOUTS: dict[str, int] = {
    "intent_llm": 60,
    "safety_llm": 60,
    "response": 60,
    "planner": 120,
}

ALL_ROLES = list(ROLE_TIMEOUTS.keys())


class LLMRegistry:
    """Реестр LLM-клиентов по ролям.

    Приоритет выбора модели для роли:
      1. Runtime override (set_model / load_from_db)
      2. Переменная окружения LLM_<ROLE>_MODEL
      3. Базовая модель OLLAMA_MODEL

    Fallback при старте: если модель для роли недоступна в Ollama — переключается на
    базовую модель и логирует WARN.
    """

    def __init__(self) -> None:
        # Runtime overrides (роль → имя модели), в т.ч. fallback-значения
        self._overrides: dict[str, str] = {}
        # Кэш OllamaClient по роли (сбрасывается при set_model)
        self._clients: dict[str, OllamaClient] = {}
        # Кэш списка доступных моделей (None = не загружено)
        self._available_models: list[str] | None = None

    # ------------------------------------------------------------------
    # Выбор модели
    # ------------------------------------------------------------------

    def _env_model(self, role: str) -> str:
        """Получить модель из переменных окружения для роли."""
        mapping = {
            "intent_llm": settings.llm_intent_model,
            "safety_llm": settings.llm_safety_model,
            "response": settings.llm_response_model,
            "planner": settings.llm_planner_model,
        }
        return mapping.get(role, "") or settings.ollama_model

    def get_model(self, role: str) -> str:
        """Получить имя модели для роли.

        Приоритет: runtime override > .env переменная > базовая модель.
        """
        if role in self._overrides:
            return self._overrides[role]
        return self._env_model(role)

    def set_model(self, role: str, model: str) -> None:
        """Установить runtime override модели для роли (in-memory).

        Сбрасывает кэш клиента для данной роли.
        Для персистентного сохранения используй set_model_persistent().

        Raises:
            ValueError: Если роль неизвестна.
        """
        if role not in ROLE_TIMEOUTS:
            raise ValueError(
                f"Неизвестная роль: {role!r}. Допустимые: {ALL_ROLES}"
            )
        self._overrides[role] = model
        self._clients.pop(role, None)
        logger.info("LLM Registry: роль '%s' → модель '%s'", role, model)

    # ------------------------------------------------------------------
    # Клиент
    # ------------------------------------------------------------------

    def get_client(self, role: str) -> OllamaClient:
        """Получить OllamaClient для роли (с кэшированием).

        Сбрасывает кэш при вызове set_model().
        """
        if role not in self._clients:
            model = self.get_model(role)
            timeout = ROLE_TIMEOUTS.get(role, settings.ollama_timeout)
            self._clients[role] = OllamaClient(
                host=settings.ollama_host,
                model=model,
                timeout=timeout,
                role=role,
            )
        return self._clients[role]

    # ------------------------------------------------------------------
    # Инициализация и fallback
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Проверить доступность моделей в Ollama и применить fallback.

        Вызывается при старте приложения (lifespan).
        Если модель роли недоступна — переключается на базовую модель WARN.
        """
        probe = OllamaClient(host=settings.ollama_host, timeout=10)
        try:
            available = await probe.list_models()
        except Exception as exc:
            logger.warning(
                "LLM Registry: не удалось получить список моделей Ollama: %s", exc
            )
            return

        self._available_models = available

        for role in ALL_ROLES:
            model = self.get_model(role)
            if not any(model in m for m in available):
                fallback = settings.ollama_model
                if model == fallback:
                    # Базовая модель тоже недоступна — просто предупреждаем
                    logger.warning(
                        "LLM Registry: базовая модель '%s' для роли '%s' "
                        "не найдена в Ollama",
                        model, role,
                    )
                else:
                    logger.warning(
                        "LLM Registry: модель '%s' для роли '%s' недоступна. "
                        "Fallback → '%s'",
                        model, role, fallback,
                    )
                    self._overrides[role] = fallback
                    self._clients.pop(role, None)

    # ------------------------------------------------------------------
    # Персистентность через SQLite (llm_role_config)
    # ------------------------------------------------------------------

    async def load_from_db(self, db: Any) -> None:
        """Загрузить конфиг ролей из таблицы llm_role_config.

        Вызывается после инициализации БД при старте.
        DB-overrides применяются поверх .env значений, но ниже runtime set_model().
        """
        try:
            from sqlalchemy import select
            from app.models.llm_role_config import LLMRoleConfig

            result = await db.execute(select(LLMRoleConfig))
            rows = result.scalars().all()
            loaded = 0
            for row in rows:
                # Применяем как overrides (сбрасываем кэш клиента)
                self._overrides[row.role] = row.model
                self._clients.pop(row.role, None)
                loaded += 1
            if loaded:
                logger.info(
                    "LLM Registry: загружено %d конфигов ролей из БД", loaded
                )
        except Exception as exc:
            logger.warning(
                "LLM Registry: не удалось загрузить конфиги из БД: %s", exc
            )

    async def set_model_persistent(self, role: str, model: str, db: Any) -> None:
        """Установить модель для роли и сохранить в БД.

        Args:
            role: Имя роли (intent_llm / safety_llm / response / planner).
            model: Имя модели Ollama.
            db: AsyncSession SQLAlchemy.
        """
        from datetime import datetime
        from sqlalchemy import select
        from app.models.llm_role_config import LLMRoleConfig

        # Обновить in-memory
        self.set_model(role, model)

        # Upsert в БД
        result = await db.execute(
            select(LLMRoleConfig).where(LLMRoleConfig.role == role)
        )
        config = result.scalar_one_or_none()
        if config is None:
            config = LLMRoleConfig(role=role, model=model, updated_at=datetime.utcnow())
            db.add(config)
        else:
            config.model = model
            config.updated_at = datetime.utcnow()

        await db.commit()
        logger.info(
            "LLM Registry: роль '%s' → '%s' сохранена в БД", role, model
        )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> dict[str, Any]:
        """Проверить статус всех ролей.

        Returns:
            Словарь role → {model, model_loaded}.
        """
        probe = OllamaClient(host=settings.ollama_host, timeout=10)
        try:
            available = await probe.list_models()
            self._available_models = available
        except Exception as exc:
            logger.warning("LLM Registry health check: %s", exc)
            available = []

        roles_status: dict[str, Any] = {}
        for role in ALL_ROLES:
            model = self.get_model(role)
            roles_status[role] = {
                "model": model,
                "model_loaded": any(model in m for m in available),
            }
        return roles_status


# Глобальный синглтон
llm_registry = LLMRegistry()
