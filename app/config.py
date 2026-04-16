"""Конфигурация приложения через pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки приложения, читаемые из переменных окружения / .env файла."""

    # Ollama
    ollama_host: str = "http://ollama:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_timeout: int = 60  # секунд

    # LLM модели по ролям (Phase 2, Issue #21)
    # Если не задано — fallback на ollama_model
    llm_intent_model: str = ""   # роль intent_llm (lightweight, ~7b)
    llm_safety_model: str = ""   # роль safety_llm (зарезервировано под v2)
    llm_response_model: str = "" # роль response (primary, ~14b)
    llm_planner_model: str = ""  # роль planner (heavy, ~32b)

    # Эмбеддинги и ChromaDB (Phase 2, Issue #22)
    embedding_model: str = "nomic-embed-text"
    chroma_path: str = "/app/data/chroma"

    # База данных
    db_path: str = "/app/data/health_assistant.db"

    # Приложение
    log_level: str = "INFO"
    secret_key: str = "change-me-in-production"

    # Админ-панель (Basic Auth)
    admin_username: str = "admin"
    admin_password: str = "admin"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def database_url(self) -> str:
        """URL для SQLAlchemy (async драйвер)."""
        return f"sqlite+aiosqlite:///{self.db_path}"

    @property
    def database_url_sync(self) -> str:
        """URL для синхронных операций (Alembic)."""
        return f"sqlite:///{self.db_path}"


settings = Settings()
