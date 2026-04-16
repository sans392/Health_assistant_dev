"""FastAPI приложение — точка входа."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация и завершение работы приложения."""
    from app.services.logging_service import setup_json_logging
    setup_json_logging(settings.log_level)
    logger.info("Запуск Health Assistant...")

    # Phase 2: инициализация ChromaDB
    from app.services.vector_store import vector_store
    vector_store.initialize()

    # Phase 2: инициализация LLM Registry (проверка моделей + загрузка из БД)
    from app.services.llm_registry import llm_registry
    from app.db import AsyncSessionLocal
    try:
        async with AsyncSessionLocal() as db:
            await llm_registry.load_from_db(db)
    except Exception as exc:
        logger.warning("LLM Registry: не удалось загрузить конфиги из БД: %s", exc)
    await llm_registry.initialize()

    yield
    logger.info("Завершение работы Health Assistant.")


app = FastAPI(
    title="Health Assistant",
    description="Локальный ассистент для анализа здоровья и тренировок",
    version="0.1.0",
    lifespan=lifespan,
)

# Статические файлы
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Admin API endpoints (issue #12 + #14)
from app.api.admin import router as admin_api_router
app.include_router(admin_api_router)

# Chat API endpoints (issue #13)
from app.api.chat import router as chat_api_router
app.include_router(chat_api_router)

# Admin UI pages — Jinja2 (issue #14)
from app.admin.views import router as admin_ui_router
app.include_router(admin_ui_router)


@app.get("/", include_in_schema=False)
async def root():
    """Редирект на чат."""
    return RedirectResponse(url="/chat")


@app.get("/health")
async def health():
    """Healthcheck endpoint.

    Проверяет доступность Ollama (включая модели по ролям) и ChromaDB.
    """
    from app.services.llm_service import ollama_client
    from app.services.llm_registry import llm_registry
    from app.services.vector_store import vector_store

    ollama_status = await ollama_client.health_check()
    roles_status = await llm_registry.health_check()
    chroma_status = vector_store.health_check()

    return {
        "status": "ok",
        "ollama": ollama_status,
        "llm_roles": roles_status,
        "chromadb": chroma_status,
    }


@app.get("/chat", include_in_schema=False)
async def chat_page():
    """Тестовый чат — HTML-интерфейс (issue #13)."""
    return FileResponse("app/static/chat.html")
