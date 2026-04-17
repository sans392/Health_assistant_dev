"""Сервис логирования пайплайна (issue #12).

Два канала:
1. PipelineLog → SQLite (структурированные данные для admin panel)
2. Structured JSON → stdout (для docker logs / ELK)
"""

import json
import logging
import sys
import uuid
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.pipeline_log import PipelineLog
from app.pipeline.orchestrator import PipelineResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON Formatter для структурированного логирования в stdout
# ---------------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    """Форматирует log-записи в однострочный JSON для docker logs."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        log_data: dict = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Добавляем extra-поля, если они есть
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
            ):
                try:
                    json.dumps(value)   # проверяем сериализуемость
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        try:
            return json.dumps(log_data, ensure_ascii=False)
        except (TypeError, ValueError):
            return json.dumps({"level": "ERROR", "message": "Ошибка сериализации лога"})


def setup_json_logging(log_level: str = "INFO") -> None:
    """Настроить структурированное JSON-логирование в stdout.

    Вызывается один раз при старте приложения (в lifespan FastAPI).

    Args:
        log_level: Уровень логирования (DEBUG, INFO, WARN, ERROR).
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    # Убираем старые handler'ы и ставим JSON
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)


# ---------------------------------------------------------------------------
# Logging Service
# ---------------------------------------------------------------------------

class LoggingService:
    """Сервис записи логов пайплайна в SQLite и stdout.

    Использование:
        await logging_service.log_pipeline_request(
            user_id=user_id,
            session_id=session_id,
            result=pipeline_result,
            db=db,
        )
    """

    async def log_pipeline_request(
        self,
        user_id: str,
        session_id: str,
        result: PipelineResult,
        db: AsyncSession,
        request_id: str | None = None,
    ) -> str:
        """Записать PipelineLog после обработки одного запроса.

        Сохраняет запись в SQLite и выводит структурированный лог в stdout.

        Args:
            user_id: Идентификатор пользователя.
            session_id: Идентификатор сессии.
            result: Результат работы PipelineOrchestrator.
            db: Асинхронная сессия SQLAlchemy.
            request_id: UUID запроса (генерируется автоматически если не передан).

        Returns:
            request_id — уникальный идентификатор лога.
        """
        # Используем request_id из результата пайплайна если он уже сгенерирован
        request_id = request_id or getattr(result, "request_id", None) or str(uuid.uuid4())

        # 1. Запись в SQLite
        await self._write_to_db(
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            result=result,
            db=db,
        )

        # 2. Structured logging в stdout
        self._log_to_stdout(request_id=request_id, user_id=user_id, result=result)

        return request_id

    async def _write_to_db(
        self,
        request_id: str,
        user_id: str,
        session_id: str,
        result: PipelineResult,
        db: AsyncSession,
    ) -> None:
        """Создать запись PipelineLog в базе данных."""
        try:
            log_entry = PipelineLog(
                id=request_id,
                user_id=user_id,
                session_id=session_id,
                timestamp=datetime.utcnow(),
                raw_query=result.raw_query,
                intent=result.intent,
                intent_confidence=result.intent_confidence,
                route=result.route,
                fast_path=result.fast_path,
                safety_level=result.safety_level,
                tools_called=result.tools_called,
                modules_used=result.modules_used,
                llm_model_used=result.llm_model,
                llm_calls_count=result.llm_calls_count,
                total_duration_ms=result.duration_ms,
                response_length=len(result.response_text),
                errors=result.errors if result.errors else None,
                response_text=result.response_text,
                stage_trace=getattr(result, "stage_trace", None) or None,
                llm_role_usage=getattr(result, "llm_role_usage", None) or None,
            )
            db.add(log_entry)
            await db.commit()
        except Exception as exc:
            logger.error(
                "LoggingService: ошибка записи в БД | request_id=%s error=%s",
                request_id, exc, exc_info=True,
            )
            await db.rollback()

    def _log_to_stdout(
        self,
        request_id: str,
        user_id: str,
        result: PipelineResult,
    ) -> None:
        """Вывести структурированный лог запроса в stdout."""
        has_errors = bool(result.errors)
        has_safety_flag = result.safety_level not in ("ok", "unknown")

        log_record = {
            "request_id": request_id,
            "user_id": user_id,
            "intent": result.intent,
            "intent_confidence": result.intent_confidence,
            "route": result.route,
            "fast_path": result.fast_path,
            "blocked": result.blocked,
            "safety_level": result.safety_level,
            "tools_called": result.tools_called,
            "modules_used": result.modules_used,
            "llm_model": result.llm_model,
            "llm_calls_count": result.llm_calls_count,
            "duration_ms": result.duration_ms,
            "response_length": len(result.response_text),
            "has_errors": has_errors,
        }

        if has_errors:
            logger.error(
                "pipeline_request",
                extra={**log_record, "errors": result.errors},
            )
        elif has_safety_flag:
            logger.warning("pipeline_request", extra=log_record)
        else:
            logger.info("pipeline_request", extra=log_record)


# Глобальный экземпляр (singleton)
logging_service = LoggingService()
