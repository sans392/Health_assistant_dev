# CLAUDE.md

Этот файл — briefing для Claude при работе над проектом. Читай его в начале каждой сессии.

## О проекте

**Health Assistant** — локальный ассистент для анализа здоровья, физических нагрузок и тренировок. Работает **оффлайн** (без интернета), LLM через локальный Ollama.

**Ключевые документы:**
- `health_assistant_architecture_v2.yaml` — полная архитектура (референс, не менять без запроса)
- `MVP_PLAN.md` — план реализации MVP (14 issues, порядок, зависимости)

## Tech Stack

| Слой | Технология |
|---|---|
| Backend | FastAPI (Python 3.11+), async |
| Database | SQLite + SQLAlchemy + Alembic |
| LLM | Ollama (qwen2.5:7b в MVP) через httpx |
| Chat UI | HTML + vanilla JS + WebSocket |
| Admin UI | Jinja2 templates + HTMX + Pico CSS |
| Container | Docker Compose |
| Сеть | Внешняя `ollama-net` (Ollama запущен отдельно) |

## Структура проекта

```
health_assistant/
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── requirements.txt
├── alembic.ini
├── alembic/
├── app/
│   ├── main.py              # FastAPI entry
│   ├── config.py             # pydantic-settings
│   ├── models/               # SQLAlchemy
│   ├── schemas/              # Pydantic
│   ├── services/             # Business logic (Ollama, logging, data_processing)
│   ├── pipeline/             # Pipeline stages (intent, safety, router, ...)
│   ├── tools/                # Tool executor tools
│   ├── api/                  # Routes (chat WS, admin API)
│   ├── admin/                # Jinja2 templates
│   └── static/               # CSS/JS для chat и admin
├── scripts/
│   └── seed_data.py
├── data/                     # SQLite DB, persistent
└── tests/
```

## Запуск

```bash
# Первый запуск (Ollama уже должен быть поднят с сетью ollama-net)
docker compose up --build

# Миграции
docker compose exec app alembic upgrade head

# Seed data
docker compose exec app python scripts/seed_data.py

# Endpoints
# - http://localhost:8000/         → redirect to /chat
# - http://localhost:8000/chat     → тестовый чат
# - http://localhost:8000/admin    → админ-панель (Basic Auth)
# - http://localhost:8000/health   → health check
```

## Конвенции

### Код
- **Язык комментариев и docstrings:** русский (проект локализован под русского пользователя, LLM отвечает на русском)
- **Язык идентификаторов:** английский
- **Type hints:** обязательно везде
- **Async:** весь I/O (DB, LLM, HTTP) — async
- **Pydantic:** для всех schemas на границах API
- **Dataclasses или Pydantic:** для внутренних DTO (pipeline result, intent result)

### Git
- Ветка для MVP: `claude/health-assistant-mvp-plan-0art6`
- **ВСЕ коммиты и push идут в эту ветку**
- Один issue = один или несколько атомарных коммитов
- Commit message: `[Issue #N] Короткое описание` + тело с деталями

### Именование
- Модули pipeline: snake_case (`intent_detection.py`, `safety_check.py`)
- Классы: PascalCase (`IntentDetector`, `SafetyChecker`)
- DTO: `{Name}Result` (`IntentResult`, `PipelineResult`)

## MVP-принципы (КРИТИЧНО)

Следуем минимализму. **НЕ добавлять** в MVP:
- RAG, ChromaDB, embedding модели
- Multi-model routing (только ОДНА модель)
- Semantic memory (vector DB)
- Output validation (hallucination check)
- Data ingestion из внешнего API (только seed data)
- Anomaly detection / flagging
- Deduplication logic
- Расчёт Recovery Score, Strain Score, Overtraining
- Proactive alerts
- Streaming ответов

Если архитектура описывает что-то сложное — **проверь, в MVP ли это**. В `MVP_PLAN.md` явно указано, что входит и что отложено.

## Работа с Ollama

- Хост: `http://ollama:11434` (внутри сети `ollama-net`)
- Модель: из `.env` (`OLLAMA_MODEL=qwen2.5:7b`)
- Всегда логировать: модель, длину промпта, длину ответа, duration_ms
- Timeout: 60s, 1 retry при timeout
- Health check проверяет доступность Ollama

## Pipeline flow (reference)

```
User Query
  → Context Builder         # session + profile
  → Intent Detection        # rule-based
  → Safety Check            # pattern-based
  → Router                  # fast_path / standard
  ↓
  [blocked] → return redirect
  [fast_path] → Response Generator → return
  [standard]
    → Tool Executor         # DB queries
    → Data Processing       # calculations
    → Response Generator    # LLM output
    → return
```

## Issues и фазы

Смотри `MVP_PLAN.md`. Issues пронумерованы #1–#14. Выполнять в порядке зависимостей.

**Порядок выполнения:**
1. #1 (scaffold) → #2 (DB) → #3 (Ollama)
2. #4 + #5 (intent + safety, независимы)
3. #6 (context) → #7 (router)
4. #8 (tools) → #9 (processing)
5. #10 (response)
6. #11 (orchestrator — клеит всё)
7. #12 (logging)
8. #13 (test chat) + #14 (admin panel)

## Тесты

- Unit-тесты для критичных модулей: intent detection, safety, routing
- Интеграционные тесты — после #11 (Orchestrator)
- Фреймворк: `pytest` + `pytest-asyncio`
- Запуск: `docker compose exec app pytest`

## Что делать при получении задачи на issue

1. Прочитать issue полностью (acceptance criteria!)
2. Прочитать `MVP_PLAN.md` чтобы понять контекст в рамках всего проекта
3. Проверить зависимости issue — готовы ли предыдущие
4. Реализовать минимально достаточно, не добавлять лишнего
5. Проверить: запускается ли `docker compose up`, проходят ли тесты
6. Коммит + push в `claude/health-assistant-mvp-plan-0art6`
7. Пометить issue как completed (комментарий или закрытие)
