# CLAUDE.md

Этот файл — briefing для Claude при работе над проектом. Читай его в начале каждой сессии.

## О проекте

**Health Assistant** — локальный ассистент для анализа здоровья, физических нагрузок и тренировок.
Работает **оффлайн** (без интернета), LLM через локальный Ollama.

**Текущий этап:** Phase 2 (после MVP). Цель — приблизить систему к архитектуре v2:
мульти-модельный роутинг, RAG (демо-набор), semantic memory, planner-route и
template-планы, расширенное Data Processing, полноценная админка и stage-level
streaming в чате.

**Ключевые документы:**
- `health_assistant_architecture_v2.yaml` — полная архитектура (референс, не менять без запроса)
- `PHASE2_PLAN.md` — план текущего этапа (issues #15–#31, порядок, зависимости)

MVP-план (issues #1–#14) закрыт и удалён из репозитория — историю смотри в git log.

## Tech Stack

| Слой | Технология |
|---|---|
| Backend | FastAPI (Python 3.11+), async |
| Database | SQLite + SQLAlchemy + Alembic |
| LLM | Ollama — multi-model через LLM Registry (роли: intent_llm / safety_llm / response / planner) |
| Vector DB | ChromaDB (embedded, persistent volume) |
| Embeddings | `nomic-embed-text` через Ollama `/api/embeddings` |
| Chat UI | HTML + vanilla JS + WebSocket (stage events + token streaming) |
| Admin UI | Jinja2 + HTMX + Pico CSS |
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
│   ├── main.py                   # FastAPI entry
│   ├── config.py                 # pydantic-settings
│   ├── models/                   # SQLAlchemy
│   ├── schemas/                  # Pydantic
│   ├── services/
│   │   ├── llm_service.py        # Ollama HTTP-клиент
│   │   ├── llm_registry.py       # per-role model routing (Phase 2)
│   │   ├── embedding_service.py  # Phase 2
│   │   ├── vector_store.py       # ChromaDB wrapper (Phase 2)
│   │   ├── logging_service.py
│   │   ├── data_ingestion/       # ЗАГЛУШКА — пока источник данных только seed
│   │   └── data_processing/
│   ├── pipeline/
│   │   ├── context_builder.py
│   │   ├── intent_detection.py         # rule-based + LLM stage 2 (Phase 2)
│   │   ├── safety_check.py             # pattern-based (v2 отложено)
│   │   ├── router.py                   # 4 маршрута (Phase 2)
│   │   ├── tool_executor.py
│   │   ├── template_plan_executor.py   # Phase 2
│   │   ├── planner.py                  # LLM tool-calls loop (Phase 2)
│   │   ├── response_generator.py
│   │   ├── memory_update.py            # Phase 2
│   │   └── orchestrator.py
│   ├── tools/                    # Tool executor tools (+ rag_retrieve)
│   ├── api/                      # Routes (chat WS, admin API)
│   ├── admin/                    # Jinja2 templates + views
│   └── static/                   # CSS/JS для chat и admin
├── scripts/
│   ├── seed_data.py              # Seed Generator v2 (параметризуемый)
│   └── seed_knowledge.py         # Knowledge Base демо-чанки (Phase 2)
├── data/                         # SQLite DB + ChromaDB persistent
└── tests/
```

## Запуск

```bash
# Первый запуск (Ollama уже должен быть поднят с сетью ollama-net)
docker compose up --build

# Миграции
docker compose exec app alembic upgrade head

# Seed data (по умолчанию — 30 дней, 1 профиль, без аномалий)
docker compose exec app python scripts/seed_data.py

# Seed knowledge base (RAG демо-набор)
docker compose exec app python scripts/seed_knowledge.py

# Endpoints
# - http://localhost:8000/                → redirect to /chat
# - http://localhost:8000/chat            → тестовый чат (stage events + streaming)
# - http://localhost:8000/admin           → админ-панель (Basic Auth)
# - http://localhost:8000/admin/llm       → конфиг моделей по ролям
# - http://localhost:8000/admin/seed      → генератор тестовых данных
# - http://localhost:8000/admin/knowledge → RAG browser
# - http://localhost:8000/admin/memory    → semantic memory browser
# - http://localhost:8000/admin/diagnostics → диагностика
# - http://localhost:8000/health          → health check
```

## Конвенции

### Код
- **Язык комментариев и docstrings:** русский (проект локализован, LLM отвечает на русском)
- **Язык идентификаторов:** английский
- **Type hints:** обязательно везде
- **Async:** весь I/O (DB, LLM, HTTP, ChromaDB где возможно) — async
- **Pydantic:** для всех schemas на границах API
- **Dataclasses или Pydantic:** для внутренних DTO (pipeline result, intent result, stage event)

### Именование
- Модули pipeline: snake_case (`intent_detection.py`, `template_plan_executor.py`)
- Классы: PascalCase (`IntentDetector`, `PlannerAgent`, `LLMRegistry`)
- DTO: `{Name}Result` или `{Name}Event` (`PipelineResult`, `StageEvent`)

## Принципы Phase 2 (КРИТИЧНО)

Придерживаемся минимализма. **В этом этапе не реализуем**:

- **Data ingestion из реального API** (+ Anomaly detection, Deduplication) — откладывается.
  Источник данных — только Seed Generator v2.
- **Safety Check v2** (контекстный LLM-анализ) — остаётся pattern-based.
  Комментарий-заглушка в `safety_check.py`.
- **Output Validation v2** (hallucination check, medical advice check) — skipped во всех
  маршрутах, место в pipeline зарезервировано.
- **Periodization** (макро/мезо/микроциклы) — заглушка-модуль с docstring "TODO v3".
- **Proactive Alerts** (HRV drop, RHR spike, weekly summary) — не запускаются.
  Пустой модуль `app/services/alerts.py` с описанием.
- **Testing & Evaluation** (eval-датасеты, hallucination tests, latency benchmarks,
  RAG quality) — только unit-тесты критичных модулей.

Если архитектура описывает что-то из списка выше — **не реализовывать**, оставить заглушку
с комментарием `TODO v3`.

Knowledge Base (RAG) — **минимальный демо-набор** (20–40 чанков по всем 5 категориям YAML).
Комментарий в коде / README, как расширять.

## Работа с Ollama и моделями

- Хост: `http://ollama:11434` (внутри сети `ollama-net`)
- **Multi-model через LLM Registry**. Роли:
  - `intent_llm` — классификация intent при low-confidence (дефолт: lightweight, YAML → `qwen2.5:7b`)
  - `safety_llm` — зарезервировано под v2, сейчас не вызывается
  - `response` — основной генератор ответов (дефолт: primary, YAML → `qwen2.5:14b`)
  - `planner` — сложное планирование + генерация планов (дефолт: heavy, YAML → `qwen2.5:32b`)
- Конфиг ролей — из `.env` + runtime overrides в SQLite (`llm_role_config`, меняется в админке).
- Если указанная модель недоступна в Ollama — fallback на `OLLAMA_MODEL` (базовая) с WARN в логах.
- Всегда логировать каждый LLM-вызов в `llm_calls`: role, model, длину промпта, длину ответа,
  duration_ms, request_id.
- Timeout: 60s (heavy — 120s), 1 retry при timeout.

## Pipeline flow (Phase 2 reference)

```
User Query
  → Context Builder         # session history + profile + RAG + semantic memory
  → Intent Detection        # rule-based → LLM fallback (low confidence)
  → Safety Check            # pattern-based (v2 отложено)
  → Router                  # fast_direct_answer | tool_simple | template_plan | planner
  ↓
  [blocked]       → return redirect
  [fast_direct]   → Response Generator → return
  [tool_simple]   → Tool Executor → Response Generator → return
  [template_plan] → Template Executor (шаги шаблона) → Response Generator → return
  [planner]       → Planner loop (LLM ↔ Tool Executor, max N iter) → Response Generator → return
  ↓
  Memory Update (async): short-term + long-term + semantic
  Response Delivery (stage events + token streaming)
```

## Issues и фазы

Смотри `PHASE2_PLAN.md`. Issues пронумерованы **#15–#31** (MVP закрыт — #1–#14).
Выполнять в порядке зависимостей, указанном в плане (секция «Оптимальный порядок исполнения»).

**Старт этапа:**
1. #15 — LLM Registry
2. #17 — Schema v2 + migration
3. #16 — ChromaDB + Embeddings
4. #18 — Knowledge Base демо + retrieval
5. #19 — Semantic Memory v1

...далее по плану.

## Тесты

- Unit-тесты для критичных модулей: intent detection, safety, routing, data processing,
  planner (mock Ollama), template executor, rag retrieval (mock Chroma).
- Интеграционные тесты — в #31 (Orchestrator v2): fast_path, tool_simple, template_plan,
  planner loop, safety block.
- Фреймворк: `pytest` + `pytest-asyncio`
- Запуск: `docker compose exec app pytest`

## Что делать при получении задачи на issue

1. Прочитать issue полностью (acceptance criteria!).
2. Прочитать `PHASE2_PLAN.md` — понять контекст issue в рамках этапа и её зависимости.
3. Проверить, что предыдущие зависимости реализованы (см. граф зависимостей).
4. Реализовать **минимально достаточно**, не добавлять лишнего.
   Если архитектура YAML описывает что-то из списка «отложено» — не реализовывать.
5. Проверить:
   - `docker compose up --build` запускается без ошибок
   - `alembic upgrade head` проходит
   - Релевантные тесты проходят (`pytest`)
   - Чат `/chat` и админка `/admin` открываются
6. Коммитить на ветку вида `claude/<short-slug>`, PR создавать только по явному запросу.
