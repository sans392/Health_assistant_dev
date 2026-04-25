# CLAUDE.md

Этот файл — briefing для Claude при работе над проектом. Читай его в начале каждой сессии.

## О проекте

**Health Assistant** — локальный ассистент для анализа здоровья, физических нагрузок и тренировок.
Работает **оффлайн** (без интернета), LLM через локальный Ollama.

Режим работы по умолчанию — поддержка: багфиксы, небольшие доработки, подготовка
заглушек из списка «TODO v3» к реализации. Если новых инструкций нет — не
затевай крупных рефакторингов.


## Tech Stack

| Слой | Технология |
|---|---|
| Backend | FastAPI (Python 3.11+), async |
| Database | SQLite + SQLAlchemy + Alembic |
| LLM | Ollama — multi-model через LLM Registry (роли: intent_llm / safety_llm / response / planner) |
| Vector DB | ChromaDB (embedded, persistent volume) |
| Embeddings | `nomic-embed-text-v2-moe` через Ollama `/api/embeddings` |
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
│   │   ├── llm_registry.py       # per-role model routing
│   │   ├── embedding_service.py
│   │   ├── vector_store.py       # ChromaDB wrapper
│   │   ├── logging_service.py
│   │   ├── data_ingestion/       # ЗАГЛУШКА — пока источник данных только seed
│   │   └── data_processing/
│   ├── pipeline/
│   │   ├── context_builder.py
│   │   ├── intent_detection.py         # rule-based + LLM stage 2
│   │   ├── safety_check.py             # pattern-based (v2 отложено)
│   │   ├── router.py                   # 4 маршрута
│   │   ├── tool_executor.py
│   │   ├── template_plan_executor.py
│   │   ├── planner.py                  # LLM tool-calls loop
│   │   ├── response_generator.py
│   │   ├── memory_update.py
│   │   └── orchestrator.py
│   ├── tools/                    # Tool executor tools (+ rag_retrieve)
│   ├── api/                      # Routes (chat WS, admin API)
│   ├── admin/                    # Jinja2 templates + views
│   └── static/                   # CSS/JS для chat и admin
├── scripts/
│   ├── seed_data.py              # Seed Generator v2 (параметризуемый)
│   └── seed_knowledge.py         # Knowledge Base демо-чанки
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
- **Язык идентификаторов:** английский
- **Type hints:** обязательно везде
- **Async:** весь I/O (DB, LLM, HTTP, ChromaDB где возможно) — async
- **Pydantic:** для всех schemas на границах API
- **Dataclasses или Pydantic:** для внутренних DTO (pipeline result, intent result, stage event)

### Стиль комментариев и docstrings

Применяется ко **всему новому коду** и к комментариям, которые ты добавляешь
рядом с правками. Старые комментарии на русском не переписываем массово —
только если рядом всё равно идёт правка.

- **Язык:** простой английский. Короткие фразы, обычные слова, без формального
  тона и без «корпоративных» оборотов.
- **Пиши как человек, а не генератор:** комментарий имеет право на жизнь только
  если в нём есть смысл, который не виден из кода.
- **Что писать:** неочевидное «почему», скрытые инварианты, обходы конкретных
  багов, тонкости поведения, которые удивят читателя.
- **Что НЕ писать:**
  - пересказ того, что и так видно из имён и сигнатуры (`# increment counter`,
    `# returns user`)
  - ссылки на текущую задачу / issue / автора (`# added for #42`, `# fix from PR`)
  - TODO без конкретики
  - многоабзацные docstrings там, где хватает одной строки
- **Docstrings:** одна короткая строка по делу. Развёрнутый блок — только если
  у функции реально нетривиальный контракт (edge cases, инварианты входа/выхода).
- **Маркеры заглушек v3:** оставляем как есть (`TODO v3`, `# stub: ...`) —
  по ним ищется работа.

Хороший пример:
```python
# Ollama drops the connection on long prompts; chunk to stay under ~8k tokens.
def chunk_prompt(text: str) -> list[str]:
    ...
```

Плохой пример:
```python
# This function chunks the prompt into smaller pieces.
# It takes a string and returns a list of strings.
def chunk_prompt(text: str) -> list[str]:
    ...
```

### Именование
- Модули pipeline: snake_case (`intent_detection.py`, `template_plan_executor.py`)
- Классы: PascalCase (`IntentDetector`, `PlannerAgent`, `LLMRegistry`)
- DTO: `{Name}Result` или `{Name}Event` (`PipelineResult`, `StageEvent`)

## Отложено на v3 (TODO v3)

Следующие подсистемы **намеренно не реализованы** и имеют заглушки/комментарии
в коде. При поступлении задачи из этого списка — смотри состояние текущей
заглушки, не ломай существующую функциональность.

- **Data ingestion из реального API** (+ Anomaly detection, Deduplication).
  Источник данных — только Seed Generator v2.
- **Safety Check v2** (контекстный LLM-анализ) — остаётся pattern-based.
  Комментарий-заглушка в `safety_check.py`.
- **Output Validation v2** (hallucination check, medical advice check) — skipped во всех
  маршрутах, место в pipeline зарезервировано.
- **Periodization** (макро/мезо/микроциклы) — заглушка-модуль с docstring `TODO v3`.
- **Proactive Alerts** (HRV drop, RHR spike, weekly summary) — не запускаются.
  Пустой модуль `app/services/alerts.py` с описанием (если отсутствует — создать при задаче v3).
- **Testing & Evaluation** (eval-датасеты, hallucination tests, latency benchmarks,
  RAG quality) — отдельно не ведём.

Knowledge Base (RAG) — сейчас **минимальный демо-набор** (20–40 чанков по 5 категориям).
Расширять через `scripts/seed_knowledge.py` или админ-страницу `/admin/knowledge`.

## Работа с Ollama и моделями

- Хост: `http://ollama:11434` (внутри сети `ollama-net`)
- **Multi-model через LLM Registry**. Роли:
  - `intent_llm` — классификация intent при low-confidence (дефолт: lightweight, `qwen3.5:9b`)
  - `safety_llm` — зарезервировано под v2, сейчас не вызывается
  - `response` — основной генератор ответов (дефолт: primary, `qwen3.5:9b`)
  - `planner` — сложное планирование + генерация планов (дефолт: heavy, `qwen3.5:9b`)
- Конфиг ролей — из `.env` + runtime overrides в SQLite (`llm_role_config`, меняется в админке).
- Если указанная модель недоступна в Ollama — fallback на `OLLAMA_MODEL` (базовая) с WARN в логах.
- Всегда логировать каждый LLM-вызов в `llm_calls`: role, model, длину промпта, длину ответа,
  duration_ms, request_id.
- Timeout: 60s (heavy — 120s), 1 retry при timeout.

## Pipeline flow

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

## Тесты

- Новые тесты добавляем **только под текущую задачу**: то, что нужно, чтобы
  убедиться в корректности своих правок и не сломать соседнее поведение.
  Не пишем тесты «впрок» и не расширяем покрытие без запроса.
- Если правка ломает существующий тест — чиним по делу: либо тест устарел и
  обновляется под новый контракт, либо это регрессия и чинить надо код.
- Фреймворк: `pytest` + `pytest-asyncio`.
- Запуск: `docker compose exec app pytest` (или конкретный файл/узел).

## Что делать при получении задачи

1. Прочитать issue / запрос полностью (acceptance criteria!).
2. Если задача относится к разделу «Отложено на v3» — сначала посмотреть, что
   уже есть в коде (заглушка / комментарий / пустой модуль), и не сломать
   существующий контракт.
3. Реализовать **минимально достаточно**, не добавлять лишнего.
4. Проверить:
   - `docker compose up --build` запускается без ошибок
   - `alembic upgrade head` проходит
   - Релевантные тесты проходят (`pytest`)
   - Чат `/chat` и админка `/admin` открываются
