# Health Assistant — Phase 2 Implementation Plan

**Контекст:** MVP (issues #1–#14) закрыт. Этот документ — план второго этапа работы
по архитектуре из `health_assistant_architecture_v2.yaml`.

Работает **агент Claude (Sonnet 4.6)** на ветках `claude/<short-slug>` под каждую
issue. Каждая issue должна быть реализована минимально достаточно в рамках
своих acceptance criteria, не добавляя лишнего.

---

## Стратегия этапа

Цель Phase 2 — **приблизить пайплайн к архитектуре v2**:
мульти-модельный роутинг, RAG (демо-набор), semantic memory, полноценный
planner-route + template-планы, расширенный Data Processing, улучшенная
observability (логи LLM-вызовов + stage events) и модернизированный UI.

### Ключевые архитектурные решения этапа

| Пункт | Решение |
|---|---|
| Multi-model LLM | **Полный роутинг по ролям**: `intent_llm`, `safety_llm`, `response`, `planner`. Модель на каждую роль выбирается в админке из списка установленных в Ollama. Дефолтно — одна и та же модель для всех ролей (если 14b/32b не установлены). |
| Стриминг чата | **Stage events + token streaming** по WebSocket. UI показывает текущую стадию пайплайна и печатает ответ LLM по токенам. |
| Route=planner | **Базовая версия**: LLM в цикле (max 3–5 итераций) запрашивает tool-calls через JSON function-calling в промпте. Heavy-модель. |
| Semantic memory | **Минимальная реализация v1**: ChromaDB-коллекция `semantic_memory`, сохранение Q/A эмбеддингов после запроса, retrieval top-k в Context Builder. Без TTL/категоризации. |
| Knowledge Base (RAG) | **Минимальный демо-набор** (~20–40 чанков по всем 5 категориям YAML), ChromaDB-коллекция `knowledge_base`, retrieval для `plan_request` / `health_concern` / `data_analysis`. Комментарии в коде/README — как расширять. |
| Data ingestion из API | **Откладывается** (включая anomaly detection и deduplication). Вместо этого — продвинутый генератор тестовых данных `seed_data.py` с управлением из админки. |

### Откладывается на будущие итерации (TODO v3+)

С пометкой в коде и CLAUDE.md — чтобы не забыть:

- **Data ingestion из реального API** (+ Anomaly detection, Deduplication)
- **Safety Check v2** (контекстный LLM-анализ, снижение false-positives)
- **Output Validation v2** (hallucination check, medical advice check, consistency)
- **Periodization** (макро/мезо/микроциклы, фазы подготовки)
- **Proactive Alerts** (HRV drop, RHR spike, inactivity, weekly summary)
- **Testing & Evaluation** (eval-датасеты, hallucination tests, latency benchmarks, RAG quality)

---

## Issues — порядок и зависимости

Issues пронумерованы продолжая MVP (#1–#14) → начинаем с **#15**.

### Phase A — Foundation (инфраструктура)

| # | Issue | Что даёт | Зависимости |
|---|-------|----------|-------------|
| 15 | **[Infra] LLM Registry + Multi-model routing** | Новый `app/services/llm_registry.py`. Роли `intent_llm`, `safety_llm`, `response`, `planner`. Конфиг моделей из `.env` + runtime overrides в SQLite (`llm_role_config`). Backward-compat: `ollama_client` делегирует в registry по роли `response`. | — |
| 16 | **[Infra] Embeddings & ChromaDB** | Зависимости (`chromadb`, `sentence-transformers` или вызов эмбеддингов через Ollama API `/api/embeddings`). Новый `app/services/embedding_service.py`, `app/services/vector_store.py`. Две коллекции: `knowledge_base`, `semantic_memory`. Docker-compose: volume для chroma persistent. | — |
| 17 | **[Data] Schema v2 + migration** | Alembic migration: <br>• `pipeline_logs`: добавить `rag_chunks_used`, `stage_trace` (JSON), `llm_role_usage` (JSON)<br>• новая таблица `llm_calls` (per-request prompt/response/model/role/duration)<br>• новая таблица `llm_role_config` (role, model, updated_at)<br>• новая таблица `rag_chunks` (id, text, category, source, confidence, sport_type, experience_level, embedding_id)<br>• `daily_facts`: колонка `recovery_score_calculated`, `strain_score`, `anomaly_flags` (JSON)<br>• `activities`: `anomaly_flags` (JSON) — заполняется в seed | #2 (уже есть) |

### Phase B — Knowledge Base & Memory

| # | Issue | Что даёт | Зависимости |
|---|-------|----------|-------------|
| 18 | **[RAG] Knowledge Base — демо-набор + retrieval** | `scripts/seed_knowledge.py`: ~20–40 заранее подготовленных чанков по категориям (physiology_norms, training_principles, recovery_science, sport_specific, nutrition_basics). Загрузка в ChromaDB при старте/команде. Новый tool `rag_retrieve(query, category?, sport_type?, top_k=5)`. README/комментарий: как добавлять источники и расширять. | #16, #17 |
| 19 | **[Memory] Semantic Memory v1** | Сохранение эмбеддингов `(query, response)` после каждого запроса. Retrieval top-k в Context Builder для обогащения контекста. Комментарии TODO v2 (TTL, категоризация). | #16, #17 |

### Phase C — Pipeline Upgrade

| # | Issue | Что даёт | Зависимости |
|---|-------|----------|-------------|
| 20 | **[Pipeline] Intent Detection v2** | Rule-based остаётся как stage 1. Если confidence < 0.85 — stage 2 через LLM (`intent_llm`). Промпт содержит список intents, последние 3 сообщения истории. Улучшенный entity extraction: body_part, intensity, больше metrics. | #15 |
| 21 | **[Pipeline] Router v2 + Template Plan Executor** | Router расширен 4 маршрутами: `fast_direct_answer`, `tool_simple`, `template_plan`, `planner`. Отдельный `template_plan_executor.py` с шаблонами: `weekly_training_plan`, `recovery_report`, `overtraining_check`, `progress_report`. Каждый шаблон — фиксированная последовательность tool-calls и data-processing модулей. | #8, #9, #18 |
| 22 | **[Pipeline] Data Processing v2** | Новые модули в `app/services/data_processing/`:<br>• `recovery_score.py` — расчётный RS с fallback на Whoop<br>• `strain_score.py` — аналог Whoop Strain<br>• `heart_rate_zones.py` — расчёт зон (Karvonen)<br>• `overtraining_detection.py` — markers + recommendation.<br>Расширить `training_load.py` (monotony, strain, acute-chronic ratio). | #8, #9 |
| 23 | **[Pipeline] Planner Route** | `app/pipeline/planner.py`: LLM-driven loop. Промпт-инструкция для JSON tool-calls. `ToolExecutor` в цикле. `max_iterations=5`, `max_tool_calls_per_iter=3`, `timeout=60s`. Heavy-модель. Логирование каждой итерации в `llm_calls`. | #15, #8, #22 |
| 24 | **[Pipeline] Response Generator v2** | Выбор модели по intent: `plan_request`→planner role, остальное→response role. RAG context в промпте для `plan_request`/`health_concern`/`data_analysis`. Улучшенные правила промпта (см. YAML: упоминать anomaly flags, ссылаться на конкретные данные). Token streaming через Ollama API `stream: true`, коллбек для stage events. | #15, #18 |

### Phase D — Observability

| # | Issue | Что даёт | Зависимости |
|---|-------|----------|-------------|
| 25 | **[Infra] Logging v2 + Stage Events** | Каждый LLM-вызов → запись в `llm_calls` (request_id, role, model, prompt, response, duration_ms, tokens). `stage_trace` в PipelineLog — хронология стадий с duration. Pub/sub механизм для передачи стадий в WebSocket (обратный коллбек из orchestrator в chat endpoint). | #17 |
| 26 | **[Pipeline] Memory Update** | Асинхронное обновление памяти: short_term (history session), long_term (profile fact extraction из диалога — rule-based), semantic (сохранение эмбеддинга Q/A). Не блокирует delivery ответа. | #19 |

### Phase E — Seed Data Generator v2

| # | Issue | Что даёт | Зависимости |
|---|-------|----------|-------------|
| 27 | **[Data] Seed Generator v2 + Admin UI** | Рефакторинг `scripts/seed_data.py` в класс `SeedGenerator` с параметрами: `days`, `user_count`, `profile_preset` (beginner/intermediate/advanced), `add_anomalies`, `missing_data_rate`, `overreaching_scenario`. Отдельные preset-сценарии: "нормальная нагрузка", "перетренированность", "восстановление", "травма". Админ-страница `/admin/seed`: форма параметров, truncate DB, preview генерируемых записей. | #17 |

### Phase F — Admin Panel v2

| # | Issue | Что даёт | Зависимости |
|---|-------|----------|-------------|
| 28 | **[UI] Admin Logs v2 + Profile Editor fix** | Лог-детали: вкладки `Stage Trace`, `LLM Calls` (таблица промптов/ответов с раскрытием), `Tool Results`, `RAG Chunks used`. Список логов: колонка `LLM calls count`, фильтр по роли модели. **Fix**: кнопка "Редактировать профиль" должна работать — починить баг (кнопка открывает модал, но save падает/не все поля). Расширить редактор на все поля (injuries, chronic_conditions, preferred_sports, resting_heart_rate, gender). | #25 |
| 29 | **[UI] Admin: LLM Config + KB browser + Semantic Memory + Diagnostics** | Новые страницы:<br>• `/admin/llm` — per-role model selection + сохранение в `llm_role_config` + test prompt per role<br>• `/admin/knowledge` — browse/add/delete/reindex чанков KB<br>• `/admin/memory` — просмотр записей semantic_memory + clear<br>• `/admin/diagnostics` — health всех сервисов (Ollama per role, ChromaDB, DB), runtime метрики, benchmark tool (запуск N запросов для замеров). | #15, #16, #18, #19 |

### Phase G — Test Chat v2

| # | Issue | Что даёт | Зависимости |
|---|-------|----------|-------------|
| 30 | **[UI] Test Chat v2 — streaming + presets + debug** | WebSocket-сообщения: `stage_start`, `stage_end` (с duration), `token` (стриминг LLM), `message` (финал). В UI — индикатор этапа (progress bar или текст "🔍 Определяю намерение..." → "📊 Запрашиваю данные..." → "🧠 Генерирую ответ..."). Выпадающий список заготовленных вопросов (10–15 штук, покрывающих intents и edge-cases). Debug-панель v2: полный trace (stages + timings), intent+confidence, route, tools, RAG chunks, LLM calls per role, errors. | #25, #24 |

### Phase H — Integration

| # | Issue | Что даёт | Зависимости |
|---|-------|----------|-------------|
| 31 | **[Pipeline] Orchestrator v2 + integration tests** | Переписать `orchestrator.py` под новые компоненты (LLM Registry, Semantic Memory, RAG, Planner, Template Executor, Stage Events). Минимальные интеграционные тесты (pytest + mock Ollama/Chroma): fast_path, tool_simple, template_plan, planner loop, safety block. | #18–#26 |

---

## Граф зависимостей

```
#15 LLM Registry ──────┬──► #20 Intent v2
                       ├──► #23 Planner
                       ├──► #24 Response Gen v2
                       └──► #29 Admin LLM Config

#16 Embeddings + Chroma ─┬──► #18 Knowledge Base (RAG)
                         └──► #19 Semantic Memory

#17 Schema v2 ─────────┬──► #18, #19
                       ├──► #25 Logging v2
                       └──► #27 Seed v2

#8, #9 (MVP) ──────────┬──► #21 Router + Template Plan
                       ├──► #22 Data Processing v2
                       └──► #23 Planner

#18 KB + #19 SemMem ───┬──► #24 Response Gen v2
                       └──► #29 Admin

#25 Logging v2 ────────┬──► #28 Admin Logs v2
                       └──► #30 Test Chat v2

#15..#26 ──────────────► #31 Orchestrator v2
```

---

## Оптимальный порядок исполнения (для Claude)

Порядок минимизирует переключение контекста и риск блокировок по зависимостям:

1. **#15** — LLM Registry (фундамент multi-model)
2. **#17** — Schema v2 (миграции нужны почти везде)
3. **#16** — ChromaDB + Embeddings (инфраструктура для RAG и SemMem)
4. **#18** — Knowledge Base демо-набор + retrieval
5. **#19** — Semantic Memory v1
6. **#20** — Intent Detection v2
7. **#22** — Data Processing v2 (независим от LLM-частей)
8. **#21** — Router v2 + Template Plan Executor
9. **#23** — Planner Route
10. **#24** — Response Generator v2 + token streaming
11. **#25** — Logging v2 + Stage Events
12. **#26** — Memory Update
13. **#27** — Seed Generator v2 + Admin UI
14. **#28** — Admin Logs v2 + Profile Editor fix
15. **#29** — Admin: LLM Config + KB + SemMem + Diagnostics
16. **#30** — Test Chat v2
17. **#31** — Orchestrator v2 + integration tests

---

## Что остаётся за рамками этапа

Отдельными заглушками в коде / разделами в CLAUDE.md:

- **Data ingestion из API** — место в архитектуре (`app/services/data_ingestion/`)
  подготовлено, но реализуется позже. Сейчас источник — Seed Generator.
- **Safety Check v2** — остаётся pattern-based из MVP + комментарий в safety_check.py.
- **Output Validation v2** — skipped во всех маршрутах, место в pipeline зарезервировано.
- **Periodization** — заглушка-модуль `data_processing/periodization.py` с docstring "TODO v3".
- **Proactive Alerts** — не запускаются. Подготовить пустой модуль `app/services/alerts.py` с описанием.
- **Testing & Evaluation** — unit-тесты критичных модулей есть, но eval-датасеты
  и бенчмарки — следующий этап.

---

## Tech Stack изменения

Добавляется к MVP-стеку:

| Компонент | Технология |
|-----------|-----------|
| Vector DB | ChromaDB (embedded, persistent-volume) |
| Embeddings | `nomic-embed-text` через Ollama `/api/embeddings` (локально, без интернета) |
| LLM Registry | собственный сервис, конфиг ролей в SQLite + `.env` overrides |
| Stage Events | WebSocket (расширение существующего `/ws/chat/{session_id}`) |

`requirements.txt` прирастает: `chromadb`, опционально — ничего больше
(эмбеддинги берём из Ollama — та же `ollama-net`).
