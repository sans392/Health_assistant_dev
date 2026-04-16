# Health Assistant — Phase 2 Implementation Plan

**Контекст:** MVP (issues #1–#14) закрыт. Этот документ — план второго этапа работы
по архитектуре из `health_assistant_architecture_v2.yaml`.

Работает **агент Claude (Sonnet 4.6)** на ветках `claude/<short-slug>` под каждую
issue. Каждая issue должна быть реализована минимально достаточно в рамках
своих acceptance criteria, не добавляя лишнего.

> **Нумерация issues в GitHub:** номера 15–20 оказались заняты прошлыми PR,
> поэтому фактические issues Phase 2 имеют номера **#21–#37**. Ниже
> в таблицах и графе зависимостей используются фактические GitHub-номера.

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

### Phase A — Foundation (инфраструктура)

| # | Issue | Что даёт | Зависимости |
|---|-------|----------|-------------|
| **#21** | **[Infra] LLM Registry + Multi-model routing** | Новый `app/services/llm_registry.py`. Роли `intent_llm`, `safety_llm`, `response`, `planner`. Конфиг моделей из `.env` + runtime overrides в SQLite (`llm_role_config`). Backward-compat: `ollama_client` делегирует в registry по роли `response`. | — |
| **#22** | **[Infra] Embeddings & ChromaDB** | Зависимости (`chromadb`, эмбеддинги через Ollama `/api/embeddings`). Новый `app/services/embedding_service.py`, `app/services/vector_store.py`. Две коллекции: `knowledge_base`, `semantic_memory`. Docker-compose: volume для chroma persistent. | — |
| **#23** | **[Data] Schema v2 + migration** | Alembic migration: <br>• `pipeline_logs`: добавить `rag_chunks_used`, `stage_trace` (JSON), `llm_role_usage` (JSON)<br>• новая таблица `llm_calls` (per-request prompt/response/model/role/duration)<br>• новая таблица `llm_role_config` (role, model, updated_at)<br>• новая таблица `rag_chunks` (id, text, category, source, confidence, sport_type, experience_level, embedding_id)<br>• `daily_facts`: колонка `recovery_score_calculated`, `strain_score`, `anomaly_flags` (JSON)<br>• `activities`: `anomaly_flags` (JSON) — заполняется в seed | #2 (MVP) |

### Phase B — Knowledge Base & Memory

| # | Issue | Что даёт | Зависимости |
|---|-------|----------|-------------|
| **#24** | **[RAG] Knowledge Base — демо-набор + retrieval** | `scripts/seed_knowledge.py`: ~20–40 заранее подготовленных чанков по категориям (physiology_norms, training_principles, recovery_science, sport_specific, nutrition_basics). Загрузка в ChromaDB. Tool `rag_retrieve(query, category?, sport_type?, top_k=5)`. README/комментарий: как добавлять источники. | #22, #23 |
| **#25** | **[Memory] Semantic Memory v1** | Сохранение эмбеддингов `(query, response)` после каждого запроса. Retrieval top-k в Context Builder для обогащения контекста. TODO v2 (TTL, категоризация). | #22, #23 |

### Phase C — Pipeline Upgrade

| # | Issue | Что даёт | Зависимости |
|---|-------|----------|-------------|
| **#26** | **[Pipeline] Intent Detection v2** | Rule-based остаётся как stage 1. Если confidence < 0.85 — stage 2 через LLM (`intent_llm`). Промпт содержит список intents, последние 3 сообщения истории. Улучшенный entity extraction: body_part, intensity, больше metrics. | #21 |
| **#27** | **[Pipeline] Data Processing v2** | Новые модули в `app/services/data_processing/`:<br>• `recovery_score.py` — расчётный RS с fallback на Whoop<br>• `strain_score.py` — аналог Whoop Strain<br>• `heart_rate_zones.py` — расчёт зон (Karvonen)<br>• `overtraining_detection.py` — markers + recommendation.<br>Расширить `training_load.py` (monotony, strain, acute-chronic ratio). | #8, #9 (MVP) |
| **#28** | **[Pipeline] Router v2 + Template Plan Executor** | Router расширен 4 маршрутами: `fast_direct_answer`, `tool_simple`, `template_plan`, `planner`. Отдельный `template_plan_executor.py` с шаблонами: `weekly_training_plan`, `recovery_report`, `overtraining_check`, `progress_report`. | #8, #9, #24, #27 |
| **#29** | **[Pipeline] Planner Route** | `app/pipeline/planner.py`: LLM-driven loop. Промпт-инструкция для JSON tool-calls. `ToolExecutor` в цикле. `max_iterations=5`, `max_tool_calls_per_iter=3`, `timeout=60s`. Heavy-модель. Логирование каждой итерации в `llm_calls`. | #21, #8, #27 |
| **#30** | **[Pipeline] Response Generator v2** | Выбор модели по intent: `plan_request`→planner role, остальное→response role. RAG context в промпте для `plan_request`/`health_concern`/`data_analysis`. Улучшенные правила промпта. Token streaming через Ollama API `stream: true`, коллбек для stage events. | #21, #24 |

### Phase D — Observability

| # | Issue | Что даёт | Зависимости |
|---|-------|----------|-------------|
| **#31** | **[Infra] Logging v2 + Stage Events** | Каждый LLM-вызов → запись в `llm_calls` (request_id, role, model, prompt, response, duration_ms, tokens). `stage_trace` в PipelineLog — хронология стадий с duration. Pub/sub механизм `StageEventBus` для передачи стадий в WebSocket. | #23 |
| **#32** | **[Pipeline] Memory Update** | Асинхронное обновление памяти: short_term (history session), long_term (profile fact extraction из диалога — rule-based), semantic (сохранение эмбеддинга Q/A). Не блокирует delivery. | #25, #23 |

### Phase E — Seed Data Generator v2

| # | Issue | Что даёт | Зависимости |
|---|-------|----------|-------------|
| **#33** | **[Data] Seed Generator v2 + Admin UI** | Рефакторинг `scripts/seed_data.py` в класс `SeedGenerator` с параметрами: `days`, `user_count`, `profile_preset`, `add_anomalies`, `missing_data_rate`, `overreaching_scenario`. Отдельные preset-сценарии: "нормальная нагрузка", "перетренированность", "восстановление", "травма". Админ-страница `/admin/seed`. | #23 |

### Phase F — Admin Panel v2

| # | Issue | Что даёт | Зависимости |
|---|-------|----------|-------------|
| **#34** | **[UI] Admin Logs v2 + Profile Editor fix** | Лог-детали: вкладки `Stage Trace`, `LLM Calls`, `Tool Results`, `RAG Chunks used`. Список: колонка `LLM calls count`, фильтр по роли модели. **Fix**: кнопка "Редактировать профиль" — починить баг. Расширить редактор на все поля. | #31 |
| **#35** | **[UI] Admin: LLM Config + KB browser + Semantic Memory + Diagnostics** | Новые страницы:<br>• `/admin/llm` — per-role model selection + test prompt per role<br>• `/admin/knowledge` — browse/add/delete/reindex чанков KB<br>• `/admin/memory` — просмотр semantic_memory + clear<br>• `/admin/diagnostics` — health сервисов, runtime метрики, benchmark. | #21, #22, #24, #25 |

### Phase G — Test Chat v2

| # | Issue | Что даёт | Зависимости |
|---|-------|----------|-------------|
| **#36** | **[UI] Test Chat v2 — streaming + presets + debug** | WebSocket-сообщения: `stage_start`, `stage_end`, `token`, `done`. UI — индикатор этапа. Dropdown заготовленных вопросов (10–15). Debug-панель v2: полный trace + timings + intent + route + tools + RAG + LLM calls per role. | #31, #30 |

### Phase H — Integration

| # | Issue | Что даёт | Зависимости |
|---|-------|----------|-------------|
| **#37** | **[Pipeline] Orchestrator v2 + integration tests** | Переписать `orchestrator.py` под новые компоненты. Минимальные интеграционные тесты (pytest + mock Ollama/Chroma): fast_path, tool_simple, template_plan, planner loop, safety block. | #24–#32 |

---

## Граф зависимостей

```
#21 LLM Registry ──────┬──► #26 Intent v2
                       ├──► #29 Planner
                       ├──► #30 Response Gen v2
                       └──► #35 Admin LLM Config

#22 Embeddings + Chroma ─┬──► #24 Knowledge Base (RAG)
                         └──► #25 Semantic Memory

#23 Schema v2 ─────────┬──► #24, #25
                       ├──► #31 Logging v2
                       └──► #33 Seed v2

#8, #9 (MVP) ──────────┬──► #28 Router + Template Plan
                       ├──► #27 Data Processing v2
                       └──► #29 Planner

#24 KB + #25 SemMem ───┬──► #30 Response Gen v2
                       └──► #35 Admin

#31 Logging v2 ────────┬──► #34 Admin Logs v2
                       └──► #36 Test Chat v2

#24..#32 ──────────────► #37 Orchestrator v2
```

---

## Оптимальный порядок исполнения (для Claude)

Порядок минимизирует переключение контекста и риск блокировок по зависимостям:

1. **#21** — LLM Registry (фундамент multi-model)
2. **#23** — Schema v2 (миграции нужны почти везде)
3. **#22** — ChromaDB + Embeddings (инфраструктура для RAG и SemMem)
4. **#24** — Knowledge Base демо-набор + retrieval
5. **#25** — Semantic Memory v1
6. **#26** — Intent Detection v2
7. **#27** — Data Processing v2 (независим от LLM-частей)
8. **#28** — Router v2 + Template Plan Executor
9. **#29** — Planner Route
10. **#30** — Response Generator v2 + token streaming
11. **#31** — Logging v2 + Stage Events
12. **#32** — Memory Update
13. **#33** — Seed Generator v2 + Admin UI
14. **#34** — Admin Logs v2 + Profile Editor fix
15. **#35** — Admin: LLM Config + KB + SemMem + Diagnostics
16. **#36** — Test Chat v2
17. **#37** — Orchestrator v2 + integration tests

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
