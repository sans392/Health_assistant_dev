# Health Assistant — MVP Implementation Plan

## Strategy: Minimum Viable Pipeline

Цель — получить **работающий end-to-end чат-ассистент** за минимальное количество шагов.
Все сложные модули (RAG, multi-model, semantic memory) откладываются на следующие итерации.

## MVP Scope

### IN (входит в MVP)
- Docker Compose + Ollama через `ollama-net`
- FastAPI backend + SQLite
- **Одна модель** (qwen2.5:7b) для всех задач
- Rule-based Intent Detection (regex/keyword)
- Pattern-based Safety Check
- Простой Router (fast_path + standard)
- Tool Executor — 6 базовых tools для запросов к БД
- Data Processing — activity_summary, training_load, trend_analyzer
- Response Generator с prompt template
- Seed data (30 дней тестовых данных)
- Per-request logging в SQLite
- Test Chat (WebSocket + HTML)
- Admin Panel (Jinja2 + HTMX)

### OUT (отложено на v2+)
- RAG / ChromaDB / Knowledge Base
- Multi-model routing (7b / 14b / 32b)
- Semantic memory (vector DB)
- Data ingestion из реального API
- Output Validation (hallucination check)
- Anomaly detection / flagging
- Deduplication logic
- Recovery Score (расчётный), Strain Score, Overtraining detection
- Proactive alerts
- Periodization, Sleep analysis
- Streaming ответов

## Issues — Порядок выполнения

### Phase 0: Infrastructure (фундамент)
| # | Issue | Зависимости |
|---|-------|-------------|
| 1 | [Infra] Project scaffolding + Docker Compose | — |
| 2 | [Data] SQLite database, models, seed data | #1 |
| 3 | [Infra] Ollama LLM client service | #1 |

### Phase 1: Pipeline (ядро логики)
| # | Issue | Зависимости |
|---|-------|-------------|
| 4 | [Pipeline] Intent Detection — rule-based | #1 |
| 5 | [Pipeline] Safety Check — pattern-based | #1 |
| 6 | [Pipeline] Context Builder | #2 |
| 7 | [Pipeline] Router | #4, #5 |
| 8 | [Pipeline] Tool Executor + DB tools | #2 |
| 9 | [Pipeline] Data Processing modules | #2, #8 |
| 10 | [Pipeline] Response Generator | #3, #6 |

### Phase 2: Integration + UI
| # | Issue | Зависимости |
|---|-------|-------------|
| 11 | [Pipeline] Orchestrator — end-to-end | #4-#10 |
| 12 | [Infra] Logging system | #2, #11 |
| 13 | [UI] Test Chat — WebSocket | #11 |
| 14 | [UI] Admin Panel | #2, #3, #12 |

## Dependency Graph

```
#1 Project Scaffolding
├── #2 Database + Models + Seed Data
│   ├── #6 Context Builder
│   ├── #8 Tool Executor
│   │   └── #9 Data Processing
│   └── #12 Logging System
├── #3 Ollama Client
│   └── #10 Response Generator
├── #4 Intent Detection
│   └── #7 Router
└── #5 Safety Check
        └── #7 Router

#4-#10 (all pipeline modules)
└── #11 Pipeline Orchestrator
    ├── #12 Logging System
    ├── #13 Test Chat
    └── #14 Admin Panel
```

## Optimal Execution Order (for Claude)

Порядок, минимизирующий переключение контекста:

1. **#1** — scaffolding, Docker Compose, config
2. **#2** — models, migrations, seed data
3. **#3** — Ollama client
4. **#4 + #5** — Intent Detection + Safety Check (параллельно, независимы)
5. **#6** — Context Builder
6. **#7** — Router
7. **#8** — Tool Executor
8. **#9** — Data Processing
9. **#10** — Response Generator
10. **#11** — Pipeline Orchestrator (склейка всего)
11. **#12** — Logging
12. **#13** — Test Chat
13. **#14** — Admin Panel

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI (Python 3.11+) |
| Database | SQLite + SQLAlchemy + Alembic |
| LLM | Ollama (qwen2.5:7b) via httpx |
| Chat UI | HTML + CSS + JS + WebSocket |
| Admin UI | Jinja2 + HTMX + Pico CSS |
| Container | Docker Compose |
| Network | External `ollama-net` |
