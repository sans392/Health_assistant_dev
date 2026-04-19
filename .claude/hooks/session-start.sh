#!/bin/bash
# SessionStart hook для Claude Code on the web.
# Ставит зависимости из requirements.txt в system Python удалённого контейнера,
# чтобы pytest и импорты приложения работали из коробки в каждой web-сессии.
# Локально (CLI / docker compose) — no-op.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(pwd)}"

if [ ! -f requirements.txt ]; then
  echo "session-start: requirements.txt не найден — пропускаю" >&2
  exit 0
fi

echo "session-start: устанавливаю зависимости из requirements.txt"
python3 -m pip install -q -r requirements.txt --break-system-packages

echo "session-start: готово"
