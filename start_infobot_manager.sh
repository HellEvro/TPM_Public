#!/usr/bin/env bash
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
if [[ -x ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
  PYTHON_BIN="python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
  echo "[WARN] Virtual environment not found. Falling back to ${PYTHON_BIN}."
fi
exec "$PYTHON_BIN" launcher/infobot_manager.py "$@"

