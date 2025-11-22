#!/usr/bin/env bash

set -euo pipefail

echo "=========================================="
echo " InfoBot Installer - macOS"
echo "=========================================="
echo

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3 not found. Install Python 3.9+ (e.g. via https://www.python.org/downloads/macos/) and re-run."
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "[WARN] git not found. Install Xcode Command Line Tools or Git to enable update checks."
fi

PYTHON_BIN=$(command -v python3)
PY_VERSION=$("$PYTHON_BIN" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
MAJOR_MINOR=$(echo "$PY_VERSION" | cut -d. -f1,2)

if [[ "$(printf '%s\n' "3.9" "$MAJOR_MINOR" | sort -V | head -n1)" != "3.9" ]]; then
  echo "[ERROR] Python $PY_VERSION detected. Python 3.9 or newer is required."
  exit 1
fi

VENV_DIR="${PROJECT_ROOT}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "[INFO] Creating virtual environment at .venv"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "${VENV_DIR}/bin/activate"

echo "[INFO] Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

echo "[INFO] Installing/updating project dependencies"
python -m pip install -r requirements.txt

LAUNCHER_SCRIPT="${PROJECT_ROOT}/start_infobot_manager.sh"
if [[ ! -f "$LAUNCHER_SCRIPT" ]]; then
cat <<'BASH' > "$LAUNCHER_SCRIPT"
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
BASH
  chmod +x "$LAUNCHER_SCRIPT"
fi

echo
echo "[SUCCESS] Installation complete!"
echo "Next steps:"
echo "  1. Run: ./start_infobot_manager.sh"
echo "  2. Use the GUI to launch app.py, bots.py and ai.py"
echo "  3. Check updates via the GUI (requires git)"
echo

