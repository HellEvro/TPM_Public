#!/usr/bin/env bash

set -euo pipefail

echo "=========================================="
echo " InfoBot Installer - macOS"
echo "=========================================="
echo

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN=""
for c in python3.14 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    v=$("$c" -c 'import sys; print(sys.version_info.major, sys.version_info.minor)' 2>/dev/null) || continue
    if [[ "$v" == "3 14" ]] || [[ "$v" =~ ^3\ (1[4-9]|[2-9][0-9])$ ]]; then
      PYTHON_BIN=$(command -v "$c")
      break
    fi
  fi
done
if [[ -z "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python 3.14+ required. Install: https://www.python.org/downloads/"
  echo "       Or: brew install python@3.14  (macOS)"
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "[WARN] git not found. Install Xcode Command Line Tools or Git to enable update checks."
fi

VENV_DIR="${PROJECT_ROOT}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "[INFO] Creating virtual environment at .venv"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "${VENV_DIR}/bin/activate"

echo "[INFO] Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel --no-warn-script-location

echo "[INFO] Installing/updating project dependencies"
python -m pip install -r requirements.txt --no-warn-script-location

echo "[INFO] Compiling protected modules (.pyc files)"
if python license_generator/compile_all.py; then
    echo "[OK] All protected modules compiled successfully!"
else
    echo "[WARN] Some modules failed to compile. This may be normal if source files are not available."
fi

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

