#!/usr/bin/env bash
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Функция проверки наличия команды
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Функция определения менеджера пакетов
detect_package_manager() {
  if command_exists apt-get; then
    echo "apt"
  elif command_exists yum; then
    echo "yum"
  elif command_exists dnf; then
    echo "dnf"
  elif command_exists pacman; then
    echo "pacman"
  elif command_exists brew; then
    echo "brew"
  else
    echo ""
  fi
}

# InfoBot требует Python 3.14 или выше
check_python_314_plus() {
  local python_cmd="$1"
  if ! command_exists "$python_cmd"; then
    return 1
  fi
  local v=$("$python_cmd" -c "import sys; exit(0 if sys.version_info[:2] >= (3, 14) else 1)" 2>/dev/null)
  return $v
}

# Проверка Python 3.14+
PYTHON_FOUND=0
PYTHON_CMD=""

if check_python_314_plus python3.14; then
  PYTHON_FOUND=1
  PYTHON_CMD="python3.14"
elif check_python_314_plus python3; then
  PYTHON_FOUND=1
  PYTHON_CMD="python3"
elif check_python_314_plus python; then
  PYTHON_FOUND=1
  PYTHON_CMD="python"
fi

if [ $PYTHON_FOUND -eq 0 ]; then
  PKG_MGR=$(detect_package_manager)
  case "$PKG_MGR" in
    apt)
      echo "[INFO] Установка Python 3.14.2+ через apt..."
      if sudo apt-get update -qq >/dev/null 2>&1 && \
         sudo apt-get install -y python3.14 python3.14-venv >/dev/null 2>&1; then
        if check_python_314_plus python3.14; then
          PYTHON_FOUND=1
          PYTHON_CMD="python3.14"
        fi
      fi
      ;;
    yum)
      echo "[INFO] Установка Python 3.14.2+ через yum..."
      if sudo yum install -y python3.14 >/dev/null 2>&1; then
        check_python_314_plus python3.14 && PYTHON_FOUND=1 && PYTHON_CMD="python3.14"
      fi
      ;;
    dnf)
      echo "[INFO] Установка Python 3.14.2+ через dnf..."
      if sudo dnf install -y python3.14 >/dev/null 2>&1; then
        check_python_314_plus python3.14 && PYTHON_FOUND=1 && PYTHON_CMD="python3.14"
      fi
      ;;
    pacman)
      echo "[INFO] Установка Python 3.14.2+ через pacman..."
      if sudo pacman -S --noconfirm python >/dev/null 2>&1; then
        check_python_314_plus python && PYTHON_FOUND=1 && PYTHON_CMD="python"
        check_python_314_plus python3 && PYTHON_FOUND=1 && PYTHON_CMD="python3"
      fi
      ;;
    brew)
      echo "[INFO] Установка Python 3.14.2+ через brew..."
      if brew install python@3.14 >/dev/null 2>&1; then
        check_python_314_plus python3.14 && PYTHON_FOUND=1 && PYTHON_CMD="python3.14"
      fi
      ;;
  esac
  if [ $PYTHON_FOUND -eq 0 ]; then
    echo "[ERROR] Python 3.14.2+ не найден. Установите: https://www.python.org/downloads/"
    exit 1
  fi
fi

# Проверка Git (только если Python установлен)
# Git не критичен для запуска, поэтому ошибки установки игнорируем
if ! command_exists git; then
  PKG_MGR=$(detect_package_manager)
  case "$PKG_MGR" in
    apt)
      echo "[INFO] Установка Git через apt..."
      sudo apt-get install -y git >/dev/null 2>&1 || true
      ;;
    yum)
      echo "[INFO] Установка Git через yum..."
      sudo yum install -y git >/dev/null 2>&1 || true
      ;;
    dnf)
      echo "[INFO] Установка Git через dnf..."
      sudo dnf install -y git >/dev/null 2>&1 || true
      ;;
    pacman)
      echo "[INFO] Установка Git через pacman..."
      sudo pacman -S --noconfirm git >/dev/null 2>&1 || true
      ;;
    brew)
      echo "[INFO] Установка Git через brew..."
      brew install git >/dev/null 2>&1 || true
      ;;
  esac
fi

# Безопасная инициализация Git репозитория (если Git установлен)
if command_exists git; then
  # Настраиваем Git пользователя (если не настроен) - ДО любых операций
  if ! git config user.name >/dev/null 2>&1; then
    git config user.name "InfoBot User" >/dev/null 2>&1
  fi
  if ! git config user.email >/dev/null 2>&1; then
    git config user.email "infobot@local" >/dev/null 2>&1
  fi
  
  # Инициализируем БЕЗ pull/fetch, чтобы не перезаписать существующие файлы
  if [ ! -d ".git" ]; then
    git init >/dev/null 2>&1
    git branch -m main >/dev/null 2>&1
    # Добавляем remote с HTTPS URL
    git remote add origin https://github.com/HellEvro/TPM_Public.git >/dev/null 2>&1
  else
    # Репозиторий уже существует - проверяем и исправляем remote URL
    if git remote get-url origin >/dev/null 2>&1; then
      # Remote существует - проверяем, используется ли SSH
      REMOTE_URL=$(git remote get-url origin 2>/dev/null)
      if echo "$REMOTE_URL" | grep -q "git@github.com"; then
        # Используется SSH - меняем на HTTPS
        git remote set-url origin https://github.com/HellEvro/TPM_Public.git >/dev/null 2>&1
      fi
    else
      # Remote не существует - добавляем
      git remote add origin https://github.com/HellEvro/TPM_Public.git >/dev/null 2>&1
    fi
  fi
  
  # Делаем первый коммит, если нет коммитов (независимо от того, новый репозиторий или существующий)
  if ! git rev-list --count HEAD >/dev/null 2>&1; then
    # Нет коммитов - делаем первый коммит
    # Добавляем все файлы
    git add -A >/dev/null 2>&1
    # Делаем коммит
    git commit -m "Initial commit: InfoBot Public repository" >/dev/null 2>&1
  fi
fi

# Проверка и обновление .venv для Python 3.14
if [[ -f "scripts/ensure_python314_venv.py" ]]; then
  echo "[INFO] Проверка и обновление .venv для Python 3.14..."
  ${PYTHON_CMD:-python3} scripts/ensure_python314_venv.py >/dev/null 2>&1
fi

# Определение Python для запуска: .venv > глобальный Python 3.14+
# Если venv нет, используем глобальный Python
if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
  PYTHON_BIN="python"
  echo "[INFO] Используется .venv"
else
  # Используем глобальный Python (venv не требуется)
  if [ -n "$PYTHON_CMD" ]; then
    PYTHON_BIN="$PYTHON_CMD"
    echo "[INFO] Используется глобальный Python (venv не найден)"
  else
    # Пробуем найти любой доступный Python
    if command_exists python3; then
      PYTHON_BIN="python3"
      echo "[INFO] Используется глобальный Python3"
    elif command_exists python; then
      PYTHON_BIN="python"
      echo "[INFO] Используется глобальный Python"
    else
      echo "[ERROR] Python не найден!"
      exit 1
    fi
  fi
fi

exec $PYTHON_BIN launcher/infobot_manager.py "$@"

