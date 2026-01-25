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

# InfoBot требует Python 3.14 (fallback на 3.12)
check_python_314() {
  local python_cmd="$1"
  if ! command_exists "$python_cmd"; then
    return 1
  fi
  local v=$("$python_cmd" --version 2>&1)
  case "$v" in *"3.14"*) return 0 ;; *) return 1 ;; esac
}

check_python_312() {
  local python_cmd="$1"
  if ! command_exists "$python_cmd"; then
    return 1
  fi
  local v=$("$python_cmd" --version 2>&1)
  case "$v" in *"3.12"*) return 0 ;; *) return 1 ;; esac
}

# Проверка Python 3.14 (приоритет)
PYTHON_FOUND=0
PYTHON_CMD=""

if check_python_314 python3.14; then
  PYTHON_FOUND=1
  PYTHON_CMD="python3.14"
elif check_python_314 python3; then
  PYTHON_FOUND=1
  PYTHON_CMD="python3"
elif check_python_314 python; then
  PYTHON_FOUND=1
  PYTHON_CMD="python"
# Fallback на Python 3.12
elif check_python_312 python3.12; then
  PYTHON_FOUND=1
  PYTHON_CMD="python3.12"
elif check_python_312 python3; then
  PYTHON_FOUND=1
  PYTHON_CMD="python3"
elif check_python_312 python; then
  PYTHON_FOUND=1
  PYTHON_CMD="python"
fi

if [ $PYTHON_FOUND -eq 0 ]; then
  PKG_MGR=$(detect_package_manager)
  case "$PKG_MGR" in
    apt)
      echo "[INFO] Установка Python 3.14 через apt..."
      if sudo apt-get update -qq >/dev/null 2>&1 && \
         sudo apt-get install -y python3.14 python3.14-venv >/dev/null 2>&1; then
        if check_python_314 python3.14; then
          PYTHON_FOUND=1
          PYTHON_CMD="python3.14"
        fi
      fi
      ;;
    yum)
      echo "[INFO] Установка Python 3.14 через yum..."
      if sudo yum install -y python3.14 >/dev/null 2>&1; then
        check_python_314 python3.14 && PYTHON_FOUND=1 && PYTHON_CMD="python3.14"
      fi
      ;;
    dnf)
      echo "[INFO] Установка Python 3.14 через dnf..."
      if sudo dnf install -y python3.14 >/dev/null 2>&1; then
        check_python_314 python3.14 && PYTHON_FOUND=1 && PYTHON_CMD="python3.14"
      fi
      ;;
    pacman)
      echo "[INFO] Установка Python через pacman..."
      if sudo pacman -S --noconfirm python >/dev/null 2>&1; then
        check_python_312 python && PYTHON_FOUND=1 && PYTHON_CMD="python"
        check_python_312 python3 && PYTHON_FOUND=1 && PYTHON_CMD="python3"
      fi
      ;;
    brew)
      echo "[INFO] Установка Python 3.14 через brew..."
      if brew install python@3.14 >/dev/null 2>&1; then
        check_python_314 python3.14 && PYTHON_FOUND=1 && PYTHON_CMD="python3.14"
      fi
      ;;
  esac
  if [ $PYTHON_FOUND -eq 0 ]; then
    echo "[ERROR] Python 3.14 не найден. Установите: https://www.python.org/downloads/"
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

# Определение Python для запуска: .venv_gpu (3.12) > .venv > python3.12
if [[ -f ".venv_gpu/bin/activate" ]]; then
  source ".venv_gpu/bin/activate"
  PYTHON_BIN="python"
elif [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
  PYTHON_BIN="python"
else
  PYTHON_BIN="${PYTHON_CMD:-python3.12}"
fi
exec $PYTHON_BIN launcher/infobot_manager.py "$@"

