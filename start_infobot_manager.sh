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

# Функция проверки версии Python (должна быть >= 3.13)
check_python_version() {
  local python_cmd="$1"
  if ! command_exists "$python_cmd"; then
    return 1
  fi
  
  local version_output
  version_output=$("$python_cmd" --version 2>&1)
  if [ $? -ne 0 ]; then
    return 1
  fi
  
  # Парсим версию (формат: Python 3.13.0)
  local version_str
  version_str=$(echo "$version_output" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
  if [ -z "$version_str" ]; then
    return 1
  fi
  
  local major minor
  major=$(echo "$version_str" | cut -d. -f1)
  minor=$(echo "$version_str" | cut -d. -f2)
  
  # Проверяем версию >= 3.13
  if [ "$major" -gt 3 ] || ([ "$major" -eq 3 ] && [ "$minor" -ge 13 ]); then
    return 0
  fi
  
  return 1
}

# Проверка Python
PYTHON_FOUND=0
PYTHON_CMD=""

if check_python_version python3; then
  PYTHON_FOUND=1
  PYTHON_CMD="python3"
elif check_python_version python; then
  PYTHON_FOUND=1
  PYTHON_CMD="python"
fi

# Если Python не найден или версия < 3.13 - пытаемся установить
if [ $PYTHON_FOUND -eq 0 ]; then
  PKG_MGR=$(detect_package_manager)
  case "$PKG_MGR" in
    apt)
      echo "[INFO] Установка Python 3.13+ через apt..."
      # Для Ubuntu/Debian нужно добавить deadsnakes PPA для Python 3.13
      if sudo apt-get update -qq >/dev/null 2>&1; then
        # Пробуем установить python3.13 напрямую или последнюю доступную версию
        if sudo apt-get install -y python3.13 python3.13-venv >/dev/null 2>&1 || \
           sudo apt-get install -y python3 python3-venv >/dev/null 2>&1; then
          if check_python_version python3.13; then
            PYTHON_FOUND=1
            PYTHON_CMD="python3.13"
          elif check_python_version python3; then
            PYTHON_FOUND=1
            PYTHON_CMD="python3"
          fi
        fi
      fi
      ;;
    yum)
      echo "[INFO] Установка Python 3.13+ через yum..."
      # Для CentOS/RHEL может потребоваться дополнительный репозиторий
      if sudo yum install -y python3.13 >/dev/null 2>&1 || \
         sudo yum install -y python3 >/dev/null 2>&1; then
        if check_python_version python3.13; then
          PYTHON_FOUND=1
          PYTHON_CMD="python3.13"
        elif check_python_version python3; then
          PYTHON_FOUND=1
          PYTHON_CMD="python3"
        fi
      fi
      ;;
    dnf)
      echo "[INFO] Установка Python 3.13+ через dnf..."
      if sudo dnf install -y python3.13 >/dev/null 2>&1 || \
         sudo dnf install -y python3 >/dev/null 2>&1; then
        if check_python_version python3.13; then
          PYTHON_FOUND=1
          PYTHON_CMD="python3.13"
        elif check_python_version python3; then
          PYTHON_FOUND=1
          PYTHON_CMD="python3"
        fi
      fi
      ;;
    pacman)
      echo "[INFO] Установка Python 3.13+ через pacman..."
      if sudo pacman -S --noconfirm python >/dev/null 2>&1; then
        if check_python_version python; then
          PYTHON_FOUND=1
          PYTHON_CMD="python"
        elif check_python_version python3; then
          PYTHON_FOUND=1
          PYTHON_CMD="python3"
        fi
      fi
      ;;
    brew)
      echo "[INFO] Установка Python 3.13+ через brew..."
      if brew install python@3.13 >/dev/null 2>&1 || \
         brew install python3 >/dev/null 2>&1; then
        if check_python_version python3.13; then
          PYTHON_FOUND=1
          PYTHON_CMD="python3.13"
        elif check_python_version python3; then
          PYTHON_FOUND=1
          PYTHON_CMD="python3"
        fi
      fi
      ;;
  esac
  
  # Если Python всё ещё не найден или версия < 3.13 - выводим сообщение
  if [ $PYTHON_FOUND -eq 0 ]; then
    echo "[ERROR] Python 3.13+ не найден. Пожалуйста, установите Python 3.13+ вручную."
    if [[ "$OSTYPE" == "darwin"* ]]; then
      echo "Для macOS: https://www.python.org/downloads/macos/"
    else
      echo "Для Linux: используйте менеджер пакетов вашего дистрибутива или установите с https://www.python.org/downloads/"
    fi
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

# Определение Python для запуска
if [[ -x ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
  PYTHON_BIN="python"
else
  # Используем найденную команду Python или fallback
  PYTHON_BIN="${PYTHON_CMD:-python3}"
  echo "[WARN] Virtual environment not found. Falling back to ${PYTHON_BIN}."
fi
exec "$PYTHON_BIN" launcher/infobot_manager.py "$@"

