@echo off
setlocal enabledelayedexpansion
cd /d %~dp0

REM InfoBot требует Python 3.14 или выше
set "PYTHON_FOUND=0"
set "PYTHON_CMD="

REM Проверяем py -3.14 (приоритет на Windows)
py -3.14 --version >nul 2>&1
if !errorlevel!==0 (
    set "PYTHON_FOUND=1"
    set "PYTHON_CMD=py -3.14"
)

REM Проверяем python3.14
if !PYTHON_FOUND!==0 (
    python3.14 --version >nul 2>&1
    if !errorlevel!==0 (
        set "PYTHON_FOUND=1"
        set "PYTHON_CMD=python3.14"
    )
)

REM Проверяем python (должна быть 3.14 или выше)
if !PYTHON_FOUND!==0 (
    python -c "import sys; exit(0 if sys.version_info[:2]>=(3,14) else 1)" >nul 2>&1
    if !errorlevel!==0 (
        set "PYTHON_FOUND=1"
        set "PYTHON_CMD=python"
    )
)

REM Если Python не найден — устанавливаем через winget
if !PYTHON_FOUND!==0 (
    winget --version >nul 2>&1
    if !errorlevel!==0 (
        echo [INFO] Установка Python 3.14.2+ через winget...
        winget install --id Python.Python.3.14 --silent --accept-package-agreements --accept-source-agreements
        timeout /t 8 /nobreak >nul
        REM Проверяем установку
        py -3.14 --version >nul 2>&1
        if !errorlevel!==0 (
            set "PYTHON_FOUND=1"
            set "PYTHON_CMD=py -3.14"
        ) else (
            REM Пробуем еще раз после небольшой задержки
            timeout /t 3 /nobreak >nul
            py -3.14 --version >nul 2>&1
            if !errorlevel!==0 (
                set "PYTHON_FOUND=1"
                set "PYTHON_CMD=py -3.14"
            )
        )
    )
    if !PYTHON_FOUND!==0 (
        echo [ERROR] Python 3.14.2+ не найден. Установите: https://www.python.org/downloads/
        start https://www.python.org/downloads/
        exit /b 1
    )
)

REM Функция проверки установки Git
set "GIT_FOUND=0"
git --version >nul 2>&1
if !errorlevel!==0 (
    set "GIT_FOUND=1"
) else (
    REM Проверяем стандартные пути установки Git
    if exist "C:\Program Files\Git\cmd\git.exe" (
        set "PATH=%PATH%;C:\Program Files\Git\cmd"
        set "GIT_FOUND=1"
    ) else if exist "C:\Program Files (x86)\Git\cmd\git.exe" (
        set "PATH=%PATH%;C:\Program Files (x86)\Git\cmd"
        set "GIT_FOUND=1"
    ) else if exist "C:\Program Files\Git\bin\git.exe" (
        set "PATH=%PATH%;C:\Program Files\Git\bin"
        set "GIT_FOUND=1"
    )
)

REM Проверка Git (только если Python установлен)
if !GIT_FOUND!==0 (
    winget --version >nul 2>&1
    if !errorlevel!==0 (
        echo [INFO] Установка Git через winget...
        REM Тихая установка Git с максимальной интеграцией и nano редактором
        REM Параметры: /VERYSILENT /NORESTART /NOCANCEL /SP- /SUPPRESSMSGBOXES
        REM /COMPONENTS=icons,ext\shellhere,assoc,assoc_sh /PATHOPTION=user /EDITOR=nano
        winget install --id Git.Git --silent --accept-package-agreements --accept-source-agreements --override "/VERYSILENT /NORESTART /NOCANCEL /SP- /SUPPRESSMSGBOXES /COMPONENTS=icons,ext\shellhere,assoc,assoc_sh /PATHOPTION=user /EDITOR=nano"
        REM Ждем завершения установки (winget может установить Git в фоне)
        timeout /t 8 /nobreak >nul
        REM Проверяем установку через стандартные пути
        if exist "C:\Program Files\Git\cmd\git.exe" (
            set "PATH=%PATH%;C:\Program Files\Git\cmd"
            set "GIT_FOUND=1"
        ) else if exist "C:\Program Files (x86)\Git\cmd\git.exe" (
            set "PATH=%PATH%;C:\Program Files (x86)\Git\cmd"
            set "GIT_FOUND=1"
        ) else if exist "C:\Program Files\Git\bin\git.exe" (
            set "PATH=%PATH%;C:\Program Files\Git\bin"
            set "GIT_FOUND=1"
        )
        REM Если Git все еще не найден, ждем еще немного
        if !GIT_FOUND!==0 (
            timeout /t 5 /nobreak >nul
            git --version >nul 2>&1
            if !errorlevel!==0 (
                set "GIT_FOUND=1"
            )
        )
    )
)

REM Безопасная инициализация Git репозитория (если Git установлен)
if !GIT_FOUND!==1 (
    REM Настраиваем Git пользователя (если не настроен) - ДО любых операций
    git config user.name >nul 2>&1
    if !errorlevel! neq 0 (
        git config user.name "InfoBot User" >nul 2>&1
    )
    git config user.email >nul 2>&1
    if !errorlevel! neq 0 (
        git config user.email "infobot@local" >nul 2>&1
    )
    
    if not exist ".git" (
        REM Инициализируем репозиторий БЕЗ pull/fetch, чтобы не перезаписать существующие файлы
        git init >nul 2>&1
        git branch -m main >nul 2>&1
        REM Добавляем remote с HTTPS URL
        git remote add origin https://github.com/HellEvro/TPM_Public.git >nul 2>&1
    ) else (
        REM Репозиторий уже существует - проверяем и исправляем remote URL
        git remote get-url origin >nul 2>&1
        if !errorlevel!==0 (
            REM Remote существует - проверяем, используется ли SSH
            for /f "tokens=*" %%i in ('git remote get-url origin') do set "REMOTE_URL=%%i"
            echo !REMOTE_URL! | findstr /C:"git@github.com" >nul 2>&1
            if !errorlevel!==0 (
                REM Используется SSH - меняем на HTTPS
                git remote set-url origin https://github.com/HellEvro/TPM_Public.git >nul 2>&1
            )
        ) else (
            REM Remote не существует - добавляем
            git remote add origin https://github.com/HellEvro/TPM_Public.git >nul 2>&1
        )
    )
    
    REM Делаем первый коммит, если нет коммитов (независимо от того, новый репозиторий или существующий)
    git rev-list --count HEAD >nul 2>&1
    if !errorlevel! neq 0 (
        REM Нет коммитов - делаем первый коммит
        REM Добавляем все файлы
        git add -A >nul 2>&1
        REM Делаем коммит
        git commit -m "Initial commit: InfoBot Public repository" >nul 2>&1
    )
)

REM Обновляем PATH для текущей сессии, если Git установлен
if !GIT_FOUND!==1 (
    REM Проверяем стандартные пути Git и добавляем в PATH
    if exist "C:\Program Files\Git\cmd\git.exe" (
        set "PATH=%PATH%;C:\Program Files\Git\cmd"
    ) else if exist "C:\Program Files (x86)\Git\cmd\git.exe" (
        set "PATH=%PATH%;C:\Program Files (x86)\Git\cmd"
    ) else if exist "C:\Program Files\Git\bin\git.exe" (
        set "PATH=%PATH%;C:\Program Files\Git\bin"
    )
)

REM Проверка и обновление .venv для Python 3.14
if exist scripts\ensure_python314_venv.py (
    echo [INFO] Проверка и обновление .venv для Python 3.14...
    if not "!PYTHON_CMD!"=="" (
        !PYTHON_CMD! scripts\ensure_python314_venv.py >nul 2>&1
    ) else (
        python scripts\ensure_python314_venv.py >nul 2>&1
    )
)

REM Определение Python для запуска: .venv_gpu > .venv > py -3.14
if exist .venv_gpu\Scripts\activate.bat (
    call .venv_gpu\Scripts\activate.bat
    set "PYTHON_BIN=python"
) else if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    set "PYTHON_BIN=python"
) else (
    if not "!PYTHON_CMD!"=="" (set "PYTHON_BIN=!PYTHON_CMD!") else (
        echo [ERROR] Python 3.14+ не найден!
        exit /b 1
    )
)

!PYTHON_BIN! launcher\infobot_manager.py %*
endlocal

