#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Настройка Python 3.14 для InfoBot (по умолчанию).
Создаёт .venv_gpu с Python 3.14 и устанавливает зависимости, включая TensorFlow с GPU.
"""

import sys
import os
import subprocess
import platform
import shutil
from pathlib import Path

if platform.system() == 'Windows':
    try:
        if getattr(sys.stdout, 'encoding', None) != 'utf-8':
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass


def check_python_311_available():
    """Проверяет, доступен ли Python 3.11 в системе (для GPU поддержки TensorFlow)."""
    # Сначала проверяем текущий Python
    if sys.version_info.major == 3 and sys.version_info.minor == 11:
        return True, 'python'  # Текущий Python уже 3.11
    
    # Если текущий не 3.11, ищем внешние команды
    commands = [
        ['py', '-3.11', '--version'],
        ['python3.11', '--version'],
        ['python311', '--version'],
    ]
    for cmd in commands:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and '3.11' in (r.stdout or '') + (r.stderr or ''):
                launcher = 'py -3.11' if cmd[0] == 'py' else cmd[0]
                return True, launcher
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return False, None

def install_python312():
    """Автоматически устанавливает Python 3.12 через системные менеджеры пакетов"""
    if platform.system() == 'Windows':
        # Windows: используем winget с параметрами для автономной установки
        try:
            result = subprocess.run(
                ['winget', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print("[INFO] Установка Python 3.12 через winget...")
                print("[INFO] Это может занять несколько минут, пожалуйста подождите...")
                print("[INFO] Используем автономный установщик для избежания проблем с Package Cache...")
                # Используем --scope machine для установки в Program Files (не требует Package Cache)
                # и --override для передачи параметров установщику
                try:
                    process = subprocess.Popen(
                        [
                            'winget', 'install', 
                            '--id', 'Python.Python.3.12',
                            '--scope', 'machine',  # Установка для всех пользователей
                            '--accept-package-agreements', 
                            '--accept-source-agreements',
                            '--override', '/quiet InstallAllUsers=1 PrependPath=1'  # Автономная установка
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True
                    )
                    
                    # Выводим вывод в реальном времени
                    output_lines = []
                    for line in process.stdout:
                        line = line.rstrip()
                        if line:
                            print(f"  {line}")
                            output_lines.append(line)
                    
                    process.wait()
                    
                    if process.returncode == 0:
                        print("[OK] Python 3.12 установлен")
                        import time
                        print("[INFO] Ожидание обновления PATH (10 секунд)...")
                        time.sleep(10)  # Увеличиваем задержку для обновления PATH
                        return True
                    else:
                        error_output = '\n'.join(output_lines[-10:])  # Последние 10 строк
                        print(f"[WARNING] Ошибка установки через winget (код {process.returncode})")
                        if error_output:
                            print(f"[WARNING] Последние строки вывода:\n{error_output}")
                        print("[INFO] Попробуйте установить Python 3.12 вручную:")
                        print("   Скачайте автономный установщик: https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe")
                        print("   Запустите с параметрами: python-3.12.0-amd64.exe /quiet InstallAllUsers=1 PrependPath=1")
                except Exception as e:
                    print(f"[ERROR] Ошибка при запуске winget: {e}")
                    print("[INFO] Попробуйте установить Python 3.12 вручную:")
                    print("   Скачайте автономный установщик: https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe")
            else:
                print("[WARNING] winget не найден или недоступен")
                print("[INFO] Установите Python 3.12 вручную:")
                print("   Скачайте автономный установщик: https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe")
        except subprocess.TimeoutExpired:
            print("[ERROR] Таймаут при установке Python 3.12 (более 10 минут)")
            print("[INFO] Установка может продолжаться в фоне. Проверьте через несколько минут")
            return False
        except (FileNotFoundError, Exception) as e:
            print(f"[WARNING] Не удалось установить Python через winget: {e}")
            print("[INFO] Установите Python 3.12 вручную:")
            print("   Скачайте автономный установщик: https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe")
            print("   Запустите с параметрами: python-3.12.0-amd64.exe /quiet InstallAllUsers=1 PrependPath=1")
    else:
        # Linux/macOS: используем системные менеджеры пакетов
        system = platform.system()
        
        if system == 'Linux':
            # Определяем менеджер пакетов
            for pkg_mgr, install_cmd in [
                ('apt-get', ['sudo', 'apt-get', 'update', '-qq', '&&', 'sudo', 'apt-get', 'install', '-y', 'python3.12', 'python3.12-venv']),
                ('yum', ['sudo', 'yum', 'install', '-y', 'python3.12']),
                ('dnf', ['sudo', 'dnf', 'install', '-y', 'python3.12']),
            ]:
                if shutil.which(pkg_mgr):
                    try:
                        print(f"[INFO] Установка Python 3.12 через {pkg_mgr}...")
                        if pkg_mgr == 'apt-get':
                            subprocess.run(['sudo', 'apt-get', 'update', '-qq'], timeout=60, check=True)
                            subprocess.run(['sudo', 'apt-get', 'install', '-y', 'python3.12', 'python3.12-venv'], timeout=300, check=True)
                        else:
                            subprocess.run(install_cmd, timeout=300, check=True)
                        print("[OK] Python 3.12 установлен")
                        return True
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, Exception) as e:
                        print(f"[WARNING] Ошибка установки через {pkg_mgr}: {e}")
                        continue
        elif system == 'Darwin':  # macOS
            if shutil.which('brew'):
                try:
                    print("[INFO] Установка Python 3.12 через brew...")
                    subprocess.run(['brew', 'install', 'python@3.12'], timeout=600, check=True)
                    print("[OK] Python 3.12 установлен")
                    return True
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, Exception) as e:
                    print(f"[WARNING] Ошибка установки через brew: {e}")
    
    return False

def find_python312_executable():
    """Находит путь к исполняемому файлу Python 3.12, избегая py launcher"""
    if platform.system() != 'Windows':
        # Для Linux/macOS используем стандартные команды
        import shutil
        for cmd in ['python3.12', 'python312']:
            exe_path = shutil.which(cmd)
            if exe_path:
                try:
                    result = subprocess.run(
                        [exe_path, '--version'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0 and '3.12' in (result.stdout or '') + (result.stderr or ''):
                        return exe_path
                except:
                    continue
        return None
    
    # Windows: проверяем стандартные пути установки Python
    possible_paths = []
    
    # 1. Проверяем через реестр (стандартные установки Python)
    try:
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Python\PythonCore\3.12\InstallPath") as key:
                install_path = winreg.QueryValueEx(key, "")[0]
                python_exe = os.path.join(install_path, "python.exe")
                if os.path.exists(python_exe):
                    possible_paths.append(python_exe)
        except:
            pass
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"SOFTWARE\Python\PythonCore\3.12\InstallPath") as key:
                install_path = winreg.QueryValueEx(key, "")[0]
                python_exe = os.path.join(install_path, "python.exe")
                if os.path.exists(python_exe):
                    possible_paths.append(python_exe)
        except:
            pass
    except ImportError:
        pass  # winreg недоступен (не Windows)
    
    # 2. Проверяем стандартные пути установки (исключая Package Cache)
    common_paths = [
        r"C:\Python312\python.exe",
        r"C:\Program Files\Python312\python.exe",
        r"C:\Program Files (x86)\Python312\python.exe",
        r"C:\Program Files (x86)\Python312-32\python.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Python\Python312\python.exe"),
        os.path.expanduser(r"~\AppData\Local\Programs\Python\Python312-32\python.exe"),
        # Также проверяем пользовательские установки Python
        os.path.expanduser(r"~\AppData\Local\Python\bin\python.exe"),  # Пользовательский Python (универсальный путь)
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            possible_paths.append(path)
    
    # 3. Проверяем через PATH (но НЕ через py launcher)
    import shutil
    for cmd in ['python3.12', 'python312']:
        exe_path = shutil.which(cmd)
        if exe_path and os.path.exists(exe_path):
            # Проверяем что это не ссылка на py launcher и не Package Cache
            if 'WindowsApps' not in exe_path and 'Package Cache' not in exe_path:
                possible_paths.append(exe_path)
    
    # 4. Проверяем все установленные версии Python через py launcher (но только для поиска реальных путей)
    try:
        # Используем py -0 для списка установленных версий, но не для запуска
        result = subprocess.run(
            ['py', '-0'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Парсим вывод и ищем Python 3.12
            import re
            for line in result.stdout.split('\n'):
                if '3.12' in line:
                    # Пытаемся извлечь путь из строки
                    # Формат обычно: " -3.12-64    C:\Python312\python.exe"
                    match = re.search(r'([A-Z]:[^\s]+python\.exe)', line)
                    if match:
                        path = match.group(1)
                        if os.path.exists(path) and 'Package Cache' not in path:
                            possible_paths.append(path)
    except:
        pass  # py launcher может быть недоступен
    
    # Проверяем каждый найденный путь
    for python_exe in possible_paths:
        try:
            result = subprocess.run(
                [python_exe, '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and '3.12' in (result.stdout or '') + (result.stderr or ''):
                return python_exe
        except:
            continue
    
    return None

def check_python_312_available():
    """Проверяет, доступен ли Python 3.12 в системе, при необходимости устанавливает"""
    # Сначала проверяем текущий Python
    if sys.version_info.major == 3 and sys.version_info.minor == 12:
        return True, sys.executable  # Текущий Python уже 3.12
    
    # Пытаемся найти Python 3.12 напрямую (без py launcher)
    python312_exe = find_python312_executable()
    if python312_exe:
        print(f"[OK] Найден Python 3.12: {python312_exe}")
        return True, python312_exe
    
    # Если не нашли напрямую, пробуем команды (но НЕ py -3.12, так как он пытается установить)
    import shutil
    commands = ['python3.12', 'python312']
    for cmd in commands:
        exe_path = shutil.which(cmd)
        if exe_path:
            try:
                r = subprocess.run([exe_path, '--version'], capture_output=True, text=True, timeout=5)
                if r.returncode == 0 and '3.12' in (r.stdout or '') + (r.stderr or ''):
                    print(f"[OK] Найден Python 3.12 через PATH: {exe_path}")
                    return True, exe_path
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
    
    # Python 3.12 не найден - пробуем установить
    print("[INFO] Python 3.12 не найден, пытаемся установить автоматически...")
    if install_python312():
        # После установки пробуем найти снова
        import time
        time.sleep(5)  # Увеличиваем задержку для обновления PATH
        python312_exe = find_python312_executable()
        if python312_exe:
            print(f"[OK] Python 3.12 найден после установки: {python312_exe}")
            return True, python312_exe
        
        for cmd in commands:
            exe_path = shutil.which(cmd)
            if exe_path:
                try:
                    r = subprocess.run([exe_path, '--version'], capture_output=True, text=True, timeout=5)
                    if r.returncode == 0 and '3.12' in (r.stdout or '') + (r.stderr or ''):
                        print(f"[OK] Python 3.12 найден после установки: {exe_path}")
                        return True, exe_path
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    continue
    
    return False, None


def install_python_311_windows():
    """Инструкции по установке Python 3.11 на Windows (для GPU поддержки)."""
    print("\n" + "=" * 80)
    print("УСТАНОВКА PYTHON 3.11 (ДЛЯ GPU ПОДДЕРЖКИ)")
    print("=" * 80)
    print("\n1. Скачайте: https://www.python.org/downloads/release/python-3110/")
    print("2. При установке отметьте 'Add Python 3.11 to PATH'")
    print("3. Проверьте: py -3.11 --version")
    print("4. Запустите снова: py -3.11 scripts/setup_python_gpu.py")
    print("=" * 80)

def install_python_312_windows():
    """Инструкции по установке Python 3.12 на Windows."""
    print("\n" + "=" * 80)
    print("УСТАНОВКА PYTHON 3.12")
    print("=" * 80)
    print("\n1. Скачайте: https://www.python.org/downloads/release/python-3120/")
    print("2. При установке отметьте 'Add Python 3.12 to PATH'")
    print("3. Проверьте: py -3.12 --version")
    print("4. Запустите снова: py -3.12 scripts/setup_python_gpu.py")
    print("=" * 80)


def create_venv(python_cmd, project_root, python_version="3.12"):
    """Создаёт .venv_gpu с указанной командой Python."""
    venv_path = project_root / '.venv_gpu'
    if venv_path.exists():
        print("Удаление старого .venv_gpu...")
        shutil.rmtree(venv_path)
    
    # Правильно обрабатываем команду Python
    # python_cmd теперь всегда должен быть путем к исполняемому файлу, а не командой
    if isinstance(python_cmd, str):
        # Если это путь к исполняемому файлу, используем его напрямую
        if os.path.exists(python_cmd):
            cmd = [python_cmd]
        elif python_cmd.startswith('py '):
            # Старый формат 'py -3.12' - не используем, так как он пытается установить
            print("[ERROR] Не используйте 'py -3.12' - он пытается установить Python через Microsoft Store")
            print("[INFO] Используйте прямой путь к Python 3.12 или установите его вручную")
            return None
        else:
            # Пробуем найти через which/shutil
            exe_path = shutil.which(python_cmd)
            if exe_path and os.path.exists(exe_path):
                cmd = [exe_path]
            else:
                cmd = [python_cmd]
    else:
        cmd = list(python_cmd)
    
    # Проверяем, что команда существует перед использованием
    if not os.path.exists(cmd[0]):
        print(f"[ERROR] Python не найден по пути: {cmd[0]}")
        print("[INFO] Попробуйте установить Python 3.12 вручную или запустите: python scripts/setup_python_gpu.py")
        return None
    
    create = cmd + ['-m', 'venv', str(venv_path)]
    print(f"Создание .venv_gpu с Python {python_version}...")
    print(f"[DEBUG] Команда создания venv: {' '.join(create)}")
    try:
        result = subprocess.run(create, check=True, capture_output=True, text=True)
        print("[OK] .venv_gpu создан")
        
        # Небольшая задержка для завершения создания venv
        import time
        time.sleep(2)
        
        # Проверяем что Python доступен в venv
        if platform.system() == 'Windows' or platform.system() == 'win32':
            venv_python = venv_path / 'Scripts' / 'python.exe'
        else:
            venv_python = venv_path / 'bin' / 'python'
        
        if venv_python.exists():
            # Проверяем что Python работает
            test_result = subprocess.run(
                [str(venv_python), '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if test_result.returncode == 0:
                print(f"[OK] Python в venv работает: {test_result.stdout.strip()}")
            else:
                print(f"[WARNING] Python в venv не отвечает, но файл существует")
        else:
            # Пробуем альтернативные пути
            alt_paths = []
            if platform.system() == 'Windows' or platform.system() == 'win32':
                alt_paths = [
                    venv_path / 'Scripts' / 'pythonw.exe',
                    venv_path / 'Scripts' / 'python3.exe',
                    venv_path / 'bin' / 'python',  # На случай если venv создан неправильно
                ]
            else:
                alt_paths = [
                    venv_path / 'bin' / 'python3',
                ]
            
            found = False
            for alt_path in alt_paths:
                if alt_path.exists():
                    print(f"[OK] Python найден по альтернативному пути: {alt_path}")
                    found = True
                    break
            
            if not found:
                print(f"[WARNING] Python не найден в venv по пути: {venv_python}")
                print(f"[INFO] Проверяем альтернативные пути...")
                for alt_path in alt_paths:
                    print(f"  - {alt_path}: {'существует' if alt_path.exists() else 'не найден'}")
        
        return venv_path
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Ошибка: {e}")
        return None


def install_dependencies(venv_path, project_root):
    """Устанавливает зависимости в .venv_gpu."""
    # Определяем правильный путь к Python в venv
    if platform.system() == 'Windows' or platform.system() == 'win32':
        python = venv_path / 'Scripts' / 'python.exe'
    else:
        python = venv_path / 'bin' / 'python'
    
    if not python.exists():
        print(f"[WARNING] Python не найден в .venv_gpu по пути: {python}")
        print(f"[INFO] Проверяем альтернативные пути...")
        # Пробуем альтернативные пути
        alt_paths = []
        if platform.system() == 'Windows' or platform.system() == 'win32':
            alt_paths = [
                venv_path / 'Scripts' / 'pythonw.exe',
                venv_path / 'Scripts' / 'python3.exe',
                venv_path / 'bin' / 'python',  # На случай если venv создан неправильно
            ]
        else:
            alt_paths = [
                venv_path / 'bin' / 'python3',
            ]
        
        found = False
        for alt_path in alt_paths:
            if alt_path.exists():
                print(f"[OK] Найден альтернативный Python: {alt_path}")
                python = alt_path
                found = True
                break
        
        if not found:
            print("[ERROR] Python не найден в .venv_gpu ни по одному из путей:")
            print(f"  - {python}")
            for alt_path in alt_paths:
                print(f"  - {alt_path}")
            print("[INFO] Попробуйте пересоздать .venv_gpu:")
            print("  1. Удалите директорию .venv_gpu")
            print("  2. Запустите снова: python scripts/setup_python_gpu.py")
            return False
    
    # Используем requirements.txt для установки всех зависимостей (включая PyTorch)
    req_main = project_root / 'requirements.txt'
    
    print("Установка зависимостей...")
    try:
        # Всегда используем python -m pip вместо прямого вызова pip
        subprocess.run([str(python), '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel', '--no-warn-script-location'], check=True)
        
        # Устанавливаем все зависимости из requirements.txt (включая PyTorch)
        if req_main.exists():
            print("[INFO] Установка всех зависимостей из requirements.txt...")
            # Проверяем наличие GPU для установки PyTorch с CUDA
            try:
                import subprocess as sp
                result = sp.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
                has_gpu = result.returncode == 0
            except:
                has_gpu = False
            
            if has_gpu:
                print("[INFO] Обнаружен NVIDIA GPU, устанавливаю PyTorch с CUDA поддержкой...")
                # Устанавливаем PyTorch с CUDA 12.1 (последняя стабильная версия)
                try:
                    subprocess.run([str(python), '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu121', '--no-warn-script-location'], check=True)
                    print("[OK] PyTorch с CUDA 12.1 установлен")
                except:
                    # Fallback на CUDA 11.8
                    try:
                        subprocess.run([str(python), '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu118', '--no-warn-script-location'], check=True)
                        print("[OK] PyTorch с CUDA 11.8 установлен")
                    except:
                        # Fallback на CPU версию
                        subprocess.run([str(python), '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--no-warn-script-location'], check=True)
                        print("[OK] PyTorch (CPU) установлен")
            else:
                print("[INFO] GPU не обнаружен, устанавливаю PyTorch (CPU версия)...")
                subprocess.run([str(python), '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--no-warn-script-location'], check=True)
                print("[OK] PyTorch (CPU) установлен")
            
            # Устанавливаем остальные зависимости из requirements.txt
            subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(req_main), '--no-warn-script-location'], check=True)
            print("[OK] Все зависимости установлены (включая PyTorch)")
        else:
            # Fallback: устанавливаем PyTorch вручную
            print("[WARNING] requirements.txt не найден, устанавливаю PyTorch вручную...")
            try:
                subprocess.run([str(python), '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu121', '--no-warn-script-location'], check=True)
                print("[OK] PyTorch с GPU установлен")
            except subprocess.CalledProcessError:
                subprocess.run([str(python), '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--no-warn-script-location'], check=True)
                print("[OK] PyTorch (CPU) установлен")
        
        print("[OK] Все зависимости установлены")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Ошибка установки: {e}")
        return False


def main():
    print("=" * 80)
    print("НАСТРОЙКА PYTHON ДЛЯ INFOBOT С GPU ПОДДЕРЖКОЙ")
    print("=" * 80)
    root = Path(__file__).resolve().parents[1]
    print(f"Текущий Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"[DEBUG] Корень проекта: {root}")
    print(f"[DEBUG] Текущая рабочая директория: {os.getcwd()}")

    # PyTorch поддерживает Python 3.14+, но для совместимости используем Python 3.12 для .venv_gpu
    ok, cmd = check_python_312_available()
    python_version = "3.12"
    if not ok:
        print("[ERROR] Python 3.12 не найден.")
        print("ВНИМАНИЕ: Для совместимости используем Python 3.12 для .venv_gpu (PyTorch работает и на 3.14+)")
        print("Для .venv_gpu требуется Python 3.12: https://www.python.org/downloads/release/python-3120/")
        if platform.system() == 'Windows':
            print("Или используйте: py -3.12")
        else:
            print("Или: sudo apt install python3.12 python3.12-venv  (Ubuntu/Debian)")
            print("     brew install python@3.12  (macOS)")
        return 1
    
    print(f"[OK] Python {python_version}: {cmd}")
    print(f"[INFO] Python {python_version} выбран для .venv_gpu (PyTorch поддерживает все версии Python)")

    venv = create_venv(cmd, root, python_version)
    if not venv:
        return 1
    if not install_dependencies(venv, root):
        return 1

    print("\n" + "=" * 80)
    print("ГОТОВО")
    print("=" * 80)
    print(f"Окружение: {venv}")
    if platform.system() == 'win32':
        print(f"Запуск: {venv}\\Scripts\\python ai.py")
    else:
        print(f"Запуск: {venv}/bin/python ai.py")
    print("=" * 80)
    return 0


if __name__ == '__main__':
    sys.exit(main())
