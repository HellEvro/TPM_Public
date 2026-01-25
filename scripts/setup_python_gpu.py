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
        # Windows: используем winget
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
                # Запускаем установку БЕЗ --silent чтобы видеть прогресс
                # Используем Popen для вывода в реальном времени
                try:
                    process = subprocess.Popen(
                        ['winget', 'install', '--id', 'Python.Python.3.12', '--accept-package-agreements', '--accept-source-agreements'],
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
                        print("[INFO] Ожидание обновления PATH (5 секунд)...")
                        time.sleep(5)
                        return True
                    else:
                        error_output = '\n'.join(output_lines[-10:])  # Последние 10 строк
                        print(f"[WARNING] Ошибка установки через winget (код {process.returncode})")
                        if error_output:
                            print(f"[WARNING] Последние строки вывода:\n{error_output}")
                        print("[INFO] Попробуйте установить Python 3.12 вручную:")
                        print("   winget install Python.Python.3.12")
                        print("   или скачайте с: https://www.python.org/downloads/release/python-3120/")
                except Exception as e:
                    print(f"[ERROR] Ошибка при запуске winget: {e}")
                    print("[INFO] Попробуйте установить Python 3.12 вручную:")
                    print("   winget install Python.Python.3.12")
            else:
                print("[WARNING] winget не найден или недоступен")
        except subprocess.TimeoutExpired:
            print("[ERROR] Таймаут при установке Python 3.12 (более 10 минут)")
            print("[INFO] Установка может продолжаться в фоне. Проверьте через несколько минут:")
            print("   py -3.12 --version")
            return False
        except (FileNotFoundError, Exception) as e:
            print(f"[WARNING] Не удалось установить Python через winget: {e}")
            print("[INFO] Установите Python 3.12 вручную:")
            print("   winget install Python.Python.3.12")
            print("   или скачайте с: https://www.python.org/downloads/release/python-3120/")
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

def check_python_312_available():
    """Проверяет, доступен ли Python 3.12 в системе, при необходимости устанавливает"""
    # Сначала проверяем текущий Python
    if sys.version_info.major == 3 and sys.version_info.minor == 12:
        return True, 'python'  # Текущий Python уже 3.12
    
    # Если текущий не 3.12, ищем внешние команды
    commands = [
        ['py', '-3.12', '--version'],
        ['python3.12', '--version'],
        ['python312', '--version'],
    ]
    for cmd in commands:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and '3.12' in (r.stdout or '') + (r.stderr or ''):
                launcher = 'py -3.12' if cmd[0] == 'py' else cmd[0]
                return True, launcher
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    
    # Python 3.12 не найден - пробуем установить
    print("[INFO] Python 3.12 не найден, пытаемся установить автоматически...")
    if install_python312():
        # После установки пробуем найти снова
        import time
        time.sleep(2)
        for cmd in commands:
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if r.returncode == 0 and '3.12' in (r.stdout or '') + (r.stderr or ''):
                    launcher = 'py -3.12' if cmd[0] == 'py' else cmd[0]
                    return True, launcher
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


def create_venv(python_cmd, project_root, python_version="3.11"):
    """Создаёт .venv_gpu с указанной командой Python."""
    venv_path = project_root / '.venv_gpu'
    if venv_path.exists():
        print("Удаление старого .venv_gpu...")
        shutil.rmtree(venv_path)
    cmd = python_cmd.split() if isinstance(python_cmd, str) else list(python_cmd)
    create = cmd + ['-m', 'venv', str(venv_path)]
    print(f"Создание .venv_gpu с Python {python_version}...")
    try:
        subprocess.run(create, check=True)
        print("[OK] .venv_gpu создан")
        
        # Небольшая задержка для завершения создания venv
        import time
        time.sleep(2)
        
        # Проверяем что Python доступен в venv
        if platform.system() == 'win32':
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
            print(f"[WARNING] Python не найден в venv по пути: {venv_python}")
        
        return venv_path
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Ошибка: {e}")
        return None


def install_dependencies(venv_path, project_root):
    """Устанавливает зависимости в .venv_gpu."""
    # Определяем правильный путь к Python в venv
    if platform.system() == 'win32':
        python = venv_path / 'Scripts' / 'python.exe'
    else:
        python = venv_path / 'bin' / 'python'
    
    if not python.exists():
        print(f"[ERROR] Python не найден в .venv_gpu по пути: {python}")
        print(f"[INFO] Проверяем альтернативные пути...")
        # Пробуем альтернативные пути
        alt_paths = []
        if platform.system() == 'win32':
            alt_paths = [
                venv_path / 'Scripts' / 'pythonw.exe',
                venv_path / 'Scripts' / 'python3.exe',
            ]
        else:
            alt_paths = [
                venv_path / 'bin' / 'python3',
            ]
        
        for alt_path in alt_paths:
            if alt_path.exists():
                print(f"[INFO] Найден альтернативный Python: {alt_path}")
                python = alt_path
                break
        else:
            print("[ERROR] Python не найден в .venv_gpu ни по одному из путей")
            return False
    
    # Используем requirements.txt для установки всех зависимостей (включая TensorFlow)
    req_main = project_root / 'requirements.txt'
    
    print("Установка зависимостей...")
    try:
        # Всегда используем python -m pip вместо прямого вызова pip
        subprocess.run([str(python), '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel', '--no-warn-script-location'], check=True)
        
        # Устанавливаем все зависимости из requirements.txt (включая TensorFlow для Python 3.12)
        if req_main.exists():
            print("[INFO] Установка всех зависимостей из requirements.txt...")
            # Для Python 3.12 заменяем tf-nightly на tensorflow
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
                with open(req_main, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Убираем environment marker для Python 3.12 (в .venv_gpu всегда Python 3.12)
                    content = content.replace('tensorflow>=2.16.0; python_version < "3.14"', 'tensorflow>=2.16.0')
                    content = content.replace('tf-nightly>=2.21.0.dev', 'tensorflow>=2.16.0')
                    tmp.write(content)
                    tmp_path = tmp.name
                
                try:
                    subprocess.run([str(python), '-m', 'pip', 'install', '-r', tmp_path, '--no-warn-script-location'], check=True)
                    print("[OK] Все зависимости установлены (включая TensorFlow для Python 3.12)")
                finally:
                    import os
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
        else:
            # Fallback: устанавливаем TensorFlow вручную
            print("[WARNING] requirements.txt не найден, устанавливаю TensorFlow вручную...")
            try:
                subprocess.run([str(python), '-m', 'pip', 'install', 'tensorflow[and-cuda]>=2.16.0', '--no-warn-script-location'], check=True)
                print("[OK] TensorFlow с GPU установлен")
            except subprocess.CalledProcessError:
                subprocess.run([str(python), '-m', 'pip', 'install', 'tensorflow>=2.16.0', '--no-warn-script-location'], check=True)
                print("[OK] TensorFlow (CPU) установлен")
        
        print("[OK] Все зависимости установлены")
        
        # Компилируем защищенные модули под Python 3.12 для .venv_gpu
        print("[INFO] Компиляция защищенных модулей под Python 3.12...")
        try:
            compile_script = project_root / 'license_generator' / 'compile_all.py'
            result = subprocess.run(
                [str(python), str(compile_script)],
                cwd=project_root,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("[OK] Защищенные модули скомпилированы под Python 3.12")
            else:
                print(f"[WARNING] Ошибка компиляции модулей: {result.stderr[:200]}")
        except Exception as e:
            print(f"[WARNING] Не удалось скомпилировать модули: {e}")
        
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

    # TensorFlow НЕ поддерживает Python 3.14+, используем Python 3.12 для .venv_gpu
    ok, cmd = check_python_312_available()
    python_version = "3.12"
    if not ok:
        print("[ERROR] Python 3.12 не найден.")
        print("ВНИМАНИЕ: TensorFlow НЕ поддерживает Python 3.14+!")
        print("Для .venv_gpu требуется Python 3.12: https://www.python.org/downloads/release/python-3120/")
        if platform.system() == 'Windows':
            print("Или используйте: py -3.12")
        else:
            print("Или: sudo apt install python3.12 python3.12-venv  (Ubuntu/Debian)")
            print("     brew install python@3.12  (macOS)")
        return 1
    
    print(f"[OK] Python {python_version}: {cmd}")
    print(f"[INFO] Python {python_version} выбран для .venv_gpu (TensorFlow требует Python <= 3.12)")

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
