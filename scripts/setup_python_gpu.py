#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Настройка Python 3.12 для InfoBot (по умолчанию).
Создаёт .venv_gpu с Python 3.12 и устанавливает зависимости, включая TensorFlow с GPU.
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


def check_python_312_available():
    """Проверяет, доступен ли Python 3.12 в системе."""
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
    return False, None


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


def create_venv(python_cmd, project_root):
    """Создаёт .venv_gpu с указанной командой Python."""
    venv_path = project_root / '.venv_gpu'
    if venv_path.exists():
        print("Удаление старого .venv_gpu...")
        shutil.rmtree(venv_path)
    cmd = python_cmd.split() if isinstance(python_cmd, str) else list(python_cmd)
    create = cmd + ['-m', 'venv', str(venv_path)]
    print("Создание .venv_gpu с Python 3.12...")
    try:
        subprocess.run(create, check=True)
        print("[OK] .venv_gpu создан")
        return venv_path
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Ошибка: {e}")
        return None


def install_dependencies(venv_path, project_root):
    """Устанавливает зависимости в .venv_gpu."""
    pip = venv_path / ('Scripts' if platform.system() == 'win32' else 'bin') / 'pip'
    python = venv_path / ('Scripts' if platform.system() == 'win32' else 'bin') / 'python'
    if not pip.exists():
        pip = python.parent / ('pip.exe' if platform.system() == 'win32' else 'pip')
    if not pip.exists():
        print("[ERROR] pip не найден в .venv_gpu")
        return False
    req = project_root / 'requirements.txt'
    print("Установка зависимостей...")
    try:
        subprocess.run([str(python), '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel', '--no-warn-script-location'], check=True)
        subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(req), '--no-warn-script-location'], check=True)
        print("[OK] Зависимости установлены")
        try:
            subprocess.run([str(python), '-m', 'pip', 'install', 'tensorflow[and-cuda]>=2.13.0', '--no-warn-script-location'], check=True)
            print("[OK] TensorFlow с GPU установлен")
        except subprocess.CalledProcessError:
            subprocess.run([str(python), '-m', 'pip', 'install', 'tensorflow>=2.13.0', '--no-warn-script-location'], check=True)
            print("[OK] TensorFlow (CPU) установлен")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Ошибка установки: {e}")
        return False


def main():
    print("=" * 80)
    print("НАСТРОЙКА PYTHON 3.12 ДЛЯ INFOBOT")
    print("=" * 80)
    root = Path(__file__).resolve().parents[1]
    print(f"Текущий Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    ok, cmd = check_python_312_available()
    if not ok:
        print("[ERROR] Python 3.12 не найден.")
        if platform.system() == 'Windows':
            install_python_312_windows()
        else:
            print("Установите: sudo apt install python3.12 python3.12-venv  (Ubuntu/Debian)")
            print("           brew install python@3.12  (macOS)")
        return 1
    print(f"[OK] Python 3.12: {cmd}")

    venv = create_venv(cmd, root)
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
