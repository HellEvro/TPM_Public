#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Автоматическая настройка Python 3.12 для работы с GPU
Создает виртуальное окружение с Python 3.12 и устанавливает зависимости
"""

import sys
import os
import subprocess
import platform
import shutil
from pathlib import Path

# Настройка кодировки для Windows консоли
if platform.system() == 'Windows':
    try:
        # Пытаемся установить UTF-8 для вывода
        if sys.stdout.encoding != 'utf-8':
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

def check_python_312_available():
    """Проверяет, доступен ли Python 3.12 в системе"""
    # Пробуем разные варианты команд
    commands = [
        ['py', '-3.12', '--version'],
        ['python3.12', '--version'],
        ['python312', '--version'],
        ['python', '--version'],  # Проверим текущую версию
    ]
    
    for cmd in commands:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version_output = result.stdout.strip()
                if '3.12' in version_output or (cmd[0] == 'py' and '3.12' in str(cmd)):
                    return True, cmd[0] if cmd[0] != 'py' else 'py -3.12'
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    
    return False, None

def check_python_314_available():
    """Проверяет, доступен ли Python 3.14 в системе (для будущей поддержки)"""
    commands = [
        ['py', '-3.14', '--version'],
        ['python3.14', '--version'],
        ['python314', '--version'],
    ]
    
    for cmd in commands:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version_output = result.stdout.strip()
                if '3.14' in version_output or (cmd[0] == 'py' and '3.14' in str(cmd)):
                    return True, cmd[0] if cmd[0] != 'py' else 'py -3.14'
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    
    return False, None

def install_python_312_windows():
    """Инструкции по установке Python 3.12 на Windows"""
    print("\n" + "=" * 80)
    print("ИНСТРУКЦИЯ ПО УСТАНОВКЕ PYTHON 3.12")
    print("=" * 80)
    print("\n1. Скачайте Python 3.12 с официального сайта:")
    print("   https://www.python.org/downloads/release/python-3120/")
    print("\n2. При установке:")
    print("   - Отметьте 'Add Python 3.12 to PATH'")
    print("   - Выберите 'Install for all users' (опционально)")
    print("\n3. После установки проверьте:")
    print("   py -3.12 --version")
    print("\n4. Запустите этот скрипт снова:")
    print("   py -3.12 scripts/setup_python_gpu.py")
    print("\n" + "=" * 80)

def create_venv_with_python(python_cmd, project_root):
    """Создает виртуальное окружение с указанной версией Python"""
    venv_path = project_root / '.venv_gpu'
    
    # Удаляем старое окружение если существует
    if venv_path.exists():
        print(f"Удаление старого виртуального окружения: {venv_path}")
        shutil.rmtree(venv_path)
    
    # Определяем версию Python из команды
    python_version_str = "указанной версии"
    if python_cmd.startswith('py'):
        if '-3.12' in python_cmd:
            python_version_str = "3.12"
        elif '-3.14' in python_cmd:
            python_version_str = "3.14+"
    else:
        # Пробуем определить версию из команды
        try:
            result = subprocess.run([python_cmd, '--version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                python_version_str = result.stdout.strip()
        except:
            pass
    
    print(f"Создание виртуального окружения с Python {python_version_str}: {venv_path}")
    
    # Формируем команду для создания venv
    if python_cmd.startswith('py'):
        # Для Windows py launcher - используем команду как есть
        if '-3.12' in python_cmd:
            cmd = ['py', '-3.12', '-m', 'venv', str(venv_path)]
        elif '-3.14' in python_cmd:
            cmd = ['py', '-3.14', '-m', 'venv', str(venv_path)]
        else:
            cmd = python_cmd.split() + ['-m', 'venv', str(venv_path)]
    else:
        cmd = [python_cmd, '-m', 'venv', str(venv_path)]
    
    try:
        subprocess.run(cmd, check=True)
        print("[OK] Виртуальное окружение создано")
        return venv_path
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Ошибка создания виртуального окружения: {e}")
        return None

def install_dependencies(venv_path, project_root):
    """Устанавливает зависимости в виртуальное окружение"""
    if platform.system() == 'Windows':
        pip_cmd = venv_path / 'Scripts' / 'pip.exe'
        python_cmd = venv_path / 'Scripts' / 'python.exe'
    else:
        pip_cmd = venv_path / 'bin' / 'pip'
        python_cmd = venv_path / 'bin' / 'python'
    
    if not pip_cmd.exists():
        print(f"[ERROR] pip не найден в {venv_path}")
        return False
    
    print("\nОбновление pip...")
    try:
        subprocess.run([str(python_cmd), '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] Ошибка обновления pip: {e}")
    
    print("Установка зависимостей из requirements.txt...")
    requirements = project_root / 'requirements.txt'
    
    try:
        subprocess.run([str(pip_cmd), 'install', '-r', str(requirements)], check=True)
        print("[OK] Зависимости установлены")
        
        # Устанавливаем TensorFlow с GPU поддержкой
        print("\nУстановка TensorFlow с поддержкой GPU...")
        try:
            subprocess.run([str(pip_cmd), 'install', 'tensorflow[and-cuda]>=2.13.0'], check=True)
            print("[OK] TensorFlow с GPU установлен")
        except subprocess.CalledProcessError:
            print("[WARNING] Не удалось установить tensorflow[and-cuda], устанавливается базовая версия...")
            subprocess.run([str(pip_cmd), 'install', 'tensorflow>=2.13.0'], check=True)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Ошибка установки зависимостей: {e}")
        return False

def verify_gpu_setup(venv_path):
    """Проверяет настройку GPU в виртуальном окружении"""
    if platform.system() == 'Windows':
        python_cmd = venv_path / 'Scripts' / 'python.exe'
    else:
        python_cmd = venv_path / 'bin' / 'python'
    
    print("\n" + "=" * 80)
    print("ПРОВЕРКА НАСТРОЙКИ GPU")
    print("=" * 80)
    
    try:
        result = subprocess.run(
            [str(python_cmd), '-c', 
             'import tensorflow as tf; '
             'print(f"TensorFlow: {tf.__version__}"); '
             'print(f"CUDA built: {tf.test.is_built_with_cuda()}"); '
             'gpus = tf.config.list_physical_devices("GPU"); '
             'print(f"GPU devices: {len(gpus)}"); '
             '[print(f"  GPU {i}: {gpu.name}") for i, gpu in enumerate(gpus)]'],
            capture_output=True,
            text=True,
            timeout=30
        )
        print(result.stdout)
        if result.returncode == 0:
            print("[OK] Проверка завершена")
            return True
        else:
            print(f"[WARNING] Ошибка проверки: {result.stderr}")
            return False
    except Exception as e:
        print(f"[ERROR] Ошибка при проверке: {e}")
        return False

def main():
    """Главная функция"""
    print("=" * 80)
    print("НАСТРОЙКА PYTHON 3.12 ДЛЯ РАБОТЫ С GPU")
    print("=" * 80)
    
    project_root = Path(__file__).parent.parent
    
    # Проверяем текущую версию Python
    current_version = sys.version_info
    print(f"\nТекущая версия Python: {current_version.major}.{current_version.minor}.{current_version.micro}")
    
    # Проверяем текущую версию Python
    current_major, current_minor = current_version.major, current_version.minor
    
    # Если Python 3.14+, пробуем использовать его (может поддерживать GPU)
    if current_major == 3 and current_minor >= 14:
        has_python314, python314_cmd = check_python_314_available()
        if has_python314:
            print(f"[OK] Python 3.14+ найден: {python314_cmd}")
            print("[INFO] Пробуем использовать Python 3.14 для GPU (поддержка будет проверена)")
            python_cmd_to_use = python314_cmd
        else:
            # Пробуем использовать текущий Python 3.14
            python_cmd_to_use = sys.executable
            print(f"[INFO] Используется текущий Python {current_major}.{current_minor} для GPU")
    else:
        # Для Python 3.13 и ниже - используем Python 3.12
        has_python312, python312_cmd = check_python_312_available()
        
        if not has_python312:
            print("\n[ERROR] Python 3.12 не найден в системе")
            if platform.system() == 'Windows':
                install_python_312_windows()
            else:
                print("\nУстановите Python 3.12:")
                print("  Ubuntu/Debian: sudo apt install python3.12 python3.12-venv")
                print("  macOS: brew install python@3.12")
                print("  Или скачайте с https://www.python.org/downloads/")
            return 1
        
        print(f"[OK] Python 3.12 найден: {python312_cmd}")
        python_cmd_to_use = python312_cmd
    
    # Проверяем наличие GPU
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
        has_gpu = result.returncode == 0
    except:
        has_gpu = False
    
    if has_gpu:
        print("[OK] NVIDIA GPU обнаружен в системе")
    else:
        print("[WARNING] NVIDIA GPU не обнаружен, но продолжаем настройку...")
    
    # Создаем виртуальное окружение
    venv_path = create_venv_with_python(python_cmd_to_use, project_root)
    if not venv_path:
        return 1
    
    # Устанавливаем зависимости
    if not install_dependencies(venv_path, project_root):
        return 1
    
    # Проверяем настройку GPU
    verify_gpu_setup(venv_path)
    
    print("\n" + "=" * 80)
    print("НАСТРОЙКА ЗАВЕРШЕНА")
    print("=" * 80)
    print(f"\nВиртуальное окружение создано: {venv_path}")
    print("\nДля использования этого окружения:")
    if platform.system() == 'Windows':
        print(f"  {venv_path}\\Scripts\\activate")
        print(f"  python ai.py")
    else:
        print(f"  source {venv_path}/bin/activate")
        print(f"  python ai.py")
    print("\n" + "=" * 80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
