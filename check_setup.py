#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт диагностики для проверки готовности системы к запуску
"""

import os
import sys
import shutil
from pathlib import Path

def check_file(filepath, description):
    """Проверяет наличие файла"""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {filepath}")
    if not exists:
        print(f"   → Файл не найден!")
    return exists

def check_python_version():
    """Проверяет версию Python"""
    version = sys.version_info
    print(f"Python версия: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   → Требуется Python 3.8 или выше!")
        return False
    return True

def check_git():
    """Проверяет наличие Git и выводит инструкции по установке"""
    git_exists = shutil.which("git") is not None
    status = "✓" if git_exists else "⚠"
    print(f"{status} Git")
    if not git_exists:
        print("   → Git не установлен (не критично, но нужен для обновлений)")
        print("\n   Инструкции по установке:")
        if os.name == 'nt':  # Windows
            print("   Windows:")
            print("     1. Скачайте Git с https://git-scm.com/download/win")
            print("     2. Запустите установщик и следуйте инструкциям")
            print("     3. Перезапустите командную строку после установки")
        elif sys.platform == 'darwin':  # macOS
            print("   macOS:")
            print("     Вариант 1 (через Homebrew):")
            print("       brew install git")
            print("     Вариант 2 (через Xcode Command Line Tools):")
            print("       xcode-select --install")
        else:  # Linux
            print("   Linux:")
            print("     Ubuntu/Debian:")
            print("       sudo apt-get update && sudo apt-get install git")
            print("     Fedora/RHEL:")
            print("       sudo dnf install git")
            print("     Arch Linux:")
            print("       sudo pacman -S git")
    return git_exists

def check_imports():
    """Проверяет возможность импорта основных модулей"""
    print("\nПроверка импортов:")
    modules = [
        ('flask', 'Flask'),
        ('flask_cors', 'CORS'),
        ('requests', 'requests'),
        ('ccxt', 'ccxt'),
    ]
    all_ok = True
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError as e:
            print(f"  ✗ {display_name}: {e}")
            all_ok = False
    return all_ok

def main():
    print("=" * 80)
    print("Диагностика InfoBot")
    print("=" * 80)
    
    # Проверка Python
    print("\n1. Проверка Python:")
    python_ok = check_python_version()
    
    # Проверка файлов
    print("\n2. Проверка файлов конфигурации:")
    config_ok = check_file('app/config.py', 'Конфигурация')
    keys_ok = check_file('app/keys.py', 'API ключи')
    config_example_ok = check_file('app/config.example.py', 'Пример конфигурации')
    
    # Проверка Git
    print("\n3. Проверка Git (опционально, для обновлений):")
    git_ok = check_git()
    
    # Проверка директорий
    print("\n4. Проверка директорий:")
    dirs_ok = True
    for dirname in ['logs', 'data', 'static', 'templates']:
        exists = os.path.exists(dirname)
        status = "✓" if exists else "✗"
        print(f"{status} {dirname}/")
        if not exists:
            try:
                os.makedirs(dirname, exist_ok=True)
                print(f"   → Создана директория {dirname}/")
            except Exception as e:
                print(f"   → Ошибка создания: {e}")
                dirs_ok = False
    
    # Проверка импортов
    print("\n5. Проверка импортов:")
    imports_ok = check_imports()
    
    # Итоги
    print("\n" + "=" * 80)
    print("ИТОГИ:")
    print("=" * 80)
    
    if not python_ok:
        print("✗ Python версия не соответствует требованиям")
        return 1
    
    if not config_ok:
        print("✗ Файл app/config.py отсутствует")
        if config_example_ok:
            print("\n  Решение:")
            print("  Скопируйте app/config.example.py в app/config.py:")
            if os.name == 'nt':
                print("    copy app\\config.example.py app\\config.py")
            else:
                print("    cp app/config.example.py app/config.py")
        return 1
    
    if not keys_ok:
        print("⚠ Файл app/keys.py отсутствует (не критично для первого запуска)")
    
    if not git_ok:
        print("⚠ Git не установлен (не критично, но рекомендуется для обновлений)")
    
    if not imports_ok:
        print("✗ Некоторые модули не установлены")
        print("\n  Решение:")
        print("  Установите зависимости:")
        print("    pip install -r requirements.txt")
        print("  Или используйте менеджер:")
        print("    start_infobot_manager.bat")
        return 1
    
    print("✓ Все проверки пройдены! Система готова к запуску.")
    print("\n  Запуск:")
    print("    start_infobot_manager.bat")
    print("  или")
    print("    python app.py")
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nПрервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nОшибка при диагностике: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

