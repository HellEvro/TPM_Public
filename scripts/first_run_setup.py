#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт автоматической настройки для первого запуска на новом ПК
Создает .venv, устанавливает зависимости, создает конфиг файлы
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_step(message):
    """Выводит сообщение о шаге"""
    # Используем ASCII символы для совместимости с Windows CMD
    print(f"\n{'='*60}")
    print(f"  {message}")
    print(f"{'='*60}")

def check_python():
    """Проверяет наличие Python"""
    try:
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print(f"[ERROR] Требуется Python 3.8 или выше. Текущая версия: {version.major}.{version.minor}")
            return False
        print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")
        return True
    except Exception as e:
        print(f"[ERROR] Ошибка проверки Python: {e}")
        return False

def create_venv():
    """Создает виртуальное окружение .venv"""
    venv_dir = Path(".venv")
    
    if venv_dir.exists():
        print("[INFO] Виртуальное окружение .venv уже существует")
        return True
    
    print_step("Создание виртуального окружения .venv")
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            check=True,
            capture_output=True,
            text=True
        )
        print("[OK] Виртуальное окружение создано")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Не удалось создать .venv: {e.stderr}")
        return False

def get_venv_python():
    """Возвращает путь к Python в .venv"""
    if os.name == "nt":  # Windows
        return Path(".venv") / "Scripts" / "python.exe"
    else:  # Linux/macOS
        return Path(".venv") / "bin" / "python"

def install_dependencies():
    """Устанавливает зависимости в .venv"""
    venv_python = get_venv_python()
    
    if not venv_python.exists():
        print("[ERROR] Python в .venv не найден!")
        return False
    
    print_step("Установка зависимостей в .venv")
    
    try:
        # Обновляем pip
        print("[INFO] Обновление pip...")
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
            check=True,
            capture_output=True,
            text=True
        )
        print("[OK] pip обновлен")
        
        # Устанавливаем зависимости
        print("[INFO] Установка зависимостей из requirements.txt...")
        print("       Это может занять несколько минут...")
        
        result = subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-r", "requirements.txt"],
            check=False,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"[ERROR] Ошибка установки зависимостей:")
            print(result.stderr)
            return False
        
        print("[OK] Зависимости установлены")
        return True
        
    except Exception as e:
        print(f"[ERROR] Ошибка при установке зависимостей: {e}")
        return False

def create_config_files():
    """Создает конфиг файлы из примеров"""
    print_step("Создание конфигурационных файлов")
    
    configs = [
        ("app/config.example.py", "app/config.py"),
        ("app/keys.example.py", "app/keys.py"),
    ]
    
    created = []
    for example, target in configs:
        example_path = Path(example)
        target_path = Path(target)
        
        if target_path.exists():
            print(f"[INFO] {target} уже существует, пропускаем")
            continue
        
        if not example_path.exists():
            print(f"[WARN] {example} не найден, пропускаем {target}")
            continue
        
        try:
            # Копируем файл
            shutil.copy2(example_path, target_path)
            
            # Удаляем заголовок из config.py
            if target.endswith("config.py"):
                try:
                    content = target_path.read_text(encoding='utf-8')
                    if '"""' in content:
                        # Удаляем docstring в начале
                        lines = content.split('\n')
                        new_lines = []
                        skip_docstring = False
                        for line in lines:
                            if line.strip().startswith('"""') and not skip_docstring:
                                if '"""' in line and line.count('"""') >= 2:
                                    # Однострочный docstring
                                    continue
                                skip_docstring = True
                                continue
                            if skip_docstring and '"""' in line:
                                skip_docstring = False
                                continue
                            if not skip_docstring:
                                new_lines.append(line)
                        target_path.write_text('\n'.join(new_lines), encoding='utf-8')
                except Exception as e:
                    print(f"[WARN] Не удалось очистить заголовок в {target}: {e}")
            
            print(f"[OK] Создан {target}")
            created.append(target)
        except Exception as e:
            print(f"[ERROR] Не удалось создать {target}: {e}")
    
    if created:
        print(f"[OK] Создано файлов: {len(created)}")
    else:
        print("[INFO] Все конфиг файлы уже существуют")
    
    return True

def create_directories():
    """Создает необходимые директории"""
    dirs = ["logs", "data"]
    
    for dirname in dirs:
        dirpath = Path(dirname)
        if not dirpath.exists():
            try:
                dirpath.mkdir(parents=True, exist_ok=True)
                print(f"[OK] Создана директория {dirname}/")
            except Exception as e:
                print(f"[WARN] Не удалось создать {dirname}/: {e}")

def main():
    """Главная функция"""
    # Используем ASCII символы для совместимости с Windows CMD
    print("\n" + "="*60)
    print("  AUTOMATIC SETUP FOR FIRST RUN")
    print("="*60)
    
    # Проверка Python
    if not check_python():
        return 1
    
    # Создание директорий
    create_directories()
    
    # Создание .venv
    if not create_venv():
        return 1
    
    # Установка зависимостей
    if not install_dependencies():
        print("\n[WARN] Не удалось установить зависимости автоматически.")
        print("       Попробуйте запустить вручную:")
        venv_python = get_venv_python()
        if os.name == "nt":
            print(f"       .venv\\Scripts\\python.exe -m pip install -r requirements.txt")
        else:
            print(f"       .venv/bin/python -m pip install -r requirements.txt")
        return 1
    
    # Создание конфиг файлов
    create_config_files()
    
    print("\n" + "="*60)
    print("  ✅ НАСТРОЙКА ЗАВЕРШЕНА УСПЕШНО!")
    print("="*60)
    print("\nТеперь вы можете запустить:")
    print("  - start_infobot_manager.bat (рекомендуется)")
    print("  - python app.py")
    print("  - python bots.py")
    print("\n")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n[INFO] Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

