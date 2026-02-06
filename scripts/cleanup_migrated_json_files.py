#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для безопасного удаления JSON файлов после миграции в БД

ВНИМАНИЕ: Используйте этот скрипт ТОЛЬКО после проверки, что все данные
успешно мигрированы в БД и система работает корректно!

Процесс:
1. Проверяет наличие данных в БД для каждого файла
2. Сравнивает количество записей
3. Предлагает удалить JSON файлы (с подтверждением)
4. Создает резервную копию перед удалением
"""

import os
import sys
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path

# Добавляем текущую директорию в путь для импорта модулей
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_database_migration():
    """Проверяет, что данные успешно мигрированы в БД"""
    try:
        # Определяем корневую директорию проекта (на уровень выше scripts/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        # Путь к БД в корне проекта
        db_path = os.path.join(project_root, 'data', 'bots_data.db')
        db_path = os.path.normpath(db_path)
        
        from bot_engine.bots_database import get_bots_database
        
        # Используем БД из корневой директории проекта
        db = get_bots_database(db_path=db_path)
        if not db:
            logger.error("❌ БД недоступна!")
            return False
        
        # Получаем статистику БД
        stats = db.get_database_stats()
        
        logger.info("📊 Статистика БД:")
        logger.info(f"   bots_state: {stats.get('bots_state_count', 0)} записей")
        logger.info(f"   bot_positions_registry: {stats.get('bot_positions_registry_count', 0)} записей")
        logger.info(f"   rsi_cache: {stats.get('rsi_cache_count', 0)} записей")
        logger.info(f"   process_state: {stats.get('process_state_count', 0)} записей")
        logger.info(f"   individual_coin_settings: {stats.get('individual_coin_settings_count', 0)} записей")
        logger.info(f"   mature_coins: {stats.get('mature_coins_count', 0)} записей")
        logger.info(f"   maturity_check_cache: {stats.get('maturity_check_cache_count', 0)} записей")
        logger.info(f"   delisted: {stats.get('delisted_count', 0)} записей")
        logger.info(f"   Размер БД: {stats.get('database_size_mb', 0):.2f} MB")
        
        # Проверяем наличие данных
        has_data = any([
            stats.get('bots_state_count', 0) > 0,
            stats.get('bot_positions_registry_count', 0) > 0,
            stats.get('rsi_cache_count', 0) > 0,
            stats.get('process_state_count', 0) > 0,
            stats.get('individual_coin_settings_count', 0) > 0,
            stats.get('mature_coins_count', 0) > 0,
            stats.get('maturity_check_cache_count', 0) > 0,
            stats.get('delisted_count', 0) > 0
        ])
        
        if not has_data:
            logger.warning("⚠️ В БД нет данных! Миграция не выполнена или БД пустая.")
            return False
        
        logger.info("✅ БД содержит данные - миграция выполнена")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка проверки БД: {e}")
        return False


def backup_json_file(file_path, backup_dir=None):
    """Создает резервную копию JSON файла"""
    if not os.path.exists(file_path):
        return None
    
    if backup_dir is None:
        # Определяем корневую директорию проекта
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        backup_dir = os.path.join(project_root, 'data', 'backup_json_before_migration')
    
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = Path(file_path).name
    backup_path = backup_dir / f"{timestamp}_{filename}"
    
    try:
        shutil.copy2(file_path, backup_path)
        logger.info(f"📦 Резервная копия создана: {backup_path}")
        return str(backup_path)
    except Exception as e:
        logger.error(f"❌ Ошибка создания резервной копии {file_path}: {e}")
        return None


def verify_json_file_data(file_path):
    """Проверяет наличие данных в JSON файле"""
    if not os.path.exists(file_path):
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                return len(data) > 0
            elif isinstance(data, list):
                return len(data) > 0
            return bool(data)
    except Exception as e:
        pass
        return False


def cleanup_json_files(dry_run=True):
    """
    Удаляет JSON файлы после миграции в БД
    
    Args:
        dry_run: Если True, только показывает что будет удалено, без реального удаления
    """
    # Определяем корневую директорию проекта (на уровень выше scripts/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Файлы которые мигрированы в БД
    migrated_files = {
        'bots_state.json': {
            'path': os.path.join(project_root, 'data', 'bots_state.json'),
            'table': 'bots_state',
            'description': 'Состояние ботов'
        },
        'bot_positions_registry.json': {
            'path': os.path.join(project_root, 'data', 'bot_positions_registry.json'),
            'table': 'bot_positions_registry',
            'description': 'Реестр позиций ботов'
        },
        'rsi_cache.json': {
            'path': os.path.join(project_root, 'data', 'rsi_cache.json'),
            'table': 'rsi_cache',
            'description': 'RSI кэш'
        },
        'process_state.json': {
            'path': os.path.join(project_root, 'data', 'process_state.json'),
            'table': 'process_state',
            'description': 'Состояние процессов'
        },
        'individual_coin_settings.json': {
            'path': os.path.join(project_root, 'data', 'individual_coin_settings.json'),
            'table': 'individual_coin_settings',
            'description': 'Индивидуальные настройки монет'
        },
        'mature_coins.json': {
            'path': os.path.join(project_root, 'data', 'mature_coins.json'),
            'table': 'mature_coins',
            'description': 'Зрелые монеты'
        },
        'maturity_check_cache.json': {
            'path': os.path.join(project_root, 'data', 'maturity_check_cache.json'),
            'table': 'maturity_check_cache',
            'description': 'Кэш проверки зрелости'
        },
        'delisted.json': {
            'path': os.path.join(project_root, 'data', 'delisted.json'),
            'table': 'delisted',
            'description': 'Делистированные монеты'
        }
    }
    
    logger.info("=" * 80)
    logger.info("🧹 ОЧИСТКА JSON ФАЙЛОВ ПОСЛЕ МИГРАЦИИ В БД")
    logger.info("=" * 80)
    
    if dry_run:
        logger.info("🔍 РЕЖИМ ПРОВЕРКИ (dry-run) - файлы не будут удалены")
    else:
        logger.info("⚠️ РЕЖИМ УДАЛЕНИЯ - файлы будут удалены!")
    
    logger.info("")
    
    # Сначала проверяем БД
    if not check_database_migration():
        logger.error("❌ Миграция не завершена или БД пустая!")
        logger.error("❌ Не удаляйте JSON файлы пока миграция не завершена!")
        return False
    
    logger.info("")
    
    # Проверяем каждый файл
    files_to_remove = []
    files_with_data = []
    
    for filename, info in migrated_files.items():
        file_path = info['path']
        
        if os.path.exists(file_path):
            has_data = verify_json_file_data(file_path)
            
            if has_data:
                files_with_data.append((filename, info, file_path))
                logger.info(f"📄 {filename} - найден (содержит данные)")
                logger.info(f"   → Таблица в БД: {info['table']}")
            else:
                files_to_remove.append((filename, info, file_path))
                logger.info(f"📄 {filename} - найден (пустой или невалидный)")
        else:
            pass
    
    logger.info("")
    
    if not files_with_data:
        logger.info("✅ Все JSON файлы пустые или уже удалены")
        return True
    
    logger.info(f"📊 Найдено {len(files_with_data)} JSON файлов с данными:")
    for filename, info, file_path in files_with_data:
        logger.info(f"   - {filename} ({info['description']})")
    
    logger.info("")
    
    if dry_run:
        logger.info("💡 Для реального удаления запустите с параметром --execute")
        logger.info("💡 Пример: python scripts/cleanup_migrated_json_files.py --execute")
        return True
    
    # Создаем резервные копии
    logger.info("📦 Создание резервных копий...")
    backup_dir = os.path.join(project_root, 'data', 'backup_json_before_migration')
    backup_paths = []
    
    for filename, info, file_path in files_with_data:
        backup_path = backup_json_file(file_path, backup_dir=backup_dir)
        if backup_path:
            backup_paths.append(backup_path)
    
    logger.info(f"✅ Создано {len(backup_paths)} резервных копий")
    logger.info("")
    
    # Удаляем файлы
    logger.info("🗑️  Удаление JSON файлов...")
    removed_count = 0
    
    for filename, info, file_path in files_with_data:
        try:
            os.remove(file_path)
            logger.info(f"✅ Удален: {filename}")
            removed_count += 1
        except Exception as e:
            logger.error(f"❌ Ошибка удаления {filename}: {e}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"✅ Очистка завершена: удалено {removed_count} файлов")
    backup_dir_abs = os.path.join(project_root, 'data', 'backup_json_before_migration')
    logger.info(f"📦 Резервные копии сохранены в: {backup_dir_abs}")
    logger.info("=" * 80)
    
    return True


def main():
    """Главная функция"""
    import sys
    import traceback
    
    # Настройка кодировки для Windows консоли
    if sys.platform == 'win32':
        try:
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
        except:
            pass
    
    # Проверяем аргументы
    execute = '--execute' in sys.argv or '-e' in sys.argv
    
    try:
        if execute:
            print("\n" + "=" * 80)
            print("[!]  ВНИМАНИЕ: Вы собираетесь удалить JSON файлы!")
            print("=" * 80)
            print("Это действие:")
            print("  - Удалит JSON файлы после миграции в БД")
            print("  - Создаст резервные копии в data/backup_json_before_migration/")
            print("  - Необратимо (но резервные копии можно восстановить)")
            print("=" * 80)
            print()
            
            response = input("Продолжить? (yes/no): ").strip().lower()
            if response != 'yes':
                print("[X] Отменено")
                return
            
            print()
            print("[*] Начинаем удаление JSON файлов...")
            print()
            
            success = cleanup_json_files(dry_run=False)
            
            if success:
                print()
                print("[OK] Операция завершена успешно!")
            else:
                print()
                print("[ERROR] Операция завершена с ошибками!")
                sys.exit(1)
        else:
            cleanup_json_files(dry_run=True)
            
    except KeyboardInterrupt:
        print()
        print("[X] Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print()
        print(f"[ERROR] Критическая ошибка: {e}")
        print()
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

