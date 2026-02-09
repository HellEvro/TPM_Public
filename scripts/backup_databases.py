#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для управления бэкапами баз данных AI и Bots

Использование:
    python scripts/backup_databases.py create              # Создать бэкап обеих БД
    python scripts/backup_databases.py create --ai-only    # Создать бэкап только AI БД
    python scripts/backup_databases.py create --bots-only   # Создать бэкап только Bots БД
    python scripts/backup_databases.py list                # Показать список бэкапов
    python scripts/backup_databases.py list --ai           # Показать список бэкапов AI БД
    python scripts/backup_databases.py list --bots          # Показать список бэкапов Bots БД
    python scripts/backup_databases.py restore <path>      # Восстановить из бэкапа
    python scripts/backup_databases.py delete <path>        # Удалить бэкап
    python scripts/backup_databases.py prune                # Оставить по 5 последних на систему
    python scripts/backup_databases.py cleanup              # Удалить старые бэкапы
    python scripts/backup_databases.py stats                # Показать статистику
"""

import sys
import os
import argparse
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot_engine.backup_service import get_backup_service
import logging

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BackupScript')


def format_size(size_mb: float) -> str:
    """Форматирует размер в читаемый вид"""
    if size_mb < 1:
        return f"{size_mb * 1024:.2f} KB"
    elif size_mb < 1024:
        return f"{size_mb:.2f} MB"
    else:
        return f"{size_mb / 1024:.2f} GB"


def print_backup_info(backup: dict):
    """Выводит информацию о бэкапе"""
    status = "[OK]" if backup.get('valid', True) else "[ERROR]"
    db_name = backup.get('db_name', 'unknown')
    size = format_size(backup['size_mb'])
    created_at = backup['created_at']
    
    print(f"  {status} {backup['filename']}")
    print(f"     БД: {db_name}")
    print(f"     Размер: {size}")
    print(f"     Создан: {created_at}")
    if not backup.get('valid', True):
        print(f"     [WARNING] Ошибка: {backup.get('error', 'Неизвестная ошибка')}")
    print()


def cmd_create(args):
    """Создает бэкап БД"""
    print("=" * 80)
    print("СОЗДАНИЕ БЭКАПА БАЗ ДАННЫХ")
    print("=" * 80)
    print()
    
    service = get_backup_service()
    
    include_ai = not args.bots_only
    include_bots = not args.ai_only
    
    if args.ai_only:
        print("Создание бэкапа AI БД...")
    elif args.bots_only:
        print("Создание бэкапа Bots БД...")
    else:
        print("Создание бэкапа обеих БД...")
    
    print()
    
    result = service.create_backup(
        include_ai=include_ai,
        include_bots=include_bots,
        max_retries=3,
        keep_last_n=5
    )
    
    # Проверяем, создан ли хотя бы один бэкап
    has_backups = result['backups']['ai'] is not None or result['backups']['bots'] is not None
    
    if has_backups:
        if result['errors']:
            print("[WARNING] Бэкап создан с предупреждениями!")
        else:
            print("[OK] Бэкап успешно создан!")
        print()
        print(f"Timestamp: {result['timestamp']}")
        print()
        
        if result['backups']['ai']:
            ai_backup = result['backups']['ai']
            print(f"AI БД:")
            print(f"   Путь: {ai_backup['path']}")
            print(f"   Размер: {format_size(ai_backup['size_mb'])}")
            print(f"   Целостность: {'[OK]' if ai_backup.get('valid', True) else '[ERROR]'}")
            print()
        elif include_ai:
            print(f"AI БД: [SKIP] БД не найдена")
            print()
        
        if result['backups']['bots']:
            bots_backup = result['backups']['bots']
            print(f"Bots БД:")
            print(f"   Путь: {bots_backup['path']}")
            print(f"   Размер: {format_size(bots_backup['size_mb'])}")
            print(f"   Целостность: {'[OK]' if bots_backup.get('valid', True) else '[ERROR]'}")
            print()
        elif include_bots:
            print(f"Bots БД: [SKIP] БД не найдена")
            print()
        
        if result['errors']:
            print("Предупреждения:")
            for error in result['errors']:
                print(f"   [WARNING] {error}")
            print()
    else:
        print("[ERROR] Ошибка создания бэкапа!")
        print()
        for error in result['errors']:
            print(f"   [ERROR] {error}")
        print()
        sys.exit(1)


def cmd_list(args):
    """Показывает список бэкапов"""
    print("=" * 80)
    print("СПИСОК БЭКАПОВ")
    print("=" * 80)
    print()
    
    service = get_backup_service()
    
    db_name = None
    if args.ai:
        db_name = 'ai_data'
    elif args.bots:
        db_name = 'bots_data'
    
    backups = service.list_backups(db_name=db_name)
    
    if not backups:
        print("[INFO] Бэкапы не найдены")
        print()
        return
    
    # Группируем по типу БД
    ai_backups = [b for b in backups if b.get('db_name') == 'ai_data']
    bots_backups = [b for b in backups if b.get('db_name') == 'bots_data']
    other_backups = [b for b in backups if b.get('db_name') not in ['ai_data', 'bots_data']]
    
    if ai_backups:
        print(f"AI БД бэкапы ({len(ai_backups)}):")
        print("-" * 80)
        for backup in ai_backups:
            print_backup_info(backup)
    
    if bots_backups:
        print(f"Bots БД бэкапы ({len(bots_backups)}):")
        print("-" * 80)
        for backup in bots_backups:
            print_backup_info(backup)
    
    if other_backups:
        print(f"Другие бэкапы ({len(other_backups)}):")
        print("-" * 80)
        for backup in other_backups:
            print_backup_info(backup)
    
    print(f"Всего бэкапов: {len(backups)}")
    print()


def cmd_restore(args):
    """Восстанавливает БД из бэкапа"""
    if not args.path:
        print("[ERROR] Ошибка: не указан путь к бэкапу")
        print("Использование: python scripts/backup_databases.py restore <path>")
        sys.exit(1)
    
    print("=" * 80)
    print("ВОССТАНОВЛЕНИЕ БД ИЗ БЭКАПА")
    print("=" * 80)
    print()
    
    backup_path = os.path.normpath(args.path)
    
    if not os.path.exists(backup_path):
        print(f"[ERROR] Бэкап не найден: {backup_path}")
        sys.exit(1)
    
    print(f"Восстановление из: {backup_path}")
    print()
    
    # Определяем имя БД
    filename = os.path.basename(backup_path)
    if filename.startswith('ai_data'):
        db_name = 'ai_data'
    elif filename.startswith('bots_data'):
        db_name = 'bots_data'
    else:
        db_name = None
        print("[WARNING] Не удалось определить тип БД из имени файла")
        print("Попытка автоматического определения...")
        print()
    
    service = get_backup_service()
    
    if service.restore_backup(backup_path, db_name=db_name):
        print()
        print("[OK] БД успешно восстановлена!")
    else:
        print()
        print("[ERROR] Ошибка восстановления БД")
        sys.exit(1)


def cmd_delete(args):
    """Удаляет бэкап"""
    if not args.path:
        print("[ERROR] Ошибка: не указан путь к бэкапу")
        print("Использование: python scripts/backup_databases.py delete <path>")
        sys.exit(1)
    
    print("=" * 80)
    print("УДАЛЕНИЕ БЭКАПА")
    print("=" * 80)
    print()
    
    backup_path = os.path.normpath(args.path)
    
    if not os.path.exists(backup_path):
        print(f"[ERROR] Бэкап не найден: {backup_path}")
        sys.exit(1)
    
    # Подтверждение
    if not args.force:
        response = input(f"Вы уверены, что хотите удалить бэкап {backup_path}? (yes/no): ")
        if response.lower() not in ['yes', 'y', 'да', 'д']:
            print("Отменено")
            return
    
    service = get_backup_service()
    
    if service.delete_backup(backup_path):
        print()
        print("[OK] Бэкап успешно удален!")
    else:
        print()
        print("[ERROR] Ошибка удаления бэкапа")
        sys.exit(1)


def cmd_prune(args):
    """Оставляет только N последних бэкапов для каждой системы (AI, Bots), остальные удаляет."""
    print("=" * 80)
    print("ОЧИСТКА ЛИШНИХ БЭКАПОВ (оставить по N последних на систему)")
    print("=" * 80)
    print()
    service = get_backup_service()
    print(f"Оставляем по {args.keep} последних бэкапов для каждой БД (AI и Bots)...")
    print()
    result = service.cleanup_excess_backups(keep_count=args.keep)
    print()
    print("Результаты:")
    print(f"   AI БД: удалено {result['ai_data']} бэкапов")
    print(f"   Bots БД: удалено {result['bots_data']} бэкапов")
    print(f"   Всего: удалено {result['total']} бэкапов")
    print()


def cmd_cleanup(args):
    """Удаляет старые бэкапы (по возрасту и с учётом минимума на тип)"""
    print("=" * 80)
    print("ОЧИСТКА СТАРЫХ БЭКАПОВ")
    print("=" * 80)
    print()
    
    service = get_backup_service()
    
    print(f"Удаление бэкапов старше {args.days} дней...")
    print(f"Сохранение минимум {args.keep} бэкапов каждого типа...")
    print()
    
    result = service.cleanup_old_backups(days=args.days, keep_count=args.keep)
    
    print()
    print("Результаты очистки:")
    print(f"   AI БД: удалено {result['ai_data']} бэкапов")
    print(f"   Bots БД: удалено {result['bots_data']} бэкапов")
    print(f"   Всего: удалено {result['total']} бэкапов")
    print()


def cmd_stats(args):
    """Показывает статистику по бэкапам"""
    print("=" * 80)
    print("СТАТИСТИКА БЭКАПОВ")
    print("=" * 80)
    print()
    
    service = get_backup_service()
    stats = service.get_backup_stats()
    
    print(f"Общая статистика:")
    print(f"   Всего бэкапов: {stats['total_backups']}")
    print(f"   Общий размер: {format_size(stats['total_size_mb'])}")
    print(f"   Невалидных бэкапов: {stats['invalid_backups']}")
    print()
    
    print(f"AI БД:")
    print(f"   Количество: {stats['ai_data_backups']}")
    print(f"   Размер: {format_size(stats['ai_data_size_mb'])}")
    print()
    
    print(f"Bots БД:")
    print(f"   Количество: {stats['bots_data_backups']}")
    print(f"   Размер: {format_size(stats['bots_data_size_mb'])}")
    print()
    
    if stats['oldest_backup']:
        print(f"Самый старый бэкап: {stats['oldest_backup']}")
    if stats['newest_backup']:
        print(f"Самый новый бэкап: {stats['newest_backup']}")
    print()


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description='Сервис управления бэкапами баз данных AI и Bots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python scripts/backup_databases.py create
  python scripts/backup_databases.py create --ai-only
  python scripts/backup_databases.py list
  python scripts/backup_databases.py restore data/backups/ai_data_20240101_120000.db
  python scripts/backup_databases.py delete data/backups/ai_data_20240101_120000.db
  python scripts/backup_databases.py prune              # Оставить по 5 последних на систему (рекомендуется)
  python scripts/backup_databases.py prune --keep 5
  python scripts/backup_databases.py cleanup --days 30 --keep 5
  python scripts/backup_databases.py stats
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Команда')
    
    # Команда create
    create_parser = subparsers.add_parser('create', help='Создать бэкап БД')
    create_group = create_parser.add_mutually_exclusive_group()
    create_group.add_argument('--ai-only', action='store_true', help='Создать бэкап только AI БД')
    create_group.add_argument('--bots-only', action='store_true', help='Создать бэкап только Bots БД')
    
    # Команда list
    list_parser = subparsers.add_parser('list', help='Показать список бэкапов')
    list_group = list_parser.add_mutually_exclusive_group()
    list_group.add_argument('--ai', action='store_true', help='Показать только бэкапы AI БД')
    list_group.add_argument('--bots', action='store_true', help='Показать только бэкапы Bots БД')
    
    # Команда restore
    restore_parser = subparsers.add_parser('restore', help='Восстановить БД из бэкапа')
    restore_parser.add_argument('path', nargs='?', help='Путь к файлу бэкапа')
    
    # Команда delete
    delete_parser = subparsers.add_parser('delete', help='Удалить бэкап')
    delete_parser.add_argument('path', nargs='?', help='Путь к файлу бэкапа')
    delete_parser.add_argument('--force', action='store_true', help='Удалить без подтверждения')
    
    # Команда prune — оставить только N последних на систему (рекомендуемая политика)
    prune_parser = subparsers.add_parser('prune', help='Оставить только N последних бэкапов для каждой БД')
    prune_parser.add_argument('--keep', type=int, default=5, help='Сколько последних бэкапов хранить (по умолчанию: 5)')
    
    # Команда cleanup
    cleanup_parser = subparsers.add_parser('cleanup', help='Удалить старые бэкапы (по возрасту)')
    cleanup_parser.add_argument('--days', type=int, default=30, help='Удалять бэкапы старше N дней (по умолчанию: 30)')
    cleanup_parser.add_argument('--keep', type=int, default=5, help='Минимум бэкапов для сохранения (по умолчанию: 5)')
    
    # Команда stats
    subparsers.add_parser('stats', help='Показать статистику по бэкапам')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'create':
            cmd_create(args)
        elif args.command == 'list':
            cmd_list(args)
        elif args.command == 'restore':
            cmd_restore(args)
        elif args.command == 'delete':
            cmd_delete(args)
        elif args.command == 'prune':
            cmd_prune(args)
        elif args.command == 'cleanup':
            cmd_cleanup(args)
        elif args.command == 'stats':
            cmd_stats(args)
    except KeyboardInterrupt:
        print("\n\n[WARNING] Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        logger.exception("Ошибка выполнения команды")
        print(f"\n[ERROR] Ошибка: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

