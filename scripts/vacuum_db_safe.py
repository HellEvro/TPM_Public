#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Безопасный VACUUM для больших БД с проверками
"""

import sys
import os
from pathlib import Path
import sqlite3
import time

# Настройка кодировки для Windows консоли
if os.name == 'nt':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def vacuum_db_safe(db_path: str):
    """Безопасный VACUUM с проверками"""
    print("=" * 80)
    print(f"БЕЗОПАСНЫЙ VACUUM БД: {db_path}")
    print("=" * 80)
    
    if not Path(db_path).exists():
        print(f"❌ Файл не найден: {db_path}")
        return False
    
    db_size_before = Path(db_path).stat().st_size
    print(f"\n📊 Размер БД до VACUUM: {db_size_before / (1024**3):.2f} GB")
    
    # Проверяем, не заблокирована ли БД
    try:
        test_conn = sqlite3.connect(str(db_path), timeout=5.0)
        test_conn.close()
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e).lower():
            print(f"\n❌ БД заблокирована другим процессом!")
            print(f"   Закройте все программы, использующие эту БД:")
            print(f"   - bots.py")
            print(f"   - ai.py")
            print(f"   - database_gui.py")
            print(f"   - Другие скрипты")
            return False
    
    # Проверяем свободное место
    try:
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        cursor = conn.cursor()
        cursor.execute("PRAGMA page_count")
        page_count = cursor.fetchone()[0]
        cursor.execute("PRAGMA freelist_count")
        freelist_count = cursor.fetchone()[0]
        cursor.execute("PRAGMA page_size")
        page_size = cursor.fetchone()[0]
        conn.close()
        
        free_size = freelist_count * page_size
        free_percent = (freelist_count / page_count * 100) if page_count > 0 else 0
        
        print(f"\n📄 АНАЛИЗ БД:")
        print(f"   Всего страниц: {page_count:,}")
        print(f"   Свободных страниц: {freelist_count:,} ({free_size / (1024**3):.2f} GB)")
        print(f"   Процент свободного места: {free_percent:.1f}%")
        
        if free_percent < 5:
            print(f"\n⚠️ Мало свободного места ({free_percent:.1f}%)")
            print(f"   VACUUM может не дать значительного эффекта")
            print(f"   💡 Рекомендуется сначала запустить очистку свечей")
        
        if db_size_before > 10 * 1024**3:  # >10 GB
            print(f"\n⚠️ БД очень большая ({db_size_before / (1024**3):.2f} GB)")
            print(f"   VACUUM может занять ОЧЕНЬ много времени (возможно, часы)")
            print(f"   ⏳ Начинаю выполнение...")
        
    except Exception as e:
        print(f"⚠️ Ошибка при проверке БД: {e}")
        print(f"   ⏳ Продолжаю выполнение VACUUM...")
    
    # Выполняем checkpoint перед VACUUM
    print(f"\n⏳ [1/3] Выполнение PRAGMA wal_checkpoint(TRUNCATE)...")
    try:
        conn = sqlite3.connect(str(db_path), timeout=60.0)
        cursor = conn.cursor()
        cursor.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()
        print(f"   ✅ Checkpoint выполнен")
    except Exception as e:
        print(f"   ⚠️ Ошибка checkpoint: {e}")
    
    # Выполняем VACUUM
    print(f"\n⏳ [2/3] Выполнение VACUUM (это может занять МНОГО времени)...")
    print(f"   💡 Не закрывайте это окно!")
    print(f"   💡 Можно следить за размером файла БД в проводнике")
    
    start_time = time.time()
    
    try:
        # Увеличиваем timeout для VACUUM
        conn = sqlite3.connect(str(db_path), timeout=3600.0)  # 1 час
        cursor = conn.cursor()
        
        # Выполняем VACUUM
        cursor.execute("VACUUM")
        conn.close()
        
        end_time = time.time()
        elapsed_minutes = (end_time - start_time) / 60
        
        # Проверяем размер после VACUUM
        db_size_after = Path(db_path).stat().st_size
        freed_size = db_size_before - db_size_after
        
        print(f"\n✅ [3/3] VACUUM завершен!")
        print(f"   Время выполнения: {elapsed_minutes:.1f} минут")
        print(f"   Размер БД после VACUUM: {db_size_after / (1024**3):.2f} GB")
        print(f"   Освобождено места: {freed_size / (1024**3):.2f} GB ({freed_size / (1024**2):.2f} MB)")
        print(f"   Уменьшение размера: {(freed_size / db_size_before * 100) if db_size_before > 0 else 0:.1f}%")
        
        return True
        
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e).lower():
            print(f"\n❌ БД была заблокирована во время VACUUM!")
            print(f"   Закройте все программы и попробуйте снова")
        else:
            print(f"\n❌ Ошибка при выполнении VACUUM: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("=" * 80)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Безопасный VACUUM для больших БД')
    parser.add_argument('db_path', nargs='?', help='Путь к БД')
    args = parser.parse_args()
    
    if args.db_path:
        db_path = args.db_path
    else:
        db_path = os.environ.get('BOTS_DB_PATH')
        if not db_path:
            db_path = str(PROJECT_ROOT / 'data' / 'bots_data.db')
    
    vacuum_db_safe(db_path)

