#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Детальная проверка удаленной БД - что реально занимает место
"""

import sys
import os
from pathlib import Path
import sqlite3

# Настройка кодировки для Windows консоли
if os.name == 'nt':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def check_remote_db(db_path: str):
    """Детальная проверка удаленной БД"""
    print("=" * 80)
    print(f"ДЕТАЛЬНАЯ ПРОВЕРКА БД: {db_path}")
    print("=" * 80)
    
    if not Path(db_path).exists():
        print(f"❌ Файл не найден: {db_path}")
        return
    
    # Размеры файлов
    db_file = Path(db_path)
    wal_file = Path(str(db_path) + '-wal')
    shm_file = Path(str(db_path) + '-shm')
    
    db_size = db_file.stat().st_size if db_file.exists() else 0
    wal_size = wal_file.stat().st_size if wal_file.exists() else 0
    shm_size = shm_file.stat().st_size if shm_file.exists() else 0
    
    print(f"\n📊 РАЗМЕРЫ ФАЙЛОВ:")
    print(f"   Основной файл: {db_size / (1024**3):.2f} GB ({db_size / (1024**2):.2f} MB)")
    if wal_file.exists() and wal_size > 0:
        print(f"   WAL файл: {wal_size / (1024**3):.2f} GB ({wal_size / (1024**2):.2f} MB) ⚠️")
    if shm_file.exists() and shm_size > 0:
        print(f"   SHM файл: {shm_size / (1024**2):.2f} MB")
    print(f"   ИТОГО: {(db_size + wal_size + shm_size) / (1024**3):.2f} GB")
    
    # Подключение к БД
    try:
        print(f"\n⏳ Подключение к БД (timeout=60 сек)...")
        conn = sqlite3.connect(str(db_path), timeout=60.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Информация о БД
        cursor.execute("PRAGMA page_size")
        page_size = cursor.fetchone()[0]
        cursor.execute("PRAGMA page_count")
        page_count = cursor.fetchone()[0]
        cursor.execute("PRAGMA freelist_count")
        freelist_count = cursor.fetchone()[0]
        
        print(f"\n📄 ПАРАМЕТРЫ БД:")
        print(f"   Размер страницы: {page_size} байт")
        print(f"   Всего страниц: {page_count:,}")
        print(f"   Свободных страниц: {freelist_count:,}")
        print(f"   Используется страниц: {page_count - freelist_count:,}")
        print(f"   Размер используемых страниц: {(page_count - freelist_count) * page_size / (1024**3):.2f} GB")
        print(f"   Размер свободного места: {freelist_count * page_size / (1024**3):.2f} GB")
        
        # Получаем список таблиц
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        
        print(f"\n📊 АНАЛИЗ ТАБЛИЦ:")
        print("=" * 80)
        
        table_stats = []
        total_rows = 0
        
        for table in tables:
            try:
                print(f"   ⏳ Проверка {table}...", end='', flush=True)
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                total_rows += row_count
                
                # Приблизительная оценка размера на основе типа таблицы
                if 'candles_cache_data' in table:
                    # id (8) + cache_id (8) + time (8) + open/high/low/close (32) + volume (8) = 64 байт + overhead
                    estimated_size = row_count * 70
                elif 'candles_cache' in table:
                    estimated_size = row_count * 250
                elif 'trades' in table or 'history' in table:
                    estimated_size = row_count * 500  # Много полей
                elif 'cache' in table:
                    estimated_size = row_count * 300
                elif 'rsi' in table:
                    estimated_size = row_count * 150
                else:
                    estimated_size = row_count * 200
                
                table_stats.append({
                    'name': table,
                    'row_count': row_count,
                    'estimated_size': estimated_size
                })
                print(f" ✅ {row_count:,} записей")
            except Exception as e:
                print(f" ❌ Ошибка: {e}")
        
        # Сортируем по размеру
        table_stats.sort(key=lambda x: x['estimated_size'], reverse=True)
        
        print(f"\n📊 ТАБЛИЦЫ ПО РАЗМЕРУ (приблизительно):")
        print("=" * 80)
        total_estimated = 0
        for stat in table_stats:
            size_gb = stat['estimated_size'] / (1024**3)
            size_mb = stat['estimated_size'] / (1024**2)
            total_estimated += stat['estimated_size']
            
            if size_gb > 0.1:
                print(f"{stat['name']:35} {stat['row_count']:>15,} записей  {size_gb:>8.2f} GB")
            elif size_mb > 1:
                print(f"{stat['name']:35} {stat['row_count']:>15,} записей  {size_mb:>8.2f} MB")
            else:
                print(f"{stat['name']:35} {stat['row_count']:>15,} записей")
        
        print("=" * 80)
        print(f"ИТОГО записей: {total_rows:,}")
        print(f"ИТОГО размер (оценка): {total_estimated / (1024**3):.2f} GB")
        print(f"Разница с размером файла: {(db_size - total_estimated) / (1024**3):.2f} GB")
        print("   (разница = индексы + свободное место + фрагментация + метаданные)")
        
        # Детальный анализ самой большой таблицы
        if table_stats:
            largest = table_stats[0]
            print(f"\n🔍 ДЕТАЛЬНЫЙ АНАЛИЗ: {largest['name']}")
            print("=" * 80)
            
            if largest['name'] == 'candles_cache_data':
                try:
                    print(f"   ⏳ Подсчет уникальных символов...")
                    cursor.execute("SELECT COUNT(DISTINCT cache_id) FROM candles_cache_data")
                    unique_symbols = cursor.fetchone()[0]
                    avg_per_symbol = largest['row_count'] / unique_symbols if unique_symbols > 0 else 0
                    print(f"   ✅ Уникальных символов: {unique_symbols}")
                    print(f"   ✅ Среднее свечей на символ: {avg_per_symbol:,.0f}")
                    
                    if avg_per_symbol > 5000:
                        excess = largest['row_count'] - (unique_symbols * 5000)
                        excess_gb = (excess * 70) / (1024**3)
                        print(f"   ⚠️ ЛИШНИХ СВЕЧЕЙ: {excess:,} ({excess_gb:.2f} GB)")
                except Exception as e:
                    print(f"   ⚠️ Ошибка: {e}")
            
            # Проверяем индексы
            print(f"\n   📋 ИНДЕКСЫ для {largest['name']}:")
            try:
                cursor.execute(f"""
                    SELECT name, sql 
                    FROM sqlite_master 
                    WHERE type='index' AND tbl_name = ?
                """, (largest['name'],))
                indexes = cursor.fetchall()
                if indexes:
                    for idx in indexes:
                        print(f"      - {idx[0]}")
                else:
                    print(f"      (нет индексов)")
            except Exception as e:
                print(f"      ⚠️ Ошибка: {e}")
        
        # Проверяем все индексы в БД
        print(f"\n📋 ВСЕ ИНДЕКСЫ В БД:")
        try:
            cursor.execute("SELECT name, tbl_name, sql FROM sqlite_master WHERE type='index' ORDER BY tbl_name")
            indexes = cursor.fetchall()
            if indexes:
                for idx in indexes:
                    print(f"   {idx[1]}.{idx[0]}")
            else:
                print(f"   (нет индексов)")
        except Exception as e:
            print(f"   ⚠️ Ошибка: {e}")
        
        conn.close()
        
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e).lower():
            print(f"\n❌ БД заблокирована другим процессом!")
            print(f"   Закройте все программы, использующие эту БД.")
        else:
            print(f"\n❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Детальная проверка удаленной БД')
    parser.add_argument('db_path', nargs='?', help='Путь к БД')
    args = parser.parse_args()
    
    if args.db_path:
        db_path = args.db_path
    else:
        db_path = os.environ.get('BOTS_DB_PATH')
        if not db_path:
            # Пробуем удаленный путь
            db_path = r'\\Evromini\projects\InfoBot\data\bots_data.db'
            if not Path(db_path).exists():
                db_path = str(PROJECT_ROOT / 'data' / 'bots_data.db')
    
    check_remote_db(db_path)

