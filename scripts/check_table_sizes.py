#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Точный анализ размера таблиц в БД через SQLite статистику
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

def get_table_sizes(db_path: str):
    """Получает точные размеры таблиц через SQLite статистику"""
    print("=" * 80)
    print(f"АНАЛИЗ РАЗМЕРОВ ТАБЛИЦ: {db_path}")
    print("=" * 80)
    
    if not Path(db_path).exists():
        print(f"❌ Файл не найден: {db_path}")
        return
    
    # Размер файла
    db_size = Path(db_path).stat().st_size
    print(f"\n📊 Размер БД: {db_size / (1024**3):.2f} GB ({db_size / (1024**2):.2f} MB)")
    
    try:
        conn = sqlite3.connect(str(db_path), timeout=60.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Включаем статистику страниц
        cursor.execute("PRAGMA page_size")
        page_size = cursor.fetchone()[0]
        print(f"📄 Размер страницы: {page_size} байт")
        
        cursor.execute("PRAGMA page_count")
        page_count = cursor.fetchone()[0]
        total_pages_size = page_count * page_size
        print(f"📄 Всего страниц: {page_count:,} ({total_pages_size / (1024**3):.2f} GB)")
        
        # Получаем список таблиц
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        
        print(f"\n📊 РАЗМЕРЫ ТАБЛИЦ (через dbstat):")
        print("=" * 80)
        
        table_sizes = []
        
        # Сначала пробуем dbstat
        use_dbstat = True
        try:
            cursor.execute("SELECT COUNT(*) FROM dbstat LIMIT 1")
        except:
            use_dbstat = False
            print("⚠️ dbstat недоступен, используем приблизительные оценки")
        
        for table in tables:
            try:
                row_count = 0
                total_size = 0
                pages = 0
                
                # Получаем количество записей
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    row_count = cursor.fetchone()[0]
                except:
                    pass
                
                if use_dbstat:
                    try:
                        # Используем dbstat для получения точного размера
                        cursor.execute(f"""
                            SELECT 
                                COUNT(*) as pages,
                                SUM(pgsize) as total_size
                            FROM dbstat
                            WHERE name = ?
                        """, (table,))
                        
                        result = cursor.fetchone()
                        if result and result[0]:
                            pages = result[0]
                            total_size = result[1] if result[1] else 0
                    except:
                        use_dbstat = False
                
                # Если dbstat не сработал, используем приблизительную оценку
                if total_size == 0 and row_count > 0:
                    # Более точные оценки на основе структуры таблицы
                    if 'candles_cache_data' in table:
                        # id (INTEGER) + cache_id (INTEGER) + time (INTEGER) + 4 REAL + volume (REAL) = ~40 байт + overhead
                        total_size = row_count * 60
                    elif 'candles_cache' in table:
                        total_size = row_count * 200
                    elif 'trades' in table or 'history' in table:
                        total_size = row_count * 400  # Больше полей
                    elif 'cache' in table:
                        total_size = row_count * 300
                    elif 'rsi' in table:
                        total_size = row_count * 100
                    else:
                        total_size = row_count * 200  # Средняя оценка
                
                if row_count > 0 or total_size > 0:
                    table_sizes.append({
                        'name': table,
                        'pages': pages,
                        'size_bytes': total_size,
                        'row_count': row_count,
                        'unsynced': 0
                    })
            except Exception as e:
                print(f"   ⚠️ Ошибка анализа {table}: {e}")
                pass
        
        # Сортируем по размеру
        table_sizes.sort(key=lambda x: x['size_bytes'], reverse=True)
        
        total_analyzed = 0
        for stat in table_sizes:
            size_gb = stat['size_bytes'] / (1024**3)
            size_mb = stat['size_bytes'] / (1024**2)
            total_analyzed += stat['size_bytes']
            
            if size_gb > 0.01:
                print(f"{stat['name']:35} {stat['row_count']:>12,} записей  {size_gb:>8.2f} GB ({size_mb:>8.2f} MB)")
                if stat['pages'] > 0:
                    print(f"   {'':35} {stat['pages']:>12,} страниц")
            elif size_mb > 1:
                print(f"{stat['name']:35} {stat['row_count']:>12,} записей  {size_mb:>8.2f} MB")
            else:
                print(f"{stat['name']:35} {stat['row_count']:>12,} записей")
        
        print("=" * 80)
        print(f"ИТОГО проанализировано: {total_analyzed / (1024**3):.2f} GB")
        print(f"Разница с размером файла: {(db_size - total_analyzed) / (1024**3):.2f} GB")
        print("   (разница может быть из-за индексов, свободного места, WAL и т.д.)")
        
        # Детальный анализ самой большой таблицы
        if table_sizes:
            largest = table_sizes[0]
            print(f"\n🔍 ДЕТАЛЬНЫЙ АНАЛИЗ САМОЙ БОЛЬШОЙ ТАБЛИЦЫ: {largest['name']}")
            print("=" * 80)
            
            if largest['name'] == 'candles_cache_data':
                try:
                    cursor.execute("SELECT COUNT(DISTINCT cache_id) FROM candles_cache_data")
                    unique_symbols = cursor.fetchone()[0]
                    avg_per_symbol = largest['row_count'] / unique_symbols if unique_symbols > 0 else 0
                    print(f"   Уникальных символов: {unique_symbols}")
                    print(f"   Среднее свечей на символ: {avg_per_symbol:,.0f}")
                    
                    if avg_per_symbol > 5000:
                        excess = largest['row_count'] - (unique_symbols * 5000)
                        excess_gb = (excess * 60) / (1024**3)
                        print(f"   ⚠️ ЛИШНИХ СВЕЧЕЙ: {excess:,} ({excess_gb:.2f} GB)")
                except Exception as e:
                    print(f"   ⚠️ Ошибка детального анализа: {e}")
            
            # Показываем пример структуры
            try:
                cursor.execute(f"SELECT * FROM {largest['name']} LIMIT 1")
                sample = cursor.fetchone()
                if sample:
                    print(f"   Пример записи: {dict(sample)}")
            except:
                pass
        
        conn.close()
        
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e).lower():
            print(f"\n❌ БД заблокирована другим процессом!")
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
    
    parser = argparse.ArgumentParser(description='Анализ размеров таблиц')
    parser.add_argument('db_path', nargs='?', help='Путь к БД')
    args = parser.parse_args()
    
    if args.db_path:
        db_path = args.db_path
    else:
        db_path = os.environ.get('BOTS_DB_PATH')
        if not db_path:
            db_path = str(PROJECT_ROOT / 'data' / 'bots_data.db')
    
    get_table_sizes(db_path)

