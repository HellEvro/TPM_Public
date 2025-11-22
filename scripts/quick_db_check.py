#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Быстрая проверка размера БД и количества записей в candles_cache_data
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

def quick_check(db_path: str):
    """Быстрая проверка БД"""
    print("=" * 80)
    print(f"БЫСТРАЯ ПРОВЕРКА БД: {db_path}")
    print("=" * 80)
    
    if not Path(db_path).exists():
        print(f"❌ Файл не найден: {db_path}")
        return
    
    # Размеры файлов
    db_file = Path(db_path)
    wal_file = Path(str(db_path) + '-wal')
    
    db_size = db_file.stat().st_size if db_file.exists() else 0
    wal_size = wal_file.stat().st_size if wal_file.exists() else 0
    
    print(f"\n📊 РАЗМЕРЫ:")
    print(f"   Основной файл: {db_size / (1024**3):.2f} GB ({db_size / (1024**2):.2f} MB)")
    if wal_file.exists() and wal_size > 0:
        print(f"   WAL файл: {wal_size / (1024**3):.2f} GB ({wal_size / (1024**2):.2f} MB)")
    print(f"   ИТОГО: {(db_size + wal_size) / (1024**3):.2f} GB")
    
    # Подключение к БД
    try:
        print(f"\n⏳ Подключение к БД (может занять время)...")
        conn = sqlite3.connect(str(db_path), timeout=60.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Проверяем candles_cache_data
        print(f"\n📊 АНАЛИЗ candles_cache_data:")
        try:
            print(f"   ⏳ Подсчет записей (может занять много времени для больших таблиц)...")
            cursor.execute("SELECT COUNT(*) FROM candles_cache_data")
            total_candles = cursor.fetchone()[0]
            print(f"   ✅ Всего свечей: {total_candles:,}")
            
            if total_candles > 0:
                # Количество символов
                cursor.execute("SELECT COUNT(DISTINCT cache_id) FROM candles_cache_data")
                unique_symbols = cursor.fetchone()[0]
                print(f"   ✅ Уникальных символов: {unique_symbols}")
                
                if unique_symbols > 0:
                    avg_candles = total_candles / unique_symbols
                    print(f"   ✅ Среднее свечей на символ: {avg_candles:,.0f}")
                    
                    # Расчет лишних свечей
                    limit = 5000
                    if avg_candles > limit:
                        excess_candles = total_candles - (unique_symbols * limit)
                        excess_gb = (excess_candles * 60) / (1024**3)  # ~60 байт на свечу
                        print(f"\n   ⚠️ ПРЕВЫШЕНИЕ ЛИМИТА!")
                        print(f"   Лишних свечей: {excess_candles:,}")
                        print(f"   Примерный размер лишних данных: {excess_gb:.2f} GB")
                    
                    # Топ символов
                    print(f"\n   📋 ТОП-5 символов по количеству свечей:")
                    cursor.execute("""
                        SELECT 
                            cc.symbol,
                            COUNT(ccd.id) as candle_count
                        FROM candles_cache_data ccd
                        JOIN candles_cache cc ON ccd.cache_id = cc.id
                        GROUP BY cc.symbol
                        ORDER BY candle_count DESC
                        LIMIT 5
                    """)
                    top_symbols = cursor.fetchall()
                    for row in top_symbols:
                        symbol = row[0]
                        count = row[1]
                        if count > limit:
                            print(f"      {symbol:20} {count:>10,} свечей ⚠️ (лимит: {limit})")
                        else:
                            print(f"      {symbol:20} {count:>10,} свечей")
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
        
        # Проверяем другие таблицы
        print(f"\n📊 ДРУГИЕ ТАБЛИЦЫ:")
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [row[0] for row in cursor.fetchall()]
            
            for table in tables:
                if table == 'candles_cache_data':
                    continue
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    if count > 0:
                        print(f"   {table:30} {count:>12,} записей")
                except:
                    pass
        except Exception as e:
            print(f"   ⚠️ Ошибка: {e}")
        
        conn.close()
        
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e).lower():
            print(f"\n❌ БД заблокирована другим процессом!")
            print(f"   Закройте все программы, использующие эту БД, и попробуйте снова.")
        else:
            print(f"\n❌ Ошибка подключения: {e}")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("РЕКОМЕНДАЦИИ:")
    print("=" * 80)
    print("💡 Если candles_cache_data содержит миллионы записей:")
    print("   1. Запустите: python scripts/cleanup_all_candles.py --skip-vacuum")
    print("   2. После очистки выполните VACUUM отдельно (когда БД не используется)")
    print("=" * 80)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Быстрая проверка размера БД')
    parser.add_argument('db_path', nargs='?', help='Путь к БД')
    args = parser.parse_args()
    
    if args.db_path:
        db_path = args.db_path
    else:
        db_path = os.environ.get('BOTS_DB_PATH')
        if not db_path:
            db_path = str(PROJECT_ROOT / 'data' / 'bots_data.db')
    
    quick_check(db_path)

