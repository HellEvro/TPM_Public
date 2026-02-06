#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Проверка количества свечей в candles_cache_data и анализ проблемы
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

def check_candles_count(db_path: str):
    """Проверка количества свечей"""
    print("=" * 80)
    print(f"ПРОВЕРКА candles_cache_data: {db_path}")
    print("=" * 80)
    
    if not Path(db_path).exists():
        print(f"❌ Файл не найден: {db_path}")
        return
    
    db_size = Path(db_path).stat().st_size
    print(f"\n📊 Размер БД: {db_size / (1024**3):.2f} GB ({db_size / (1024**2):.2f} MB)")
    
    try:
        print(f"\n⏳ Подключение к БД...")
        conn = sqlite3.connect(str(db_path), timeout=120.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Общее количество свечей
        print(f"⏳ Подсчет общего количества свечей (может занять время)...")
        cursor.execute("SELECT COUNT(*) FROM candles_cache_data")
        total_candles = cursor.fetchone()[0]
        print(f"✅ Всего свечей: {total_candles:,}")
        
        # Количество символов
        cursor.execute("SELECT COUNT(DISTINCT cache_id) FROM candles_cache_data")
        unique_symbols = cursor.fetchone()[0]
        print(f"✅ Уникальных символов: {unique_symbols}")
        
        if unique_symbols > 0:
            avg_per_symbol = total_candles / unique_symbols
            print(f"✅ Среднее свечей на символ: {avg_per_symbol:,.0f}")
            
            # Расчет ожидаемого размера
            expected_size = total_candles * 70  # ~70 байт на свечу
            print(f"\n📊 РАСЧЕТ РАЗМЕРА:")
            print(f"   Ожидаемый размер данных: {expected_size / (1024**3):.2f} GB")
            print(f"   Реальный размер БД: {db_size / (1024**3):.2f} GB")
            print(f"   Разница: {(db_size - expected_size) / (1024**3):.2f} GB")
            print(f"   (разница = индексы + фрагментация + свободное место)")
            
            # Проверяем превышение лимита
            limit = 5000
            if avg_per_symbol > limit:
                excess_candles = total_candles - (unique_symbols * limit)
                excess_gb = (excess_candles * 70) / (1024**3)
                print(f"\n⚠️ ПРЕВЫШЕНИЕ ЛИМИТА:")
                print(f"   Лимит на символ: {limit:,} свечей")
                print(f"   Лишних свечей: {excess_candles:,}")
                print(f"   Размер лишних данных: {excess_gb:.2f} GB")
                print(f"\n💡 Нужно запустить очистку:")
                print(f"   python scripts/cleanup_all_candles.py --skip-vacuum")
            else:
                print(f"\n✅ Все символы в пределах лимита ({limit:,} свечей)")
                print(f"   Проблема не в количестве свечей, а в фрагментации БД!")
                print(f"\n💡 Нужно выполнить VACUUM для дефрагментации:")
                print(f"   python scripts/vacuum_db_safe.py \"{db_path}\"")
            
            # Топ символов с превышением
            print(f"\n📋 СИМВОЛЫ С ПРЕВЫШЕНИЕМ ЛИМИТА:")
            cursor.execute("""
                SELECT 
                    cc.symbol,
                    COUNT(ccd.id) as count
                FROM candles_cache_data ccd
                JOIN candles_cache cc ON ccd.cache_id = cc.id
                GROUP BY cc.symbol
                HAVING count > ?
                ORDER BY count DESC
                LIMIT 20
            """, (limit,))
            
            excess_symbols = cursor.fetchall()
            if excess_symbols:
                total_excess = 0
                for row in excess_symbols:
                    symbol = row[0]
                    count = row[1]
                    excess = count - limit
                    total_excess += excess
                    print(f"   {symbol:20} {count:>10,} свечей ⚠️ (+{excess:,} лишних)")
                
                excess_gb = (total_excess * 70) / (1024**3)
                print(f"\n   ВСЕГО лишних свечей в топ-20: {total_excess:,} ({excess_gb:.2f} GB)")
            else:
                print(f"   ✅ Нет символов с превышением лимита")
        
        # Проверяем свободное место
        cursor.execute("PRAGMA page_count")
        page_count = cursor.fetchone()[0]
        cursor.execute("PRAGMA freelist_count")
        freelist_count = cursor.fetchone()[0]
        cursor.execute("PRAGMA page_size")
        page_size = cursor.fetchone()[0]
        
        free_size = freelist_count * page_size
        free_percent = (freelist_count / page_count * 100) if page_count > 0 else 0
        
        print(f"\n📄 ФРАГМЕНТАЦИЯ БД:")
        print(f"   Всего страниц: {page_count:,}")
        print(f"   Свободных страниц: {freelist_count:,} ({free_size / (1024**3):.2f} GB)")
        print(f"   Процент свободного места: {free_percent:.1f}%")
        
        if free_percent > 10:
            print(f"\n⚠️ Много свободного места ({free_percent:.1f}%)!")
            print(f"   Это указывает на сильную фрагментацию БД")
            print(f"   VACUUM освободит {free_size / (1024**3):.2f} GB")
        
        conn.close()
        
        print("\n" + "=" * 80)
        print("РЕКОМЕНДАЦИИ:")
        print("=" * 80)
        
        if avg_per_symbol > limit:
            print("1. Запустите очистку лишних свечей:")
            print(f"   python scripts/cleanup_all_candles.py --skip-vacuum")
            print("\n2. После очистки выполните VACUUM:")
            print(f"   python scripts/vacuum_db_safe.py \"{db_path}\"")
        else:
            print("1. Выполните VACUUM для дефрагментации:")
            print(f"   python scripts/vacuum_db_safe.py \"{db_path}\"")
            print("\n   ⚠️ VACUUM для 28 GB БД может занять ЧАСЫ!")
            print("   Убедитесь, что все программы, использующие БД, закрыты")
        
        print("=" * 80)
        
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e).lower():
            print(f"\n❌ БД заблокирована другим процессом!")
            print(f"   Закройте все программы, использующие эту БД")
        else:
            print(f"\n❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Проверка количества свечей')
    parser.add_argument('db_path', nargs='?', help='Путь к БД')
    args = parser.parse_args()
    
    if args.db_path:
        db_path = args.db_path
    else:
        db_path = os.environ.get('BOTS_DB_PATH')
        if not db_path:
            db_path = str(PROJECT_ROOT / 'data' / 'bots_data.db')
    
    check_candles_count(db_path)

