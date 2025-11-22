#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Детальный анализ размера БД и таблиц
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

def analyze_database(db_path: str):
    """Детальный анализ размера БД"""
    print("=" * 80)
    print(f"АНАЛИЗ РАЗМЕРА БД: {db_path}")
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
    
    total_size = db_size + wal_size + shm_size
    
    print(f"\n📊 РАЗМЕРЫ ФАЙЛОВ:")
    print(f"   Основной файл: {db_size / (1024**3):.2f} GB ({db_size / (1024**2):.2f} MB)")
    if wal_file.exists():
        print(f"   WAL файл: {wal_size / (1024**3):.2f} GB ({wal_size / (1024**2):.2f} MB) ⚠️")
    if shm_file.exists():
        print(f"   SHM файл: {shm_size / (1024**2):.2f} MB")
    print(f"   ИТОГО: {total_size / (1024**3):.2f} GB ({total_size / (1024**2):.2f} MB)")
    
    # Подключение к БД
    try:
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Получаем список всех таблиц
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        
        print(f"\n📋 АНАЛИЗ ТАБЛИЦ:")
        print(f"   Всего таблиц: {len(tables)}")
        
        table_stats = []
        
        for table in tables:
            try:
                # Количество записей
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                
                # Приблизительный размер данных таблицы
                # Используем page_count для оценки
                cursor.execute(f"SELECT COUNT(*) FROM pragma_page_count()")
                page_count = cursor.execute(f"PRAGMA page_count").fetchone()[0]
                page_size = cursor.execute(f"PRAGMA page_size").fetchone()[0]
                
                # Подсчет записей
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                
                # Приблизительная оценка размера на основе типа таблицы
                # Для candles_cache_data: ~50 байт на свечу (time, open, high, low, close, volume = 6*8 = 48 байт + overhead)
                if 'candles_cache_data' in table.lower():
                    table_size = row_count * 60  # ~60 байт на свечу с учетом индексов
                elif 'candles' in table.lower():
                    table_size = row_count * 50
                elif 'trades' in table.lower() or 'history' in table.lower():
                    table_size = row_count * 300  # Больше полей
                elif 'cache' in table.lower():
                    table_size = row_count * 200
                else:
                    table_size = row_count * 150  # Средний размер записи
                
                table_stats.append({
                    'name': table,
                    'count': row_count,
                    'size_estimate': table_size
                })
                    
            except Exception as e:
                print(f"   ⚠️ Ошибка анализа таблицы {table}: {e}")
                table_stats.append({
                    'name': table,
                    'count': 0,
                    'size_estimate': 0
                })
        
        # Сортируем по размеру
        table_stats.sort(key=lambda x: x['size_estimate'], reverse=True)
        
        print(f"\n📊 ТАБЛИЦЫ ПО РАЗМЕРУ:")
        total_estimated = 0
        for stat in table_stats:
            size_mb = stat['size_estimate'] / (1024**2)
            size_gb = stat['size_estimate'] / (1024**3)
            total_estimated += stat['size_estimate']
            
            if stat['count'] > 0:
                if size_gb > 0.1:
                    print(f"   {stat['name']:30} {stat['count']:>12,} записей  {size_gb:>8.2f} GB ({size_mb:>8.2f} MB)")
                elif size_mb > 1:
                    print(f"   {stat['name']:30} {stat['count']:>12,} записей  {size_mb:>8.2f} MB")
                else:
                    print(f"   {stat['name']:30} {stat['count']:>12,} записей")
        
        print(f"\n   ИТОГО (оценка): {total_estimated / (1024**3):.2f} GB")
        
        # Детальный анализ candles_cache_data
        if 'candles_cache_data' in [t['name'] for t in table_stats]:
            print(f"\n🔍 ДЕТАЛЬНЫЙ АНАЛИЗ candles_cache_data:")
            try:
                cursor.execute("""
                    SELECT 
                        COUNT(DISTINCT cache_id) as unique_symbols,
                        COUNT(*) as total_candles,
                        MIN(time) as oldest_time,
                        MAX(time) as newest_time
                    FROM candles_cache_data
                """)
                stats = cursor.fetchone()
                if stats:
                    print(f"   Уникальных символов: {stats[0]}")
                    print(f"   Всего свечей: {stats[1]:,}")
                    if stats[1] > 0:
                        avg_per_symbol = stats[1] / stats[0] if stats[0] > 0 else 0
                        print(f"   Среднее свечей на символ: {avg_per_symbol:,.0f}")
                        if avg_per_symbol > 5000:
                            excess = stats[1] - (stats[0] * 5000)
                            print(f"   ⚠️ ПРЕВЫШЕНИЕ ЛИМИТА: {excess:,} лишних свечей!")
                
                # Топ символов по количеству свечей
                cursor.execute("""
                    SELECT 
                        cc.symbol,
                        COUNT(ccd.id) as candle_count
                    FROM candles_cache_data ccd
                    JOIN candles_cache cc ON ccd.cache_id = cc.id
                    GROUP BY cc.symbol
                    ORDER BY candle_count DESC
                    LIMIT 10
                """)
                top_symbols = cursor.fetchall()
                if top_symbols:
                    print(f"\n   ТОП-10 символов по количеству свечей:")
                    for row in top_symbols:
                        symbol = row[0]
                        count = row[1]
                        if count > 5000:
                            excess = count - 5000
                            print(f"      {symbol:15} {count:>8,} свечей ⚠️ (+{excess:,} лишних)")
                        else:
                            print(f"      {symbol:15} {count:>8,} свечей")
            except Exception as e:
                print(f"   ⚠️ Ошибка детального анализа: {e}")
        
        # Анализ других больших таблиц
        print(f"\n🔍 АНАЛИЗ ДРУГИХ БОЛЬШИХ ТАБЛИЦ:")
        for stat in table_stats[:5]:
            if stat['name'] == 'candles_cache_data':
                continue
            if stat['count'] > 10000:
                print(f"\n   {stat['name']}:")
                try:
                    # Пытаемся получить примеры данных
                    cursor.execute(f"SELECT * FROM {stat['name']} LIMIT 1")
                    sample = cursor.fetchone()
                    if sample:
                        print(f"      Пример записи: {dict(sample)}")
                except:
                    pass
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Ошибка подключения к БД: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("РЕКОМЕНДАЦИИ:")
    print("=" * 80)
    
    if wal_size > 100 * 1024 * 1024:  # >100 MB
        print("⚠️ WAL файл очень большой!")
        print("💡 Выполните: PRAGMA wal_checkpoint(TRUNCATE)")
    
    # Проверяем candles_cache_data
    try:
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM candles_cache_data")
        total_candles = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(DISTINCT cache_id) FROM candles_cache_data")
        unique_symbols = cursor.fetchone()[0]
        conn.close()
        
        if unique_symbols > 0:
            avg_candles = total_candles / unique_symbols
            if avg_candles > 5000:
                excess_candles = total_candles - (unique_symbols * 5000)
                print(f"\n⚠️ candles_cache_data содержит слишком много свечей!")
                print(f"   Всего: {total_candles:,} свечей")
                print(f"   Символов: {unique_symbols}")
                print(f"   Среднее: {avg_candles:,.0f} свечей на символ (лимит: 5000)")
                print(f"   Лишних свечей: {excess_candles:,}")
                print(f"\n💡 Запустите очистку:")
                print(f"   python scripts/cleanup_all_candles.py --skip-vacuum")
    except:
        pass
    
    print("=" * 80)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Анализ размера БД')
    parser.add_argument('db_path', nargs='?', help='Путь к БД (по умолчанию bots_data.db)')
    args = parser.parse_args()
    
    if args.db_path:
        db_path = args.db_path
    else:
        db_path = os.environ.get('BOTS_DB_PATH')
        if not db_path:
            db_path = str(PROJECT_ROOT / 'data' / 'bots_data.db')
    
    analyze_database(db_path)

