#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для очистки старых свечей из candles_cache_data

Оставляет только последние N свечей для каждого символа.
"""

import sys
import os
from pathlib import Path
import sqlite3
from datetime import datetime

# Настройка кодировки для Windows консоли
if os.name == 'nt':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Путь к БД (может быть локальным или удаленным)
DB_PATH = os.environ.get('BOTS_DB_PATH', None)
if not DB_PATH:
    DB_PATH = PROJECT_ROOT / 'data' / 'bots_data.db'
else:
    DB_PATH = Path(DB_PATH)

MAX_CANDLES_PER_SYMBOL = 5000  # Максимум свечей на символ

print("=" * 80)
print("ОЧИСТКА СТАРЫХ СВЕЧЕЙ ИЗ candles_cache_data")
print("=" * 80)
print(f"БД: {DB_PATH}")
print(f"Максимум свечей на символ: {MAX_CANDLES_PER_SYMBOL}")
print("=" * 80)

if not DB_PATH.exists():
    print(f"❌ БД не найдена: {DB_PATH}")
    sys.exit(1)

try:
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Получаем список всех символов
    cursor.execute("SELECT id, symbol FROM candles_cache")
    symbols = cursor.fetchall()
    
    print(f"\n📊 Найдено символов: {len(symbols)}")
    
    total_deleted = 0
    total_kept = 0
    
    for symbol_row in symbols:
        cache_id = symbol_row['id']
        symbol = symbol_row['symbol']
        
        # Подсчитываем количество свечей для этого символа
        cursor.execute("SELECT COUNT(*) FROM candles_cache_data WHERE cache_id = ?", (cache_id,))
        count = cursor.fetchone()[0]
        
        if count <= MAX_CANDLES_PER_SYMBOL:
            print(f"✅ {symbol}: {count:,} свечей (в пределах лимита)")
            total_kept += count
            continue
        
        # Получаем временные метки свечей, отсортированные по времени
        cursor.execute("""
            SELECT time FROM candles_cache_data 
            WHERE cache_id = ? 
            ORDER BY time ASC
        """, (cache_id,))
        
        times = [row[0] for row in cursor.fetchall()]
        
        if len(times) <= MAX_CANDLES_PER_SYMBOL:
            total_kept += len(times)
            continue
        
        # Определяем границу времени (оставляем последние MAX_CANDLES_PER_SYMBOL свечей)
        cutoff_time = times[-MAX_CANDLES_PER_SYMBOL]
        
        # Удаляем старые свечи
        cursor.execute("""
            DELETE FROM candles_cache_data 
            WHERE cache_id = ? AND time < ?
        """, (cache_id, cutoff_time))
        
        deleted_count = cursor.rowcount
        kept_count = count - deleted_count
        
        print(f"🧹 {symbol}: удалено {deleted_count:,} старых свечей, оставлено {kept_count:,} (было {count:,})")
        
        total_deleted += deleted_count
        total_kept += kept_count
        
        # Обновляем метаданные
        if kept_count > 0:
            cursor.execute("""
                SELECT MIN(time) as first_time, MAX(time) as last_time 
                FROM candles_cache_data 
                WHERE cache_id = ?
            """, (cache_id,))
            time_info = cursor.fetchone()
            
            cursor.execute("""
                UPDATE candles_cache 
                SET candles_count = ?, first_candle_time = ?, last_candle_time = ?
                WHERE id = ?
            """, (kept_count, time_info['first_time'], time_info['last_time'], cache_id))
    
    # Коммитим изменения
    conn.commit()
    
    # Выполняем VACUUM для освобождения места
    print("\n" + "=" * 80)
    print("ОСВОБОЖДЕНИЕ МЕСТА (VACUUM)...")
    print("=" * 80)
    print("⏳ Это может занять некоторое время для больших БД...")
    
    cursor.execute("VACUUM")
    
    # Проверяем новый размер БД
    new_size_mb = DB_PATH.stat().st_size / (1024 * 1024)
    new_size_gb = new_size_mb / 1024
    
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ:")
    print("=" * 80)
    print(f"✅ Удалено старых свечей: {total_deleted:,}")
    print(f"✅ Оставлено свечей: {total_kept:,}")
    print(f"📊 Новый размер БД: {new_size_mb:.2f} MB ({new_size_gb:.2f} GB)")
    
    conn.close()
    
    print("\n✅ Очистка завершена успешно!")
    
except Exception as e:
    print(f"\n❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

