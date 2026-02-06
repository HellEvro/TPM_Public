#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Прямая очистка БД с обработкой блокировок
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

MAX_CANDLES = 1000

def cleanup_direct():
    """Прямая очистка БД"""
    db_path = PROJECT_ROOT / 'data' / 'bots_data.db'
    
    print("=" * 80)
    print(f"ПРЯМАЯ ОЧИСТКА БД: {db_path}")
    print("=" * 80)
    
    if not db_path.exists():
        print(f"❌ Файл не найден")
        return
    
    size_before = db_path.stat().st_size
    print(f"📊 Размер БД: {size_before / (1024**3):.2f} GB")
    
    # Пробуем разные способы подключения
    conn = None
    for attempt in range(5):
        try:
            print(f"\n⏳ Попытка подключения #{attempt + 1}...")
            conn = sqlite3.connect(str(db_path), timeout=600.0)
            conn.row_factory = sqlite3.Row
            print("✅ Подключение успешно!")
            break
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e).lower():
                print(f"   ⏳ БД заблокирована, жду 10 секунд...")
                time.sleep(10)
            else:
                print(f"   ❌ Ошибка: {e}")
                if attempt < 4:
                    time.sleep(5)
                else:
                    print("❌ Не удалось подключиться после 5 попыток")
                    return
    
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        
        # Проверяем количество свечей
        print("\n⏳ Подсчет свечей...")
        cursor.execute("SELECT COUNT(*) FROM candles_cache_data")
        total_before = cursor.fetchone()[0]
        print(f"📊 Всего свечей: {total_before:,}")
        
        # Получаем символы
        cursor.execute("SELECT id, symbol FROM candles_cache")
        symbols = cursor.fetchall()
        print(f"📊 Символов: {len(symbols)}")
        
        total_deleted = 0
        processed = 0
        
        for cache_id, symbol in symbols:
            processed += 1
            if processed % 50 == 0:
                print(f"⏳ Обработано: {processed}/{len(symbols)}")
            
            # Количество свечей
            cursor.execute("SELECT COUNT(*) FROM candles_cache_data WHERE cache_id = ?", (cache_id,))
            count = cursor.fetchone()[0]
            
            if count <= MAX_CANDLES:
                continue
            
            # Удаляем лишние
            excess = count - MAX_CANDLES
            cursor.execute(f"""
                DELETE FROM candles_cache_data
                WHERE id IN (
                    SELECT id FROM candles_cache_data
                    WHERE cache_id = ?
                    ORDER BY time ASC
                    LIMIT ?
                )
            """, (cache_id, excess))
            
            deleted = cursor.rowcount
            total_deleted += deleted
            
            if deleted > 0:
                print(f"   🗑️ {symbol}: удалено {deleted:,} свечей")
            
            conn.commit()
        
        print(f"\n✅ Удалено свечей: {total_deleted:,}")
        
        cursor.execute("SELECT COUNT(*) FROM candles_cache_data")
        total_after = cursor.fetchone()[0]
        print(f"📊 Осталось свечей: {total_after:,}")
        
        conn.close()
        
        # VACUUM
        print(f"\n⏳ Выполнение VACUUM...")
        conn = sqlite3.connect(str(db_path), timeout=3600.0)
        cursor = conn.cursor()
        cursor.execute("VACUUM")
        conn.close()
        
        size_after = db_path.stat().st_size
        freed = (size_before - size_after) / (1024**3)
        
        print(f"\n✅ Готово!")
        print(f"📊 Размер до: {size_before / (1024**3):.2f} GB")
        print(f"📊 Размер после: {size_after / (1024**3):.2f} GB")
        print(f"💾 Освобождено: {freed:.2f} GB")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        if conn:
            conn.close()

if __name__ == '__main__':
    cleanup_direct()

