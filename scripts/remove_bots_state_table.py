#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для удаления старой таблицы bots_state

ВАЖНО: Таблица bots_state была удалена при миграции, данные перенесены в:
- bots (основные данные ботов)
- auto_bot_config (конфигурация автобота)

Если таблица bots_state все еще существует, этот скрипт удалит её.
"""

import sys
import os
from pathlib import Path

# Настройка кодировки для Windows консоли
if os.name == 'nt':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 80)
print("УДАЛЕНИЕ СТАРОЙ ТАБЛИЦЫ bots_state")
print("=" * 80)

try:
    from bot_engine.bots_database import get_bots_database
    import sqlite3
    
    bots_db = get_bots_database()
    db_path = bots_db.db_path
    
    print(f"\n[INFO] База данных: {db_path}")
    
    with bots_db._get_connection() as conn:
        cursor = conn.cursor()
        
        # Проверяем, существует ли таблица bots_state
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='bots_state'")
        bots_state_exists = cursor.fetchone() is not None
        
        if bots_state_exists:
            # Проверяем количество записей
            cursor.execute("SELECT COUNT(*) FROM bots_state")
            count = cursor.fetchone()[0]
            print(f"\n[INFO] Таблица bots_state существует, записей: {count}")
            
            # Проверяем, есть ли данные в таблице bots
            cursor.execute("SELECT COUNT(*) FROM bots")
            bots_count = cursor.fetchone()[0]
            print(f"[INFO] Записей в таблице bots: {bots_count}")
            
            if bots_count > 0:
                # Данные мигрированы - можно безопасно удалить
                print("\n[ACTION] Удаление таблицы bots_state...")
                cursor.execute("DROP TABLE IF EXISTS bots_state")
                conn.commit()
                print("✅ Таблица bots_state успешно удалена")
                print("\n[INFO] Данные ботов находятся в таблице 'bots'")
                print("[INFO] В GUI выберите таблицу 'bots' вместо 'bots_state'")
            else:
                print("\n[WARNING] В таблице bots нет данных!")
                print("[WARNING] Не удаляю bots_state - возможно данные еще не мигрированы")
        else:
            print("\n[INFO] Таблица bots_state не существует (уже удалена)")
            print("[INFO] Данные ботов находятся в таблице 'bots'")
        
        print("\n" + "=" * 80)
        print("ПРОВЕРКА ТАБЛИЦЫ bots:")
        print("=" * 80)
        cursor.execute("SELECT COUNT(*) FROM bots")
        bots_count = cursor.fetchone()[0]
        print(f"Записей в таблице bots: {bots_count}")
        
        if bots_count > 0:
            cursor.execute("SELECT symbol, status, entry_price, position_side FROM bots LIMIT 10")
            rows = cursor.fetchall()
            print(f"\nПримеры ботов (первые {min(10, bots_count)}):")
            for row in rows:
                print(f"  - {row['symbol']}: {row['status']}, entry={row['entry_price']}, side={row['position_side']}")
        
        print("\n" + "=" * 80)
        
except Exception as e:
    print(f"\n[ERROR] Ошибка: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

