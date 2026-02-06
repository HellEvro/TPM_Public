#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Проверка таблицы bots в bots_data.db
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

try:
    from bot_engine.bots_database import get_bots_database
    import sqlite3
    
    bots_db = get_bots_database()
    db_path = bots_db.db_path
    
    print("=" * 80)
    print(f"ПРОВЕРКА ТАБЛИЦЫ bots В {db_path}")
    print("=" * 80)
    
    with bots_db._get_connection() as conn:
        cursor = conn.cursor()
        
        # Проверяем, существует ли таблица bots_state
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='bots_state'")
        bots_state_exists = cursor.fetchone() is not None
        print(f"\n[1] Таблица bots_state существует: {bots_state_exists}")
        
        if bots_state_exists:
            cursor.execute("SELECT COUNT(*) FROM bots_state")
            bots_state_count = cursor.fetchone()[0]
            print(f"   [INFO] Записей в bots_state: {bots_state_count}")
        
        # Проверяем таблицу bots
        cursor.execute("SELECT COUNT(*) FROM bots")
        bots_count = cursor.fetchone()[0]
        print(f"\n[2] Записей в таблице bots: {bots_count}")
        
        if bots_count > 0:
            # Показываем первые 5 ботов
            cursor.execute("SELECT symbol, status, entry_price, position_side, unrealized_pnl_usdt FROM bots LIMIT 5")
            rows = cursor.fetchall()
            print(f"\n[3] Примеры ботов (первые 5):")
            for row in rows:
                print(f"   - {row['symbol']}: {row['status']}, entry={row['entry_price']}, side={row['position_side']}, PnL={row['unrealized_pnl_usdt']}")
        
        # Проверяем, сколько ботов со статусом в позиции
        cursor.execute("SELECT COUNT(*) FROM bots WHERE status LIKE '%position%' OR status LIKE '%LONG%' OR status LIKE '%SHORT%'")
        in_position_count = cursor.fetchone()[0]
        print(f"\n[4] Ботов в позиции: {in_position_count}")
        
        # Показываем все статусы
        cursor.execute("SELECT status, COUNT(*) as count FROM bots GROUP BY status")
        status_rows = cursor.fetchall()
        print(f"\n[5] Распределение по статусам:")
        for row in status_rows:
            print(f"   - {row['status']}: {row['count']}")
        
        print("\n" + "=" * 80)
        print("ВЫВОД:")
        print("=" * 80)
        if bots_state_exists and bots_state_count == 0:
            print("✅ Таблица bots_state пуста (это нормально - данные мигрированы в таблицу bots)")
        print(f"✅ Данные ботов находятся в таблице 'bots' ({bots_count} записей)")
        print("✅ В GUI выберите таблицу 'bots' вместо 'bots_state'")
        print("=" * 80)
        
except Exception as e:
    print(f"\n[ERROR] Ошибка: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

