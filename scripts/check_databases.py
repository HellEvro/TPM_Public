#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для проверки баз данных на наличие проблем
"""

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def check_table(conn, table_name):
    """Проверяет таблицу на наличие данных"""
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        return count
    except sqlite3.OperationalError as e:
        return f"ERROR: {e}"

def get_table_schema(conn, table_name):
    """Получает схему таблицы"""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    return cursor.fetchall()

def check_bots_db():
    """Проверяет bots_data.db"""
    print("=" * 80)
    print("ПРОВЕРКА bots_data.db")
    print("=" * 80)
    
    db_path = PROJECT_ROOT / 'data' / 'bots_data.db'
    if not db_path.exists():
        print(f"❌ Файл не найден: {db_path}")
        return
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Получаем список таблиц
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    
    print(f"\n📋 Таблицы в базе ({len(tables)}):")
    for table in tables:
        count = check_table(conn, table)
        print(f"   {table}: {count} записей")
    
    # Проверяем bot_trades_history
    print(f"\n🔍 Детальная проверка bot_trades_history:")
    try:
        cursor.execute("SELECT COUNT(*) FROM bot_trades_history")
        total = cursor.fetchone()[0]
        print(f"   Всего записей: {total}")
        
        if total > 0:
            cursor.execute("SELECT COUNT(*) FROM bot_trades_history WHERE status = 'OPEN'")
            open_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM bot_trades_history WHERE status = 'CLOSED'")
            closed_count = cursor.fetchone()[0]
            print(f"   Открытых: {open_count}")
            print(f"   Закрытых: {closed_count}")
            
            # Проверяем последние записи
            cursor.execute("""
                SELECT symbol, direction, status, entry_price, exit_price, pnl, 
                       decision_source, created_at
                FROM bot_trades_history
                ORDER BY created_at DESC
                LIMIT 5
            """)
            print(f"\n   Последние 5 записей:")
            for row in cursor.fetchall():
                print(f"      {row[0]} {row[1]} | {row[2]} | entry={row[3]} | exit={row[4]} | pnl={row[5]} | source={row[6]} | {row[7]}")
        else:
            print("   ⚠️ Таблица пуста!")
    except sqlite3.OperationalError as e:
        print(f"   ❌ Ошибка: {e}")
    
    # Проверяем bots
    print(f"\n🔍 Детальная проверка bots:")
    try:
        cursor.execute("SELECT COUNT(*) FROM bots")
        total = cursor.fetchone()[0]
        print(f"   Всего ботов: {total}")
        
        if total > 0:
            cursor.execute("SELECT COUNT(*) FROM bots WHERE status LIKE '%position%'")
            in_position = cursor.fetchone()[0]
            print(f"   В позиции: {in_position}")
            
            cursor.execute("""
                SELECT symbol, status, position_side, entry_price, unrealized_pnl
                FROM bots
                WHERE status LIKE '%position%'
                LIMIT 5
            """)
            print(f"\n   Боты в позиции:")
            for row in cursor.fetchall():
                print(f"      {row[0]} | {row[1]} | {row[2]} | entry={row[3]} | pnl={row[4]}")
    except sqlite3.OperationalError as e:
        print(f"   ❌ Ошибка: {e}")
    
    conn.close()

def check_ai_db():
    """Проверяет ai_data.db"""
    print("\n" + "=" * 80)
    print("ПРОВЕРКА ai_data.db")
    print("=" * 80)
    
    db_path = PROJECT_ROOT / 'data' / 'ai_data.db'
    if not db_path.exists():
        print(f"❌ Файл не найден: {db_path}")
        return
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Проверяем таблицы с трейдами
    trade_tables = ['bot_trades', 'exchange_trades', 'simulated_trades']
    
    print(f"\n📋 Таблицы с трейдами:")
    for table in trade_tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"   {table}: {count} записей")
            
            if count > 0 and table == 'bot_trades':
                cursor.execute("SELECT COUNT(*) FROM bot_trades WHERE status = 'CLOSED'")
                closed = cursor.fetchone()[0]
                print(f"      Закрытых: {closed}")
        except sqlite3.OperationalError:
            print(f"   {table}: таблица не существует")
    
    conn.close()

def check_app_db():
    """Проверяет app_data.db"""
    print("\n" + "=" * 80)
    print("ПРОВЕРКА app_data.db")
    print("=" * 80)
    
    db_path = PROJECT_ROOT / 'data' / 'app_data.db'
    if not db_path.exists():
        print(f"❌ Файл не найден: {db_path}")
        return
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Проверяем таблицы
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    
    print(f"\n📋 Таблицы в базе ({len(tables)}):")
    for table in tables:
        count = check_table(conn, table)
        print(f"   {table}: {count} записей")
    
    # Проверяем positions
    print(f"\n🔍 Детальная проверка positions:")
    try:
        cursor.execute("SELECT COUNT(*) FROM positions")
        total = cursor.fetchone()[0]
        print(f"   Всего позиций: {total}")
        
        if total > 0:
            cursor.execute("""
                SELECT symbol, side, pnl, roi, position_category
                FROM positions
                ORDER BY created_at DESC
                LIMIT 5
            """)
            print(f"\n   Последние 5 позиций:")
            for row in cursor.fetchall():
                print(f"      {row[0]} {row[1]} | pnl={row[2]} | roi={row[3]} | category={row[4]}")
    except sqlite3.OperationalError as e:
        print(f"   ❌ Ошибка: {e}")
    
    conn.close()

if __name__ == '__main__':
    check_bots_db()
    check_ai_db()
    check_app_db()
    
    print("\n" + "=" * 80)
    print("ПРОВЕРКА ЗАВЕРШЕНА")
    print("=" * 80)

