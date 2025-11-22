#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для проверки размера таблиц в bots_data.db
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

db_path = PROJECT_ROOT / 'data' / 'bots_data.db'

if not db_path.exists():
    print(f"❌ БД не найдена: {db_path}")
    sys.exit(1)

print("=" * 80)
print(f"АНАЛИЗ РАЗМЕРА БД: {db_path.name}")
print("=" * 80)

# Размер файла
file_size_mb = db_path.stat().st_size / (1024 * 1024)
file_size_gb = file_size_mb / 1024
print(f"\n📊 Общий размер файла: {file_size_mb:.2f} MB ({file_size_gb:.2f} GB)")

# Проверяем WAL файлы
wal_path = Path(str(db_path) + '-wal')
shm_path = Path(str(db_path) + '-shm')

if wal_path.exists():
    wal_size_mb = wal_path.stat().st_size / (1024 * 1024)
    print(f"📊 Размер WAL файла: {wal_size_mb:.2f} MB")
    total_size_mb = file_size_mb + wal_size_mb
    print(f"📊 Общий размер (БД + WAL): {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Получаем список всех таблиц
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [row[0] for row in cursor.fetchall()]

print(f"\n📋 Найдено таблиц: {len(tables)}")
print("\n" + "=" * 80)
print("РАЗМЕР КАЖДОЙ ТАБЛИЦЫ:")
print("=" * 80)

total_rows = 0
table_sizes = []

for table in tables:
    try:
        # Количество строк
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        row_count = cursor.fetchone()[0]
        
        # Размер таблицы (приблизительно)
        cursor.execute(f"SELECT COUNT(*) FROM sqlite_master WHERE name='{table}'")
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        
        # Пытаемся получить размер через page_count
        try:
            cursor.execute(f"PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            cursor.execute(f"PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            table_size_mb = (page_count * page_size) / (1024 * 1024)
        except:
            table_size_mb = 0
        
        # Для больших таблиц получаем примерный размер данных
        if row_count > 0:
            # Берем первую строку для оценки размера
            cursor.execute(f"SELECT * FROM {table} LIMIT 1")
            sample = cursor.fetchone()
            if sample:
                # Очень приблизительная оценка
                estimated_row_size = sum(len(str(v)) if v else 0 for v in sample)
                estimated_size_mb = (row_count * estimated_row_size) / (1024 * 1024)
            else:
                estimated_size_mb = 0
        else:
            estimated_size_mb = 0
        
        table_sizes.append({
            'name': table,
            'rows': row_count,
            'size_mb': max(table_size_mb, estimated_size_mb),
            'columns': len(columns)
        })
        
        total_rows += row_count
        
        print(f"\n📊 {table}:")
        print(f"   Записей: {row_count:,}")
        print(f"   Колонок: {len(columns)}")
        if table_size_mb > 0:
            print(f"   Размер: ~{table_size_mb:.2f} MB")
        elif estimated_size_mb > 0:
            print(f"   Примерный размер: ~{estimated_size_mb:.2f} MB")
        
        # Для больших таблиц показываем топ-5 самых больших колонок
        if row_count > 10000:
            print(f"   ⚠️ Большая таблица!")
            
    except Exception as e:
        print(f"\n❌ Ошибка при анализе таблицы {table}: {e}")

# Сортируем по размеру
table_sizes.sort(key=lambda x: x['size_mb'], reverse=True)

print("\n" + "=" * 80)
print("ТАБЛИЦЫ ПО РАЗМЕРУ (топ-10):")
print("=" * 80)
for i, table_info in enumerate(table_sizes[:10], 1):
    print(f"{i}. {table_info['name']}: {table_info['rows']:,} записей, ~{table_info['size_mb']:.2f} MB")

print(f"\n📊 Всего записей во всех таблицах: {total_rows:,}")

# Проверяем конкретные большие таблицы
print("\n" + "=" * 80)
print("ДЕТАЛЬНЫЙ АНАЛИЗ БОЛЬШИХ ТАБЛИЦ:")
print("=" * 80)

large_tables = ['candles_cache_data', 'bot_trades_history', 'rsi_cache', 'candles_cache']
for table in large_tables:
    if table in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            
            # Для candles_cache_data проверяем размер свечей
            if table == 'candles_cache_data':
                cursor.execute("SELECT COUNT(*) FROM candles_cache_data")
                candles_count = cursor.fetchone()[0]
                print(f"\n📊 {table}:")
                print(f"   Всего свечей: {candles_count:,}")
                if candles_count > 0:
                    avg_size = (file_size_mb * 1024 * 1024) / candles_count if candles_count > 0 else 0
                    print(f"   Средний размер свечи: ~{avg_size:.2f} байт")
            
            # Для bot_trades_history
            elif table == 'bot_trades_history':
                cursor.execute("SELECT COUNT(*) FROM bot_trades_history")
                trades_count = cursor.fetchone()[0]
                print(f"\n📊 {table}:")
                print(f"   Всего сделок: {trades_count:,}")
            
            print(f"   Записей: {count:,}")
        except Exception as e:
            print(f"\n❌ Ошибка при анализе {table}: {e}")

conn.close()

print("\n" + "=" * 80)
print("РЕКОМЕНДАЦИИ:")
print("=" * 80)
if file_size_mb > 1024:
    print("⚠️ БД очень большая (>1 GB)!")
    print("💡 Рекомендуется:")
    print("   1. Проверить, не накапливаются ли данные без очистки")
    print("   2. Рассмотреть архивацию старых данных")
    print("   3. Проверить, не дублируются ли данные")
    if wal_path.exists() and wal_path.stat().st_size > 100 * 1024 * 1024:
        print("   4. WAL файл большой - выполните PRAGMA wal_checkpoint(TRUNCATE)")

print("=" * 80)

