#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Проверка размера WAL файлов
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

db_path = PROJECT_ROOT / 'data' / 'bots_data.db'
wal_path = Path(str(db_path) + '-wal')
shm_path = Path(str(db_path) + '-shm')

print("=" * 80)
print("РАЗМЕР ФАЙЛОВ БД:")
print("=" * 80)

if db_path.exists():
    db_size_mb = db_path.stat().st_size / (1024 * 1024)
    db_size_gb = db_size_mb / 1024
    print(f"📊 bots_data.db: {db_size_mb:.2f} MB ({db_size_gb:.2f} GB)")
else:
    print("❌ bots_data.db не найден")

if wal_path.exists():
    wal_size_mb = wal_path.stat().st_size / (1024 * 1024)
    wal_size_gb = wal_size_mb / 1024
    print(f"📊 bots_data.db-wal: {wal_size_mb:.2f} MB ({wal_size_gb:.2f} GB)")
    if wal_size_mb > 100:
        print(f"⚠️ WAL файл очень большой! Нужно выполнить checkpoint")
else:
    print("ℹ️ WAL файл не существует")

if shm_path.exists():
    shm_size_mb = shm_path.stat().st_size / (1024 * 1024)
    print(f"📊 bots_data.db-shm: {shm_size_mb:.2f} MB")
else:
    print("ℹ️ SHM файл не существует")

print("=" * 80)

