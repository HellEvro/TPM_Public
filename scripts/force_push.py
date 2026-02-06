#!/usr/bin/env python3
"""Принудительный push всех изменений"""
import subprocess
import sys
import os

def run(cmd, cwd=None):
    """Выполняет команду и возвращает результат"""
    print(f"Выполняю: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if result.stdout:
        print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, flush=True)
    return result.returncode == 0

os.chdir(r"E:\Drive\TRADEBOT\InfoBot")

print("=" * 80)
print("ШАГ 1: Добавляем все файлы")
print("=" * 80)
run(["git", "add", "-A"])

print("\n" + "=" * 80)
print("ШАГ 2: Создаем commit")
print("=" * 80)
run(["git", "commit", "-m", "Исправления фильтра защиты от повторных входов: полная инициализация loss_reentry_info во всех ветках"])

print("\n" + "=" * 80)
print("ШАГ 3: Push в основной репозиторий")
print("=" * 80)
run(["git", "push", "origin", "main"])

print("\n" + "=" * 80)
print("ШАГ 4: Синхронизация в публичный репозиторий")
print("=" * 80)
run([sys.executable, "scripts/sync_to_public.py"])

public_dir = r"E:\Drive\TRADEBOT\InfoBot_Public"
if os.path.exists(public_dir) and os.path.exists(os.path.join(public_dir, ".git")):
    print("\n" + "=" * 80)
    print("ШАГ 5: Commit и push в публичный репозиторий")
    print("=" * 80)
    run(["git", "add", "-A"], cwd=public_dir)
    run(["git", "commit", "-m", "Исправления фильтра защиты от повторных входов"], cwd=public_dir)
    run(["git", "push", "origin", "main"], cwd=public_dir)

print("\n" + "=" * 80)
print("✅ ВСЕ ОПЕРАЦИИ ЗАВЕРШЕНЫ")
print("=" * 80)
