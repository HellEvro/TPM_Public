#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Синхронизация текущего состояния приватного репо в InfoBot_Public и push.

Использовать после отката приватного репо (или после коммитов), чтобы паблик
совпадал с приватным по содержимому. Запускать из корня InfoBot.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT.parent / "InfoBot_Public"


def run(cmd, cwd=None):
    cwd = cwd or str(ROOT)
    r = subprocess.run(cmd, shell=False, cwd=cwd, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        print(r.stderr or r.stdout, file=sys.stderr)
    return r.returncode


def main():
    if not PUBLIC.exists():
        print(f"Папка не найдена: {PUBLIC}", file=sys.stderr)
        sys.exit(1)
    if not (PUBLIC / ".git").exists():
        print(f"В {PUBLIC} не найден .git", file=sys.stderr)
        sys.exit(1)

    # 1) Синк файлов
    sync_script = ROOT / "scripts" / "sync_to_public.py"
    if sync_script.exists():
        print("Синхронизация в InfoBot_Public...")
        if run([sys.executable, str(sync_script)], cwd=str(ROOT)) != 0:
            sys.exit(1)
    else:
        print("sync_to_public.py не найден", file=sys.stderr)
        sys.exit(1)

    # 2) В паблике: add, commit, push
    print("Коммит и push в публичный репозиторий...")
    if run(["git", "add", "-A"], cwd=str(PUBLIC)) != 0:
        sys.exit(1)
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(PUBLIC), capture_output=True, text=True, encoding="utf-8"
    )
    if not (status.stdout and status.stdout.strip()):
        print("Изменений в паблике нет — коммит не требуется.")
        return
    msg = sys.argv[1] if len(sys.argv) > 1 else "Sync: состояние после отката к f0a0c509 + п.1 RSI из конфига"
    if run(["git", "commit", "-m", msg], cwd=str(PUBLIC)) != 0:
        sys.exit(1)
    if run(["git", "push", "origin", "main"], cwd=str(PUBLIC)) != 0:
        sys.exit(1)
    print("Готово: паблик обновлён и запушен.")


if __name__ == "__main__":
    main()
