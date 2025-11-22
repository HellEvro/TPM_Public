#!/usr/bin/env python3
"""Utility script to stage, commit, and push repository changes.

Example:
    python scripts/git_commit_push.py "Refactor bot history storage"
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def run_git_command(args: Sequence[str], cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run a git command, returning the completed process object."""
    result = subprocess.run(
        args,
        check=False,
        text=True,
        capture_output=True,
        encoding='utf-8',
        errors='replace',
        cwd=cwd,
    )
    return result


def git_pull_with_merge(cwd: str | None = None) -> None:
    """Выполняет git pull с merge для получения удаленных изменений."""
    # Сначала получаем удаленные изменения
    fetch_result = run_git_command(["git", "fetch"], cwd=cwd)
    if fetch_result.returncode != 0:
        print((fetch_result.stderr or "").strip() or "Не удалось получить удаленные изменения.")
        sys.exit(fetch_result.returncode)
    
    # Проверяем, есть ли удаленные коммиты
    check_result = run_git_command(["git", "rev-list", "--count", "HEAD..@{upstream}"], cwd=cwd)
    has_remote_commits = False
    if check_result.returncode == 0 and check_result.stdout:
        try:
            count = int(check_result.stdout.strip())
            has_remote_commits = count > 0
        except ValueError:
            pass
    
    if has_remote_commits:
        print("Обнаружены удаленные коммиты. Выполняю merge...")
        # Выполняем merge с стратегией --no-edit для автоматического merge commit
        merge_result = run_git_command(["git", "pull", "--no-rebase", "--no-edit"], cwd=cwd)
        if merge_result.returncode != 0:
            print((merge_result.stderr or "").strip() or "Ошибка при merge удаленных изменений.")
            print("Возможно, требуется разрешить конфликты вручную.")
            sys.exit(merge_result.returncode)
        if merge_result.stdout and merge_result.stdout.strip():
            print(merge_result.stdout.strip())


def ensure_changes_present() -> None:
    """Exit early if there are no changes to commit."""
    status = run_git_command(["git", "status", "--porcelain"])
    if status.returncode != 0:
        print((status.stderr or "").strip() or "Не удалось получить статус репозитория.")
        sys.exit(status.returncode)

    if not (status.stdout and status.stdout.strip()):
        print("Изменения отсутствуют — коммит не требуется.")
        sys.exit(0)


def git_add_all(cwd: str | None = None) -> None:
    result = run_git_command(["git", "add", "-A"], cwd=cwd)
    if result.returncode != 0:
        print((result.stderr or "").strip() or "Не удалось подготовить файлы к коммиту.")
        sys.exit(result.returncode)


def git_commit(message: str, cwd: str | None = None) -> None:
    result = run_git_command(["git", "commit", "-m", message], cwd=cwd)
    if result.returncode != 0:
        print((result.stderr or "").strip() or "Коммит не выполнен.")
        sys.exit(result.returncode)
    if result.stdout and result.stdout.strip():
        print(result.stdout.strip())


def git_push(cwd: str | None = None) -> None:
    result = run_git_command(["git", "push"], cwd=cwd)
    if result.returncode != 0:
        print((result.stderr or "").strip() or "Push завершился с ошибкой.")
        sys.exit(result.returncode)
    if result.stdout and result.stdout.strip():
        print(result.stdout.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ставит изменения в git, коммитит и пушит их.",
    )
    parser.add_argument(
        "message",
        help="Описание всех изменений, которые сделал агент.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Шаг 1: Синхронизация в InfoBot_Public
    root_dir = Path(__file__).parent.parent
    sync_script = root_dir / "sync_to_public.py"
    if sync_script.exists():
        print("=" * 80)
        print("СИНХРОНИЗАЦИЯ В ПУБЛИЧНУЮ ВЕРСИЮ")
        print("=" * 80)
        result = subprocess.run(
            [sys.executable, str(sync_script)],
            cwd=str(root_dir),
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
        )
        if result.stdout:
            print(result.stdout)
        if result.returncode != 0:
            print(f"[WARNING] Синхронизация завершилась с ошибкой: {result.stderr}")
        print()
    
    # Шаг 2: Коммит в основном репозитории
    git_pull_with_merge()
    ensure_changes_present()
    git_add_all()
    git_commit(args.message)
    git_push()
    
    # Шаг 3: Коммит в публичном репозитории
    public_dir = root_dir.parent / "InfoBot_Public"
    if public_dir.exists() and (public_dir / ".git").exists():
        print("=" * 80)
        print("КОММИТ И PUSH В ПУБЛИЧНЫЙ РЕПОЗИТОРИЙ")
        print("=" * 80)
        public_cwd = str(public_dir)
        
        # Проверяем наличие изменений
        status = run_git_command(["git", "status", "--porcelain"], cwd=public_cwd)
        if status.returncode == 0 and status.stdout and status.stdout.strip():
            git_pull_with_merge(cwd=public_cwd)
            git_add_all(cwd=public_cwd)
            git_commit(args.message, cwd=public_cwd)
            git_push(cwd=public_cwd)
            print("[OK] Коммит и push в публичный репозиторий выполнены успешно\n")
        else:
            print("Изменения в публичном репозитории отсутствуют — коммит не требуется.\n")


if __name__ == "__main__":
    main()

