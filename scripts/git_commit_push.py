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

# Принудительно отключаем буферизацию вывода
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)


def run_git_command(args: Sequence[str], cwd: str | None = None, capture: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a git command, returning the completed process object."""
    if capture:
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
    else:
        # Выполняем команду с выводом в реальном времени
        result = subprocess.run(
            args,
            check=False,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=cwd,
        )
        # Создаем объект с пустыми stdout/stderr для совместимости
        return type('CompletedProcess', (), {
            'returncode': result.returncode,
            'stdout': '',
            'stderr': ''
        })()


def git_pull_with_merge(cwd: str | None = None) -> None:
    """Выполняет git pull с merge для получения удаленных изменений."""
    # КРИТИЧНО: fetch и pull БЕЗ capture_output для использования credential manager
    print("[FETCH] Получаю удаленные изменения...", flush=True)
    fetch_result = subprocess.run(
        ["git", "fetch"],
        cwd=cwd,
        check=False,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    if fetch_result.returncode != 0:
        print(f"⚠️ Не удалось получить удаленные изменения (код {fetch_result.returncode})", flush=True)
        # Не прерываем выполнение, продолжаем
    
    # Проверяем, есть ли удаленные коммиты (эту команду можно с capture)
    check_result = run_git_command(["git", "rev-list", "--count", "HEAD..@{upstream}"], cwd=cwd)
    has_remote_commits = False
    if check_result.returncode == 0 and check_result.stdout:
        try:
            count = int(check_result.stdout.strip())
            has_remote_commits = count > 0
        except ValueError:
            pass
    
    if has_remote_commits:
        print("Обнаружены удаленные коммиты. Выполняю merge...", flush=True)
        # КРИТИЧНО: pull БЕЗ capture_output для использования credential manager
        merge_result = subprocess.run(
            ["git", "pull", "--no-rebase", "--no-edit"],
            cwd=cwd,
            check=False,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        if merge_result.returncode != 0:
            print(f"⚠️ Ошибка при merge удаленных изменений (код {merge_result.returncode})", flush=True)
            print("⚠️ Возможно, требуется разрешить конфликты вручную.", flush=True)
            # Не прерываем выполнение, продолжаем


def ensure_changes_present() -> None:
    """Exit early if there are no changes to commit."""
    # Сначала добавляем все файлы, чтобы проверить реальное состояние
    run_git_command(["git", "add", "-A"])
    
    status = run_git_command(["git", "status", "--porcelain"])
    if status.returncode != 0:
        print((status.stderr or "").strip() or "Не удалось получить статус репозитория.")
        sys.exit(status.returncode)

    # Проверяем также staged изменения
    diff_staged = run_git_command(["git", "diff", "--cached", "--name-only"])
    has_unstaged = status.stdout and status.stdout.strip()
    has_staged = diff_staged.stdout and diff_staged.stdout.strip()
    
    if not has_unstaged and not has_staged:
        print("Изменения отсутствуют — коммит не требуется.")
        sys.exit(0)


def git_add_all(cwd: str | None = None) -> None:
    result = run_git_command(["git", "add", "-A"], cwd=cwd, capture=True)
    if result.returncode != 0:
        error_msg = (result.stderr or "").strip() or "Не удалось подготовить файлы к коммиту."
        print(f"❌ {error_msg}", flush=True)
        sys.exit(result.returncode)
    print("✅ Файлы добавлены в staging", flush=True)


def git_commit(message: str, cwd: str | None = None) -> None:
    # Выполняем commit БЕЗ перехвата вывода
    result = subprocess.run(["git", "commit", "-m", message], cwd=cwd, check=False, text=True, encoding='utf-8', errors='replace')
    if result.returncode != 0:
        # Проверяем причину ошибки
        check_result = run_git_command(["git", "status", "--porcelain"], cwd=cwd)
        if check_result.stdout and not check_result.stdout.strip():
            print("ℹ️ Изменения отсутствуют — коммит не требуется.", flush=True)
            return
        print(f"❌ ОШИБКА коммита (код {result.returncode})", flush=True)
        sys.exit(result.returncode)
    print("✅ Коммит создан успешно", flush=True)


def git_push(cwd: str | None = None) -> None:
    print(f"[PUSH] Выполняю git push (cwd={cwd or os.getcwd()})...", flush=True)
    # КРИТИЧНО: push БЕЗ capture_output - чтобы git мог использовать credential manager из .gitconfig
    # Это позволяет использовать сохраненные пароли (как в Sourcetree)
    result = subprocess.run(
        ["git", "push", "origin", "main"],
        cwd=cwd,
        check=False,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    if result.returncode != 0:
        print(f"[PUSH] origin/main не сработал (код {result.returncode}), пробую origin/master...", flush=True)
        result = subprocess.run(
            ["git", "push", "origin", "master"],
            cwd=cwd,
            check=False,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
    if result.returncode != 0:
        print(f"[PUSH] origin/master не сработал (код {result.returncode}), пробую git push без параметров...", flush=True)
        result = subprocess.run(
            ["git", "push"],
            cwd=cwd,
            check=False,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
    if result.returncode != 0:
        print(f"❌ ОШИБКА PUSH (код {result.returncode})", flush=True)
        sys.exit(result.returncode)
    print(f"[PUSH] ✅ Успешно выполнен (код {result.returncode})", flush=True)


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
    sys.stdout.flush()  # Принудительно выводим буфер
    
    # Шаг 1: Синхронизация в InfoBot_Public
    root_dir = Path(__file__).parent.parent
    sync_script = root_dir / "scripts" / "sync_to_public.py"
    if sync_script.exists():
        print("=" * 80, flush=True)
        print("СИНХРОНИЗАЦИЯ В ПУБЛИЧНУЮ ВЕРСИЮ", flush=True)
        print("=" * 80, flush=True)
        result = subprocess.run(
            [sys.executable, str(sync_script)],
            cwd=str(root_dir),
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
        )
        if result.stdout:
            print(result.stdout, flush=True)
        if result.returncode != 0:
            print(f"[WARNING] Синхронизация завершилась с ошибкой: {result.stderr}", flush=True)
        print(flush=True)
    
    # Шаг 2: Коммит в основном репозитории
    print("=" * 80, flush=True)
    print("ШАГ 2: КОММИТ В ОСНОВНОМ РЕПОЗИТОРИИ", flush=True)
    print("=" * 80, flush=True)
    
    try:
        git_pull_with_merge()
    except SystemExit:
        pass  # Игнорируем системные выходы при ошибках fetch
    
    print("\n[ШАГ 1] Добавляем все файлы (git add -A)...", flush=True)
    git_add_all()
    
    # Проверяем наличие изменений после добавления
    print("[ШАГ 2] Проверяем наличие изменений...", flush=True)
    status = run_git_command(["git", "status", "--porcelain"])
    diff_staged = run_git_command(["git", "diff", "--cached", "--name-only"])
    has_unstaged = status.stdout and status.stdout.strip()
    has_staged = diff_staged.stdout and diff_staged.stdout.strip()
    
    if status.stdout:
        print(f"Unstaged файлы: {status.stdout.strip()}", flush=True)
    if diff_staged.stdout:
        print(f"Staged файлы: {diff_staged.stdout.strip()}", flush=True)
    
    if not has_unstaged and not has_staged:
        print("⚠️ Изменения отсутствуют — коммит не требуется.", flush=True)
    else:
        print(f"\n[ШАГ 3] Создаем commit с сообщением: {args.message}", flush=True)
        git_commit(args.message)
        print(f"\n[ШАГ 4] Выполняем push в основной репозиторий...", flush=True)
        git_push()
        print("\n✅ Коммит и push в основной репозиторий выполнены успешно\n", flush=True)
    
    # Шаг 3: Коммит в публичном репозитории
    public_dir = root_dir.parent / "InfoBot_Public"
    if public_dir.exists() and (public_dir / ".git").exists():
        print("=" * 80, flush=True)
        print("КОММИТ И PUSH В ПУБЛИЧНЫЙ РЕПОЗИТОРИЙ", flush=True)
        print("=" * 80, flush=True)
        public_cwd = str(public_dir)
        
        # Проверяем наличие изменений
        status = run_git_command(["git", "status", "--porcelain"], cwd=public_cwd)
        if status.returncode == 0 and status.stdout and status.stdout.strip():
            try:
                git_pull_with_merge(cwd=public_cwd)
            except SystemExit:
                pass
            git_add_all(cwd=public_cwd)
            git_commit(args.message, cwd=public_cwd)
            git_push(cwd=public_cwd)
            print("✅ Коммит и push в публичный репозиторий выполнены успешно\n", flush=True)
        else:
            print("⚠️ Изменения в публичном репозитории отсутствуют — коммит не требуется.\n", flush=True)


if __name__ == "__main__":
    main()

