#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-merge hook: защита bot_config.py от перезаписи при git pull
Этот hook выполняется после git merge/pull
Работает на Windows, Linux и macOS
"""

import os
import sys
import shutil
import subprocess

BOT_CONFIG = "configs/bot_config.py"
BACKUP_SUFFIX = ".local_backup"

def main():
    bot_config_path = BOT_CONFIG
    backup_path = bot_config_path + BACKUP_SUFFIX
    
    # Проверяем, существует ли файл
    if not os.path.exists(bot_config_path):
        return 0
    
    try:
        # Проверяем, установлен ли skip-worktree
        result = subprocess.run(
            ['git', 'ls-files', '-v', bot_config_path],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        skip_worktree_set = False
        if result.returncode == 0 and result.stdout.strip():
            skip_worktree_set = result.stdout.strip().startswith('S')
        
        if not skip_worktree_set:
            # Если skip-worktree не установлен, устанавливаем его
            subprocess.run(
                ['git', 'update-index', '--skip-worktree', bot_config_path],
                timeout=5,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        
        # Если есть бэкап локальной версии, восстанавливаем её
        if os.path.exists(backup_path):
            try:
                shutil.copy2(backup_path, bot_config_path)
                # Удаляем бэкап после восстановления
                try:
                    os.remove(backup_path)
                except Exception:
                    pass
                print(f"[post-merge] Восстановлена локальная версия {bot_config_path}")
            except Exception as e:
                print(f"[post-merge] Ошибка восстановления {bot_config_path}: {e}")
                
    except Exception as e:
        # Игнорируем ошибки
        pass
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

