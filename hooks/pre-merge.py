#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-merge hook: создание бэкапа bot_config.py перед merge/pull
Этот hook выполняется перед git merge/pull
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
        
        if skip_worktree_set:
            # Создаем бэкап локальной версии перед merge
            try:
                shutil.copy2(bot_config_path, backup_path)
            except Exception as e:
                print(f"[pre-merge] Ошибка создания бэкапа {bot_config_path}: {e}")
                
    except Exception:
        # Игнорируем ошибки
        pass
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

