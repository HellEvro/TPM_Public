#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Установка git hooks для защиты bot_config.py от перезаписи при git pull
"""

import os
import sys
import shutil
from pathlib import Path

def main():
    """Устанавливает git hooks из папки hooks/ в .git/hooks/"""
    script_dir = Path(__file__).parent.parent
    hooks_source = script_dir / "hooks"
    hooks_target = script_dir / ".git" / "hooks"
    
    if not hooks_source.exists():
        print(f"❌ Папка hooks/ не найдена: {hooks_source}")
        return 1
    
    if not hooks_target.exists():
        print(f"❌ Папка .git/hooks/ не найдена: {hooks_target}")
        print("   Убедитесь, что вы находитесь в git репозитории")
        return 1
    
    hooks_to_install = [
        "post-merge",
        "pre-merge",
        "post-merge.py",
        "pre-merge.py",
    ]
    
    installed = 0
    for hook_name in hooks_to_install:
        source_file = hooks_source / hook_name
        target_file = hooks_target / hook_name
        
        if source_file.exists():
            try:
                shutil.copy2(source_file, target_file)
                # Делаем файл исполняемым (на Unix-системах)
                if os.name != 'nt':
                    os.chmod(target_file, 0o755)
                print(f"✅ Установлен: {hook_name}")
                installed += 1
            except Exception as e:
                print(f"❌ Ошибка установки {hook_name}: {e}")
        else:
            print(f"⚠️  Файл не найден: {source_file}")
    
    if installed > 0:
        print(f"\n✅ Установлено хуков: {installed}/{len(hooks_to_install)}")
        print("   Git hooks для защиты bot_config.py активированы!")
    else:
        print("\n❌ Не удалось установить хуки")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

