#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
П.4 REVERTED_COMMITS_FIXES: создание/дополнение bot_config.py из bot_config.example.py.

- Если bot_config.py нет — копируется bot_config.example.py.
- Запуск: из корня проекта «python scripts/patch_bot_config.py» или вызов из bot_engine при импорте.
"""
from pathlib import Path
import shutil
import sys

# Корень проекта = родитель папки scripts. Конфиги в configs/
ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs"
BOT_CONFIG = CONFIGS_DIR / "bot_config.py"
EXAMPLE = CONFIGS_DIR / "bot_config.example.py"


def ensure_bot_config_from_example():
    """
    Создаёт configs/bot_config.py из configs/bot_config.example.py, если bot_config.py отсутствует.
    Не перезаписывает существующий bot_config.py.
    """
    if BOT_CONFIG.exists():
        return True
    if not EXAMPLE.exists():
        return False
    try:
        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(EXAMPLE, BOT_CONFIG)
        return True
    except OSError:
        return False


def main():
    existed = BOT_CONFIG.exists()
    ok = ensure_bot_config_from_example()
    if not ok:
        print("Ошибка: configs/bot_config.example.py не найден или не удалось создать configs/bot_config.py", file=sys.stderr)
        return 1
    if not existed and BOT_CONFIG.exists():
        print("Создан configs/bot_config.py из configs/bot_config.example.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
