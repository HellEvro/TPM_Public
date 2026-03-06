"""
Патч 014: восстановление конфигов и заглушек после отката репозитория (например до a30f003a от 16.02.2026).

Не перезаписывает существующие configs/keys.py, configs/app_config.py, configs/bot_config.py.
Создаёт только недостающие файлы из примеров и заглушку app/config.py.
Для пользователей в паблике и локально: после git pull / reset конфиги не в git — этот патч
восстанавливает работоспособность, если файлы были удалены или затронуты.
"""
from pathlib import Path
import shutil


def apply(project_root: Path) -> bool:
    root = Path(project_root)
    app_config_py = root / "app" / "config.py"
    configs_dir = root / "configs"
    app_dir = root / "app"

    # 1. Заглушка app/config.py — только если отсутствует
    if not app_config_py.exists():
        app_dir.mkdir(parents=True, exist_ok=True)
        stub = (
            "# Реальный конфиг в configs/app_config.py\n"
            "from configs.app_config import *  # noqa: F401, F403\n"
        )
        app_config_py.write_text(stub, encoding="utf-8")

    # 2. configs/app_config.py — из примера только если нет
    app_config_example = configs_dir / "app_config.example.py"
    app_config_target = configs_dir / "app_config.py"
    if not app_config_target.exists() and app_config_example.exists():
        configs_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(app_config_example, app_config_target)

    # 3. configs/keys.py — из примера только если нет
    keys_example = configs_dir / "keys.example.py"
    keys_target = configs_dir / "keys.py"
    if not keys_target.exists() and keys_example.exists():
        configs_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(keys_example, keys_target)

    # 4. configs/bot_config.py — из примера только если нет
    bot_config_example = configs_dir / "bot_config.example.py"
    bot_config_target = configs_dir / "bot_config.py"
    if not bot_config_target.exists() and bot_config_example.exists():
        configs_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bot_config_example, bot_config_target)

    # 5. bot_engine/bot_config.py — в этой версии может требоваться; создаём из configs если есть example
    bot_engine_config = root / "bot_engine" / "bot_config.py"
    if not bot_engine_config.exists() and bot_config_target.exists():
        root.joinpath("bot_engine").mkdir(parents=True, exist_ok=True)
        shutil.copy2(bot_config_target, bot_engine_config)

    return True
