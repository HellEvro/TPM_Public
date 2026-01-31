"""
Живое чтение конфига: перезагрузка bot_config при изменении файла и чтение AIConfig из модуля.
Используется для применения настроек AI/автобота «на лету» без перезапуска процесса.
"""

import os
import sys
import importlib


def _get_bot_config_path():
    """Путь к bot_config.py (не импортирует bot_config)."""
    try:
        bot_engine_dir = os.path.dirname(__import__('bot_engine').__file__)
        return os.path.join(bot_engine_dir, 'bot_config.py')
    except Exception:
        return None


_last_mtime = [0.0]


def reload_bot_config_if_changed():
    """
    Перезагружает bot_engine.bot_config, если файл bot_config.py изменился.
    Вызывать в начале цикла обработки (process_auto_bot_signals / auto_trainer._run),
    чтобы подхватывать сохранённые из UI настройки без перезапуска.
    """
    path = _get_bot_config_path()
    if not path or not os.path.isfile(path):
        return
    try:
        mtime = os.path.getmtime(path)
        if mtime > _last_mtime[0]:
            _last_mtime[0] = mtime
            mod = sys.modules.get('bot_engine.bot_config')
            if mod is not None:
                importlib.reload(mod)
    except Exception:
        pass


def get_ai_config_attr(name: str, default=None):
    """
    Читает атрибут AIConfig из текущего (возможно перезагруженного) модуля bot_config.
    После вызова reload_bot_config_if_changed() возвращает актуальные значения с диска.
    """
    try:
        mod = sys.modules.get('bot_engine.bot_config')
        if mod is None:
            import bot_engine.bot_config  # noqa: F401
            mod = sys.modules['bot_engine.bot_config']
        return getattr(mod.AIConfig, name, default)
    except Exception:
        return default
