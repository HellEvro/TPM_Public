"""
Живое чтение конфига: перезагрузка bot_config при изменении файла и чтение AIConfig из модуля.
Используется для применения настроек AI/автобота «на лету» без перезапуска процесса.
"""

import os


def _get_bot_config_path():
    """Путь к configs/bot_config.py (все конфиги в configs/)."""
    try:
        project_root = os.path.dirname(os.path.dirname(__import__('bot_engine').__file__))
        return os.path.join(project_root, 'configs', 'bot_config.py')
    except Exception:
        return None


_last_mtime = [0.0]


def reload_bot_config_if_changed():
    """
    Перезагружает конфиг, если configs/bot_config.py изменился.
    Вызывать в начале цикла обработки, чтобы подхватывать настройки из UI без перезапуска.
    """
    path = _get_bot_config_path()
    if not path or not os.path.isfile(path):
        return
    try:
        mtime = os.path.getmtime(path)
        if mtime > _last_mtime[0]:
            _last_mtime[0] = mtime
            from bot_engine.config_loader import reload_config
            reload_config()
    except Exception:
        pass


def get_ai_config_attr(name: str, default=None):
    """Читает атрибут AIConfig (актуальный после reload_bot_config_if_changed)."""
    try:
        from bot_engine.config_loader import AIConfig
        return getattr(AIConfig, name, default)
    except Exception:
        return default
