# Торговый движок ботов
# П.4 REVERTED_COMMITS_FIXES: автопатч — создание bot_config.py из example при первом импорте
def _ensure_bot_config():
    from pathlib import Path
    import shutil
    _root = Path(__file__).resolve().parent.parent
    _config = _root / "bot_engine" / "bot_config.py"
    _example = _root / "bot_engine" / "bot_config.example.py"
    if _config.exists():
        return
    if _example.exists():
        try:
            shutil.copy2(_example, _config)
        except OSError:
            pass

_ensure_bot_config()
