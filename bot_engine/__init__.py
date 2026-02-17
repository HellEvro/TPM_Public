# Торговый движок ботов
# Конфиги только в configs/. Автопатч RSI: подмена устаревшего fallback rsi6h в configs/bot_config.py на делегирование в config_loader.
from pathlib import Path

# Блок с безопасными функциями RSI/тренд (единый источник — config_loader, без fallback rsi6h)
_RSI_SAFE_FUNCTIONS_BLOCK = '''
def get_rsi_from_coin_data(coin_data, timeframe=None):
    """Единый источник истины: bot_engine.config_loader. Без fallback rsi6h — см. config_loader."""
    from bot_engine.config_loader import get_rsi_from_coin_data as _get_rsi
    return _get_rsi(coin_data, timeframe=timeframe)


def get_trend_from_coin_data(coin_data, timeframe=None):
    """Единый источник истины: bot_engine.config_loader. Без fallback trend6h."""
    from bot_engine.config_loader import get_trend_from_coin_data as _get_trend
    return _get_trend(coin_data, timeframe=timeframe)

'''


def _patch_configs_bot_config_rsi_if_needed():
    """
    Если в configs/bot_config.py есть get_rsi_from_coin_data/get_trend_from_coin_data с fallback rsi6h —
    подменяем только эти две функции на делегирование в config_loader. Остальной конфиг не трогаем.
    """
    import re
    _root = Path(__file__).resolve().parent.parent
    _config = _root / "configs" / "bot_config.py"
    if not _config.exists():
        return
    try:
        text = _config.read_text(encoding="utf-8")
    except Exception:
        return
    if "get_rsi_from_coin_data as _get_rsi" in text or "get_trend_from_coin_data as _get_trend" in text:
        return
    if "def get_rsi_from_coin_data" not in text or "def get_trend_from_coin_data" not in text:
        return
    has_old_rsi_fallback = "rsi6h" in text or ("timeframe == '6h'" in text and "get_rsi_from_coin_data" in text)
    if not has_old_rsi_fallback:
        return
    # Блок от первой def get_rsi до следующей топ-уровневой def/class (конец get_trend)
    pattern = re.compile(
        r'(def get_rsi_from_coin_data\(coin_data, timeframe=None\):.*?'
        r'def get_trend_from_coin_data\(coin_data, timeframe=None\):.*?)'
        r'(?=\n(?:def |class )|\Z)',
        re.DOTALL
    )
    match = pattern.search(text)
    if not match:
        return
    new_text = text[: match.start(1)] + _RSI_SAFE_FUNCTIONS_BLOCK.rstrip() + "\n\n" + text[match.end():]
    try:
        _config.write_text(new_text, encoding="utf-8")
    except Exception:
        return


def ensure_rsi_fix_applied():
    """
    Вызвать при старте app/bots: применяет автопатч configs/bot_config.py (RSI без fallback rsi6h).
    Конфиги только в configs/ — bot_engine/bot_config.py удалён.
    """
    _patch_configs_bot_config_rsi_if_needed()


_patch_configs_bot_config_rsi_if_needed()
