"""
Общие AI-фильтры для симуляций и бэктестов.

Повторяет ключевые проверки из боевого контура:
- RSI временной фильтр
- Exit Scam (анти pump/dump)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from bot_engine.filters import check_rsi_time_filter
from bot_engine.maturity_checker import check_coin_maturity


def _normalize_candle(candle: Dict[str, Any]) -> Dict[str, float]:
    return {
        'open': float(candle.get('open', 0) or 0),
        'close': float(candle.get('close', 0) or 0),
        'high': float(candle.get('high', candle.get('close', 0)) or 0),
        'low': float(candle.get('low', candle.get('close', 0)) or 0),
        'time': candle.get('time')
    }


def _get_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'on'}
    return bool(value)


def _check_global_switches(config: Dict[str, Any]) -> Tuple[bool, str]:
    trading_enabled = _get_bool(config.get('trading_enabled', True), True)
    if not trading_enabled:
        return False, 'Торговля отключена'

    use_test_server = _get_bool(config.get('use_test_server', False), False)
    if use_test_server:
        return False, 'Режим тестового сервера'

    return True, 'Торговля включена'


def _check_scope(symbol: str, config: Dict[str, Any]) -> Tuple[bool, str]:
    scope = config.get('scope', 'all')
    whitelist = config.get('whitelist', []) or []
    blacklist = config.get('blacklist', []) or []
    whitelist = [coin.upper() for coin in whitelist]
    blacklist = [coin.upper() for coin in blacklist]
    symbol_up = symbol.upper()

    if scope == 'whitelist':
        if symbol_up not in whitelist:
            return False, f'Scope whitelist блокирует {symbol}'
        return True, 'Scope whitelist разрешает'

    if scope == 'blacklist':
        if symbol_up in blacklist:
            return False, f'Scope blacklist блокирует {symbol}'
        return True, 'Scope blacklist разрешает'

    # scope == 'all'
    if symbol_up in blacklist:
        return False, f'Scope all/blacklist блокирует {symbol}'
    return True, 'Scope all'


def _check_trend(signal: str, trend: Optional[str], config: Dict[str, Any]) -> Tuple[bool, str]:
    trend = (trend or 'NEUTRAL').upper()
    if signal == 'ENTER_LONG' and config.get('avoid_down_trend', False) and trend == 'DOWN':
        return False, 'avoid_down_trend'
    if signal == 'ENTER_SHORT' and config.get('avoid_up_trend', False) and trend == 'UP':
        return False, 'avoid_up_trend'
    return True, 'Trend ok'


def _check_maturity(symbol: str, candles: List[Dict[str, Any]], config: Dict[str, Any]) -> Tuple[bool, str]:
    if not config.get('enable_maturity_check', True):
        return True, 'Maturity disabled'
    if not candles:
        return False, 'Нет свечей для проверки зрелости'
    try:
        result = check_coin_maturity(symbol, candles, config)
        if result.get('is_mature'):
            return True, 'Maturity ok'
        return False, f"Молодая монета: {result.get('reason', '')}"
    except Exception as exc:  # pragma: no cover
        return False, f'Ошибка maturity: {exc}'


def run_rsi_time_filter(
    candles: List[Dict[str, Any]],
    current_rsi: float,
    signal: str,
    config: Dict[str, Any],
) -> Tuple[bool, str]:
    if not candles:
        return False, 'Нет свечей для временного фильтра'
    try:
        result = check_rsi_time_filter(candles, current_rsi, signal, config)
        return bool(result.get('allowed')), result.get('reason', '')
    except Exception as exc:  # pragma: no cover
        return False, f'Ошибка временного фильтра: {exc}'


def run_exit_scam_filter(
    candles: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Tuple[bool, str]:
    from bot_engine.config_loader import get_config_value
    exit_scam_enabled = get_config_value(config, 'exit_scam_enabled')
    if not exit_scam_enabled:
        return True, 'ExitScam отключен'

    exit_scam_candles = get_config_value(config, 'exit_scam_candles')
    exit_scam_candles = int(exit_scam_candles) if exit_scam_candles is not None else None
    single_candle_percent = get_config_value(config, 'exit_scam_single_candle_percent')
    multi_candle_count = get_config_value(config, 'exit_scam_multi_candle_count')
    multi_candle_percent = get_config_value(config, 'exit_scam_multi_candle_percent')
    if exit_scam_candles is None or single_candle_percent is None or multi_candle_count is None or multi_candle_percent is None:
        return False, 'ExitScam: в конфиге не заданы exit_scam_* параметры'
    single_candle_percent = float(single_candle_percent)
    multi_candle_count = int(multi_candle_count)
    multi_candle_percent = float(multi_candle_percent)

    if len(candles) < exit_scam_candles:
        return False, f'Недостаточно свечей для ExitScam ({len(candles)}/{exit_scam_candles})'

    normalized = [_normalize_candle(c) for c in candles[-exit_scam_candles:]]

    for candle in normalized:
        open_p = candle['open']
        close_p = candle['close']
        if open_p <= 0:
            continue
        change_pct = abs((close_p - open_p) / open_p) * 100
        if change_pct > single_candle_percent:
            return False, f'ExitScam: свеча изменилась на {change_pct:.1f}% (> {single_candle_percent}%)'

    if len(normalized) >= multi_candle_count:
        subset = normalized[-multi_candle_count:]
        first_open = subset[0]['open']
        last_close = subset[-1]['close']
        if first_open > 0:
            total_change = abs((last_close - first_open) / first_open) * 100
            if total_change > multi_candle_percent:
                return False, f'ExitScam: {multi_candle_count} свечей изменились на {total_change:.1f}% (> {multi_candle_percent}%)'

    return True, 'ExitScam пройден'


def apply_entry_filters(
    symbol: str,
    candles: List[Dict[str, Any]],
    current_rsi: float,
    signal: str,
    config: Dict[str, Any],
    trend: Optional[str] = None,
) -> Tuple[bool, str]:
    reason_parts: List[str] = []

    switches_allowed, switches_reason = _check_global_switches(config)
    reason_parts.append(switches_reason)
    if not switches_allowed:
        return False, f"{symbol}: {switches_reason}"

    scope_allowed, scope_reason = _check_scope(symbol, config)
    reason_parts.append(scope_reason)
    if not scope_allowed:
        return False, f"{symbol}: {scope_reason}"

    trend_allowed, trend_reason = _check_trend(signal, trend, config)
    reason_parts.append(trend_reason)
    if not trend_allowed:
        return False, f"{symbol}: {trend_reason}"

    maturity_allowed, maturity_reason = _check_maturity(symbol, candles, config)
    reason_parts.append(maturity_reason)
    if not maturity_allowed:
        return False, f"{symbol}: {maturity_reason}"

    allowed_rsi, rsi_reason = run_rsi_time_filter(candles, current_rsi, signal, config)
    reason_parts.append(rsi_reason)
    if not allowed_rsi:
        return False, f"{symbol}: {rsi_reason}"

    allowed_exit, exit_reason = run_exit_scam_filter(candles, config)
    reason_parts.append(exit_reason)
    if not allowed_exit:
        return False, f"{symbol}: {exit_reason}"

    return True, ', '.join(reason_parts)

