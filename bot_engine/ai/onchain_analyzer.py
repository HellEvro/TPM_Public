"""
On-Chain Analysis — опциональное дополнение (можно отключить).

Модуль для анализа ончейн-данных:
- активность китов (Glassnode, Whale Alert)
- network metrics (active addresses, tx count)
- интеграция с торговыми решениями

Настройки в bot_config: AI_ONCHAIN_*.
При AI_ONCHAIN_ENABLED=False все функции возвращают нейтральный результат и не дергают API.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger('OnChain')


def _onchain_enabled() -> bool:
    """Проверка, включён ли On-Chain Analysis в конфиге."""
    try:
        from bot_engine.config_loader import AIConfig
        return bool(getattr(AIConfig, 'AI_ONCHAIN_ENABLED', False))
    except Exception:
        return False


def get_onchain_signal(symbol: str) -> Dict[str, Any]:
    """
    Возвращает on-chain сигнал для символа.

    При AI_ONCHAIN_ENABLED=False возвращает нейтральный stub (enabled=False).
    Иначе — заглушка под будущую интеграцию (Glassnode, Whale Alert).
    Использовать только при включённом дополнении.

    Args:
        symbol: Символ монеты (BTC, ETH, …)

    Returns:
        {
            'symbol': str,
            'signal': 'bullish' | 'bearish' | 'neutral',
            'score': float (-1..1),
            'confidence': float (0..1),
            'enabled': bool,
            'sources': list,
            'timestamp': str,
            'status': 'ok' | 'not_implemented' | 'disabled',
        }
    """
    if not _onchain_enabled():
        return {
            'symbol': symbol,
            'signal': 'neutral',
            'score': 0.0,
            'confidence': 0.0,
            'enabled': False,
            'sources': [],
            'timestamp': datetime.now().isoformat(),
            'status': 'disabled',
        }
    # Заглушка под будущую интеграцию (Glassnode, Whale Alert)
    try:
        from bot_engine.config_loader import AIConfig
        _ = (
            getattr(AIConfig, 'AI_ONCHAIN_GLASSNODE_API_KEY', ''),
            getattr(AIConfig, 'AI_ONCHAIN_WHALE_ALERT_API_KEY', ''),
        )
    except Exception:
        pass
    return {
        'symbol': symbol,
        'signal': 'neutral',
        'score': 0.0,
        'confidence': 0.0,
        'enabled': True,
        'sources': [],
        'timestamp': datetime.now().isoformat(),
        'status': 'not_implemented',
        'message': 'On-Chain API integration pending (Glassnode, Whale Alert)',
    }


def integrate_onchain_signal(
    symbol: str,
    current_signal: Dict[str, Any],
    onchain_weight: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Интегрирует on-chain сигнал в текущий торговый сигнал.

    При AI_ONCHAIN_ENABLED=False возвращает current_signal без изменений (onchain_used=False).

    Args:
        symbol: Символ монеты
        current_signal: Текущий сигнал (должен содержать 'score' или аналог)
        onchain_weight: Вес on-chain (0–1). Если None — из AIConfig.AI_ONCHAIN_WEIGHT

    Returns:
        Обновлённый сигнал с ключами onchain_used, onchain_data.
    """
    if not _onchain_enabled():
        return {**current_signal, 'onchain_used': False, 'onchain_data': None}
    data = get_onchain_signal(symbol)
    if data.get('status') == 'disabled' or data.get('confidence', 0) < 0.1:
        return {**current_signal, 'onchain_used': False, 'onchain_data': data}
    try:
        from bot_engine.config_loader import AIConfig
        w = onchain_weight if onchain_weight is not None else getattr(AIConfig, 'AI_ONCHAIN_WEIGHT', 0.15)
    except Exception:
        w = 0.15
    orig = current_signal.get('score', 0)
    oc_score = data.get('score', 0) * 100
    combined = orig * (1 - w) + oc_score * w
    out = {
        **current_signal,
        'score': combined,
        'onchain_used': True,
        'onchain_data': data,
    }
    return out
