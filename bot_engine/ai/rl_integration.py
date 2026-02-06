"""
RL (Reinforcement Learning) — опциональное дополнение (AI_RL_ENABLED).

Обёртка над rl_agent для использования в пайплайне.
При AI_RL_ENABLED=False не дергаем RL; при True — заглушка под будущую интеграцию.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger('RL')


def _rl_enabled() -> bool:
    try:
        from bot_engine.config_loader import AIConfig
        return bool(getattr(AIConfig, 'AI_RL_ENABLED', False))
    except Exception:
        return False


def get_rl_signal(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Возвращает RL-сигнал для символа.

    При AI_RL_ENABLED=False возвращает None (не используется).
    Иначе — заглушка под будущую интеграцию RLTrader.
    """
    if not _rl_enabled():
        return None
    # Заглушка: интеграция RLTrader в workflow — позже
    return {
        'symbol': symbol,
        'signal': 'neutral',
        'score': 0.0,
        'confidence': 0.0,
        'enabled': True,
        'status': 'not_implemented',
    }
