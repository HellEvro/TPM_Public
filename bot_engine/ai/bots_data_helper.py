#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Вспомогательный модуль для доступа к данным, подготовленным bots.py

Этот модуль предоставляет единый интерфейс для доступа к данным,
которые bots.py подготавливает при запуске:
- Зрелые монеты (mature_coins_storage)
- Индивидуальные настройки монет (individual_coin_settings)
- RSI данные (coins_rsi_data)
- Кэш свечей (candles_cache)
- Конфигурации (auto_bot_config, system_config)
"""

import os
import json
import logging
from typing import Dict, Set, Optional, Any

logger = logging.getLogger('AI.BotsDataHelper')


def get_mature_coins() -> Set[str]:
    """
    Получает список зрелых монет из bots.py
    
    Returns:
        Множество символов зрелых монет
    """
    mature_coins_set = set()
    
    try:
        # 1. Пробуем загрузить из файла напрямую
        mature_coins_file = os.path.join('data', 'mature_coins.json')
        if os.path.exists(mature_coins_file):
            with open(mature_coins_file, 'r', encoding='utf-8') as f:
                mature_coins_data = json.load(f)
                mature_coins_set = set(mature_coins_data.keys())
                pass
                return mature_coins_set
    except Exception as e:
        pass
    
    try:
        # 2. Пробуем импортировать из bots_modules если доступно
        from bots_modules.imports_and_globals import mature_coins_storage
        mature_coins_set = set(mature_coins_storage.keys())
        pass
        return mature_coins_set
    except ImportError:
        pass
    except Exception as e:
        pass
    
    return mature_coins_set


def get_individual_coin_settings(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Получает индивидуальные настройки монеты из bots.py
    
    Args:
        symbol: Символ монеты
        
    Returns:
        Словарь с настройками или None
    """
    if not symbol:
        return None
    
    try:
        from bots_modules.imports_and_globals import get_individual_coin_settings
        settings = get_individual_coin_settings(symbol)
        if settings:
            return settings
    except ImportError:
        pass
    except Exception as e:
        pass
    
    # Fallback: загружаем из файла
    try:
        from bot_engine.storage import load_individual_coin_settings
        all_settings = load_individual_coin_settings() or {}
        normalized_symbol = symbol.upper()
        return all_settings.get(normalized_symbol)
    except Exception as e:
        pass
    
    return None


def get_rsi_cache() -> Optional[Dict[str, Any]]:
    """
    Получает кэш RSI данных из bots.py
    
    Returns:
        Словарь с RSI данными или None
    """
    try:
        from bots_modules.imports_and_globals import coins_rsi_data, rsi_data_lock
        with rsi_data_lock:
            return coins_rsi_data.get('candles_cache', {})
    except ImportError:
        pass
    except Exception as e:
        pass
    
    return None


def get_auto_bot_config() -> Optional[Dict[str, Any]]:
    """
    Получает конфигурацию Auto Bot — единый источник для bots.py и ai.py.
    - При запуске bots.py: из bots_data (загружено из configs/bot_config.py).
    - При отдельном запуске ai.py: fallback на БД (фильтры), затем на configs/bot_config.py
      (DEFAULT_AUTO_BOT_CONFIG), чтобы ExitScam, AI пороги и прочие настройки совпадали с ботами.
    """
    try:
        from bots_modules.imports_and_globals import bots_data, bots_data_lock
        with bots_data_lock:
            cfg = bots_data.get('auto_bot_config')
            if cfg is not None and isinstance(cfg, dict):
                return cfg
    except ImportError:
        pass
    except Exception as e:
        pass

    # Fallback при отдельном запуске ai.py: загружаем из БД (фильтры)
    try:
        from bot_engine.bots_database import get_bots_database
        db = get_bots_database()
        filters = db.load_coin_filters()
        if filters:
            return filters
    except Exception as e:
        pass

    # Fallback: конфиг из bot_config.py (тот же источник, что и у bots.py)
    try:
        from copy import deepcopy
        from bot_engine.config_loader import DEFAULT_AUTO_BOT_CONFIG
        return deepcopy(DEFAULT_AUTO_BOT_CONFIG)
    except Exception as e:
        pass

    return None


def is_bots_service_available() -> bool:
    """
    Проверяет доступность сервиса bots.py
    
    Returns:
        True если bots.py запущен и доступен
    """
    try:
        from bots_modules.imports_and_globals import system_initialized
        return system_initialized
    except ImportError:
        return False

