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
                logger.debug(f"✅ Загружен список зрелых монет из файла: {len(mature_coins_set)} монет")
                return mature_coins_set
    except Exception as e:
        logger.debug(f"   ⚠️ Не удалось загрузить из файла: {e}")
    
    try:
        # 2. Пробуем импортировать из bots_modules если доступно
        from bots_modules.imports_and_globals import mature_coins_storage
        mature_coins_set = set(mature_coins_storage.keys())
        logger.debug(f"✅ Загружен список зрелых монет из памяти: {len(mature_coins_set)} монет")
        return mature_coins_set
    except ImportError:
        logger.debug("   💡 bots_modules недоступен - список зрелых монет не загружен")
    except Exception as e:
        logger.debug(f"   ⚠️ Ошибка загрузки из памяти: {e}")
    
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
        logger.debug(f"   ⚠️ Ошибка получения настроек из памяти: {e}")
    
    # Fallback: загружаем из файла
    try:
        from bot_engine.storage import load_individual_coin_settings
        all_settings = load_individual_coin_settings() or {}
        normalized_symbol = symbol.upper()
        return all_settings.get(normalized_symbol)
    except Exception as e:
        logger.debug(f"   ⚠️ Ошибка загрузки настроек из файла: {e}")
    
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
        logger.debug(f"   ⚠️ Ошибка получения RSI кэша: {e}")
    
    return None


def get_auto_bot_config() -> Optional[Dict[str, Any]]:
    """
    Получает конфигурацию Auto Bot из bots.py
    
    Returns:
        Словарь с конфигурацией или None
    """
    try:
        from bots_modules.imports_and_globals import bots_data, bots_data_lock
        with bots_data_lock:
            return bots_data.get('auto_bot_config', {})
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"   ⚠️ Ошибка получения конфигурации: {e}")
    
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

