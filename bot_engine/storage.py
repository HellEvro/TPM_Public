"""
Управление хранением данных (RSI кэш, состояние ботов, зрелые монеты)
"""

import os
import json
import logging
import time
import threading
import importlib
from datetime import datetime

logger = logging.getLogger('Storage')

# Блокировки файлов для предотвращения одновременной записи
_file_locks = {}
_lock_lock = threading.Lock()

def _get_file_lock(filepath):
    """Получить блокировку для файла"""
    with _lock_lock:
        if filepath not in _file_locks:
            _file_locks[filepath] = threading.Lock()
        return _file_locks[filepath]

# Пути к файлам
RSI_CACHE_FILE = 'data/rsi_cache.json'
BOTS_STATE_FILE = 'data/bots_state.json'
INDIVIDUAL_COIN_SETTINGS_FILE = 'data/individual_coin_settings.json'
MATURE_COINS_FILE = 'data/mature_coins.json'
# ❌ ОТКЛЮЧЕНО: optimal_ema удален (EMA фильтр убран)
# OPTIMAL_EMA_FILE = 'data/optimal_ema.json'
PROCESS_STATE_FILE = 'data/process_state.json'
SYSTEM_CONFIG_FILE = 'data/system_config.json'


def save_json_file(filepath, data, description="данные", max_retries=3):
    """Универсальная функция сохранения JSON с retry логикой"""
    file_lock = _get_file_lock(filepath)
    
    with file_lock:  # Блокируем файл для этого процесса
        for attempt in range(max_retries):
            try:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
                # Атомарная запись через временный файл
                temp_file = filepath + '.tmp'
                
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                # Заменяем оригинальный файл
                if os.name == 'nt':  # Windows
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    os.rename(temp_file, filepath)
                else:  # Unix/Linux
                    os.rename(temp_file, filepath)
                
                logger.debug(f" {description} сохранены в {filepath}")
                return True
                
            except (OSError, PermissionError) as e:
                if attempt < max_retries - 1:
                    wait_time = 0.1 * (2 ** attempt)  # Экспоненциальная задержка
                    logger.warning(f" Попытка {attempt + 1} неудачна, повторяем через {wait_time}с: {e}")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f" Ошибка сохранения {description} после {max_retries} попыток: {e}")
                    # Удаляем временный файл
                    if 'temp_file' in locals() and os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except:
                            pass
                    return False
            except Exception as e:
                logger.error(f" Неожиданная ошибка сохранения {description}: {e}")
                # Удаляем временный файл
                if 'temp_file' in locals() and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                return False


def load_json_file(filepath, default=None, description="данные"):
    """Универсальная функция загрузки JSON с блокировкой"""
    file_lock = _get_file_lock(filepath)
    
    with file_lock:  # Блокируем файл для чтения
        try:
            if not os.path.exists(filepath):
                logger.info(f" Файл {filepath} не найден")
                return default
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.debug(f" {description} загружены из {filepath}")
            return data
            
        except Exception as e:
            logger.error(f" Ошибка загрузки {description}: {e}")
            return default


# RSI Cache
def save_rsi_cache(coins_data, stats):
    """Сохраняет RSI кэш"""
    cache_data = {
        'timestamp': datetime.now().isoformat(),
        'coins': coins_data,
        'stats': stats
    }
    return save_json_file(RSI_CACHE_FILE, cache_data, "RSI кэш")


def load_rsi_cache():
    """Загружает RSI кэш"""
    cache_data = load_json_file(RSI_CACHE_FILE, description="RSI кэш")
    
    if not cache_data:
        return None
    
    # Проверяем возраст кэша (не старше 6 часов)
    try:
        cache_timestamp = datetime.fromisoformat(cache_data['timestamp'])
        age_hours = (datetime.now() - cache_timestamp).total_seconds() / 3600
        
        if age_hours > 6:
            logger.warning(f" RSI кэш устарел ({age_hours:.1f} часов)")
            return None
        
        logger.info(f" RSI кэш загружен (возраст: {age_hours:.1f}ч)")
        return cache_data
        
    except Exception as e:
        logger.error(f" Ошибка проверки возраста кэша: {e}")
        return None


def clear_rsi_cache():
    """Очищает RSI кэш"""
    try:
        if os.path.exists(RSI_CACHE_FILE):
            os.remove(RSI_CACHE_FILE)
            logger.info(" RSI кэш очищен")
            return True
        return False
    except Exception as e:
        logger.error(f" Ошибка очистки RSI кэша: {e}")
        return False


# Bots State
def save_bots_state(bots_data, auto_bot_config):
    """Сохраняет состояние ботов"""
    state_data = {
        'bots': bots_data,
        'auto_bot_config': auto_bot_config,
        'last_saved': datetime.now().isoformat(),
        'version': '1.0'
    }
    success = save_json_file(BOTS_STATE_FILE, state_data, "состояние ботов")
    if success:
        logger.info(f" Состояние {len(bots_data)} ботов сохранено")
    return success


def load_bots_state():
    """Загружает состояние ботов"""
    return load_json_file(BOTS_STATE_FILE, default={}, description="состояние ботов")


# Auto Bot Config
def save_auto_bot_config(config):
    """Больше не сохраняет конфигурацию автобота в JSON.
    
    Настройки хранятся только в bot_engine/bot_config.py
    """
    logger.debug(" Пропуск сохранения конфигурации автобота (используется bot_config.py)")
    return True


def load_auto_bot_config():
    """Не загружает конфигурацию автобота из JSON.
    
    Настройки читаются напрямую из bot_engine/bot_config.py
    """
    logger.debug(" Пропуск загрузки конфигурации автобота из JSON (используется bot_config.py)")
    return {}


# Individual coin settings
def save_individual_coin_settings(settings):
    """Сохраняет индивидуальные настройки монет"""
    settings_to_save = settings or {}

    # Если настроек нет, не создаем пустой файл и не засоряем логи
    if not settings_to_save:
        if os.path.exists(INDIVIDUAL_COIN_SETTINGS_FILE):
            try:
                os.remove(INDIVIDUAL_COIN_SETTINGS_FILE)
                logger.info(" Индивидуальные настройки монет очищены")
            except OSError as error:
                logger.warning(f" Не удалось удалить файл индивидуальных настроек: {error}")
                return False
        else:
            logger.debug(" Индивидуальных настроек монет нет — файл не создаем")
        return True

    success = save_json_file(
        INDIVIDUAL_COIN_SETTINGS_FILE,
        settings_to_save,
        "индивидуальные настройки монет"
    )
    if success:
        logger.info(f" Индивидуальные настройки монет сохранены ({len(settings_to_save)} записей)")
    return success


def load_individual_coin_settings():
    """Загружает индивидуальные настройки монет"""
    data = load_json_file(
        INDIVIDUAL_COIN_SETTINGS_FILE,
        default={},
        description="индивидуальные настройки монет"
    )
    if not data:
        return {}
    logger.info(f" Загружено индивидуальных настроек монет: {len(data)}")
    return data


# Mature Coins
def save_mature_coins(storage):
    """Сохраняет хранилище зрелых монет"""
    success = save_json_file(MATURE_COINS_FILE, storage, "зрелые монеты")
    if success:
        logger.debug(f" Сохранено {len(storage)} зрелых монет")
    return success


def load_mature_coins():
    """Загружает хранилище зрелых монет"""
    data = load_json_file(MATURE_COINS_FILE, default={}, description="зрелые монеты")
    if data:
        logger.info(f" Загружено {len(data)} зрелых монет")
    return data


# ❌ ОТКЛЮЧЕНО: Optimal EMA удален (EMA фильтр убран из системы)
# def save_optimal_ema(ema_data):
#     """Сохраняет оптимальные EMA периоды"""
#     return True
# 
# def load_optimal_ema():
#     """Загружает оптимальные EMA периоды"""
#     return {}


# Process State
def save_process_state(process_state):
    """Сохраняет состояние процессов"""
    state_data = {
        'process_state': process_state,
        'last_saved': datetime.now().isoformat(),
        'version': '1.0'
    }
    return save_json_file(PROCESS_STATE_FILE, state_data, "состояние процессов")


def load_process_state():
    """Загружает состояние процессов"""
    data = load_json_file(PROCESS_STATE_FILE, description="состояние процессов")
    return data.get('process_state', {}) if data else {}


# System Config
def save_system_config(config):
    """Сохраняет системную конфигурацию в bot_config.py"""
    try:
        from bots_modules.config_writer import save_system_config_to_py
        attrs = {}
        for key, value in config.items():
            attrs[key.upper()] = value
        success = save_system_config_to_py(attrs)
        if success:
            logger.info(" Системная конфигурация сохранена (bot_config.py)")
        return success
    except Exception as e:
        logger.error(f" Ошибка сохранения системной конфигурации: {e}")
        return False


def load_system_config():
    """Перезагружает SystemConfig из bot_config.py"""
    try:
        module = importlib.import_module('bot_engine.bot_config')
        importlib.reload(module)
        return module.SystemConfig
    except Exception as e:
        logger.error(f" Ошибка загрузки системной конфигурации: {e}")
        return None

