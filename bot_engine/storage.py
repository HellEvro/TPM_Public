"""
Управление хранением данных (RSI кэш, состояние ботов, зрелые монеты)
"""

import os
import json
import logging
import time
import threading
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
AUTO_BOT_CONFIG_FILE = 'data/auto_bot_config.json'
MATURE_COINS_FILE = 'data/mature_coins.json'
OPTIMAL_EMA_FILE = 'data/optimal_ema.json'
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
                
                logger.debug(f"[STORAGE] {description} сохранены в {filepath}")
                return True
                
            except (OSError, PermissionError) as e:
                if attempt < max_retries - 1:
                    wait_time = 0.1 * (2 ** attempt)  # Экспоненциальная задержка
                    logger.warning(f"[MATURITY_STORAGE] Попытка {attempt + 1} неудачна, повторяем через {wait_time}с: {e}")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"[STORAGE] Ошибка сохранения {description} после {max_retries} попыток: {e}")
                    # Удаляем временный файл
                    if 'temp_file' in locals() and os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except:
                            pass
                    return False
            except Exception as e:
                logger.error(f"[STORAGE] Неожиданная ошибка сохранения {description}: {e}")
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
                logger.info(f"[STORAGE] Файл {filepath} не найден")
                return default
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.debug(f"[STORAGE] {description} загружены из {filepath}")
            return data
            
        except Exception as e:
            logger.error(f"[STORAGE] Ошибка загрузки {description}: {e}")
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
            logger.warning(f"[STORAGE] RSI кэш устарел ({age_hours:.1f} часов)")
            return None
        
        logger.info(f"[STORAGE] RSI кэш загружен (возраст: {age_hours:.1f}ч)")
        return cache_data
        
    except Exception as e:
        logger.error(f"[STORAGE] Ошибка проверки возраста кэша: {e}")
        return None


def clear_rsi_cache():
    """Очищает RSI кэш"""
    try:
        if os.path.exists(RSI_CACHE_FILE):
            os.remove(RSI_CACHE_FILE)
            logger.info("[STORAGE] RSI кэш очищен")
            return True
        return False
    except Exception as e:
        logger.error(f"[STORAGE] Ошибка очистки RSI кэша: {e}")
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
        logger.info(f"[STORAGE] Состояние {len(bots_data)} ботов сохранено")
    return success


def load_bots_state():
    """Загружает состояние ботов"""
    return load_json_file(BOTS_STATE_FILE, default={}, description="состояние ботов")


# Auto Bot Config
def save_auto_bot_config(config):
    """Сохраняет конфигурацию автобота"""
    success = save_json_file(AUTO_BOT_CONFIG_FILE, config, "конфигурация автобота")
    if success:
        logger.info("[STORAGE] Конфигурация автобота сохранена")
    return success


def load_auto_bot_config():
    """Загружает конфигурацию автобота"""
    return load_json_file(AUTO_BOT_CONFIG_FILE, description="конфигурация автобота")


# Mature Coins
def save_mature_coins(storage):
    """Сохраняет хранилище зрелых монет"""
    success = save_json_file(MATURE_COINS_FILE, storage, "зрелые монеты")
    if success:
        logger.debug(f"[STORAGE] Сохранено {len(storage)} зрелых монет")
    return success


def load_mature_coins():
    """Загружает хранилище зрелых монет"""
    data = load_json_file(MATURE_COINS_FILE, default={}, description="зрелые монеты")
    if data:
        logger.info(f"[STORAGE] Загружено {len(data)} зрелых монет")
    return data


# Optimal EMA
def save_optimal_ema(ema_data):
    """Сохраняет оптимальные EMA периоды"""
    success = save_json_file(OPTIMAL_EMA_FILE, ema_data, "оптимальные EMA")
    if success:
        logger.info(f"[STORAGE] Сохранено {len(ema_data)} EMA периодов")
    return success


def load_optimal_ema():
    """Загружает оптимальные EMA периоды"""
    data = load_json_file(OPTIMAL_EMA_FILE, default={}, description="оптимальные EMA")
    if data:
        logger.info(f"[STORAGE] Загружено {len(data)} EMA периодов")
    return data


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
    """Сохраняет системную конфигурацию"""
    success = save_json_file(SYSTEM_CONFIG_FILE, config, "системная конфигурация")
    if success:
        logger.info("[STORAGE] Системная конфигурация сохранена")
    return success


def load_system_config():
    """Загружает системную конфигурацию"""
    return load_json_file(SYSTEM_CONFIG_FILE, description="системная конфигурация")

