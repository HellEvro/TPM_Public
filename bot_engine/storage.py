"""
Управление хранением данных (RSI кэш, состояние ботов, зрелые монеты)

✅ МИГРАЦИЯ В БД: Все данные теперь хранятся ТОЛЬКО в базе данных (data/bots_data.db)
JSON файлы больше не используются - только БД!
"""

import os
import json
import logging
import time
import threading
import importlib
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger('Storage')

# Инициализация БД (ленивая загрузка)
_bots_db = None
_bots_db_lock = threading.Lock()

def _get_bots_database():
    """Получает экземпляр базы данных Bots (ленивая инициализация)"""
    global _bots_db

    with _bots_db_lock:
        if _bots_db is None:
            try:
                from bot_engine.bots_database import get_bots_database
                _bots_db = get_bots_database()
            except Exception as e:
                logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Не удалось инициализировать Bots Database: {e}")
                logger.error("❌ БД обязательна для работы! Проверьте конфигурацию.")
                raise  # Поднимаем исключение - БД обязательна!

        return _bots_db

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
SYSTEM_CONFIG_FILE = 'configs/system_config.json'

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
            return data

        except Exception as e:
            logger.error(f" Ошибка загрузки {description}: {e}")
            return default

# RSI Cache
def save_rsi_cache(coins_data, stats):
    """Сохраняет RSI кэш в БД"""
    db = _get_bots_database()

    try:
        if db.save_rsi_cache(coins_data, stats):
            return True
        return False
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения RSI кэша в БД: {e}")
        raise  # Поднимаем исключение - БД обязательна!

def load_rsi_cache():
    """Загружает RSI кэш из БД"""
    db = _get_bots_database()

    try:
        cache_data = db.load_rsi_cache(max_age_hours=6.0)
        return cache_data
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки RSI кэша из БД: {e}")
        raise  # Поднимаем исключение - БД обязательна!

def clear_rsi_cache():
    """Очищает RSI кэш в БД"""
    db = _get_bots_database()

    try:
        if db.clear_rsi_cache():
            logger.info("✅ RSI кэш очищен в БД")
            return True
        return False
    except Exception as e:
        logger.error(f"❌ Ошибка очистки RSI кэша в БД: {e}")
        raise  # Поднимаем исключение - БД обязательна!

# Bots State
def save_bots_state(bots_data, auto_bot_config):
    """Сохраняет состояние ботов в БД"""
    db = _get_bots_database()

    try:
        if db.save_bots_state(bots_data, auto_bot_config):
            # ✅ ИСПРАВЛЕНО: Логируем только если есть боты или раз в 5 минут, чтобы не спамить
            bots_count = len(bots_data) if isinstance(bots_data, dict) else 0
            if bots_count > 0:
                # Логируем только раз в 5 минут для уменьшения спама
                import time
                last_log_time = getattr(save_bots_state, '_last_log_time', 0)
                if time.time() - last_log_time > 300:  # 5 минут
                    logger.info(f"💾 Состояние {bots_count} ботов сохранено в БД")
                    save_bots_state._last_log_time = time.time()
            # Не логируем когда ботов 0 - это спам
            return True
        return False
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения состояния ботов в БД: {e}")
        raise  # Поднимаем исключение - БД обязательна!

def load_bots_state():
    """Загружает состояние ботов из БД"""
    db = _get_bots_database()

    try:
        state_data = db.load_bots_state()
        return state_data if state_data else {}
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки состояния ботов из БД: {e}")
        raise  # Поднимаем исключение - БД обязательна!

# Auto Bot Config
def save_auto_bot_config(config):
    """Больше не сохраняет конфигурацию автобота в JSON.

    Настройки хранятся только в configs/bot_config.py
    """
    return True

def load_auto_bot_config():
    """Не загружает конфигурацию автобота из JSON.

    Настройки читаются напрямую из configs/bot_config.py
    """
    return {}

# Individual coin settings
def save_individual_coin_settings(settings):
    """Сохраняет индивидуальные настройки монет в БД"""
    settings_to_save = settings or {}

    db = _get_bots_database()

    try:
        if not settings_to_save:
            # Очищаем настройки в БД
            if db.remove_all_individual_coin_settings():
                logger.info("✅ Индивидуальные настройки монет очищены в БД")
                return True
            return False
        else:
            if db.save_individual_coin_settings(settings_to_save):
                logger.info(f"💾 Индивидуальные настройки монет сохранены в БД ({len(settings_to_save)} записей)")
                return True
            return False
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения индивидуальных настроек в БД: {e}")
        raise  # Поднимаем исключение - БД обязательна!

def load_individual_coin_settings():
    """Загружает индивидуальные настройки монет из БД"""
    db = _get_bots_database()

    try:
        settings = db.load_individual_coin_settings()
        if settings:
            logger.info(f"✅ Загружено индивидуальных настроек монет из БД: {len(settings)}")
        return settings if settings else {}
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки индивидуальных настроек из БД: {e}")
        raise  # Поднимаем исключение - БД обязательна!

# Mature Coins
def save_mature_coins(storage):
    """Сохраняет хранилище зрелых монет в БД"""
    db = _get_bots_database()

    try:
        if db.save_mature_coins(storage):
            return True
        return False
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения зрелых монет в БД: {e}")
        raise  # Поднимаем исключение - БД обязательна!

def load_mature_coins():
    """Загружает хранилище зрелых монет из БД"""
    db = _get_bots_database()

    try:
        data = db.load_mature_coins()
        if data:
            logger.info(f"✅ Загружено {len(data)} зрелых монет из БД")
        return data if data else {}
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки зрелых монет из БД: {e}")
        raise  # Поднимаем исключение - БД обязательна!

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
    """Сохраняет состояние процессов в БД"""
    db = _get_bots_database()

    try:
        if db.save_process_state(process_state):
            # Убрано избыточное DEBUG логирование для уменьшения спама
            # 
            return True
        return False
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения состояния процессов в БД: {e}")
        raise  # Поднимаем исключение - БД обязательна!

def load_process_state():
    """Загружает состояние процессов из БД"""
    db = _get_bots_database()

    try:
        process_state_data = db.load_process_state()
        return process_state_data if process_state_data else {}
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки состояния процессов из БД: {e}")
        raise  # Поднимаем исключение - БД обязательна!

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
        from bot_engine.config_loader import reload_config
        module = reload_config()
        return module.SystemConfig
    except Exception as e:
        logger.error(f" Ошибка загрузки системной конфигурации: {e}")
        return None

# Bot Positions Registry
def save_bot_positions_registry(registry):
    """Сохраняет реестр позиций ботов в БД"""
    db = _get_bots_database()

    try:
        if db.save_bot_positions_registry(registry):
            return True
        return False
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения реестра позиций в БД: {e}")
        raise  # Поднимаем исключение - БД обязательна!

def load_bot_positions_registry():
    """Загружает реестр позиций ботов из БД"""
    db = _get_bots_database()

    try:
        registry = db.load_bot_positions_registry()
        return registry if registry else {}
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки реестра позиций из БД: {e}")
        raise  # Поднимаем исключение - БД обязательна!

# Maturity Check Cache
def save_maturity_check_cache(coins_count: int, config_hash: str = None) -> bool:
    """Сохраняет кэш проверки зрелости в БД"""
    db = _get_bots_database()

    try:
        if db.save_maturity_check_cache(coins_count, config_hash):
            return True
        return False
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения кэша проверки зрелости в БД: {e}")
        raise  # Поднимаем исключение - БД обязательна!

def load_maturity_check_cache() -> dict:
    """Загружает кэш проверки зрелости из БД"""
    db = _get_bots_database()

    try:
        cache_data = db.load_maturity_check_cache()
        return cache_data if cache_data else {'coins_count': 0, 'config_hash': None}
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки кэша проверки зрелости из БД: {e}")
        raise  # Поднимаем исключение - БД обязательна!

# Delisted Coins
def save_delisted_coins(delisted: list) -> bool:
    """Сохраняет делистированные монеты в БД"""
    db = _get_bots_database()

    try:
        if db.save_delisted_coins(delisted):
            return True
        return False
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения делистированных монет в БД: {e}")
        raise  # Поднимаем исключение - БД обязательна!

def load_delisted_coins() -> list:
    """Загружает делистированные монеты из БД"""
    db = _get_bots_database()

    try:
        delisted = db.load_delisted_coins()
        return delisted if delisted else []
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки делистированных монет из БД: {e}")
        raise  # Поднимаем исключение - БД обязательна!

def is_coin_delisted(symbol: str) -> bool:
    """Проверяет, делистирована ли монета (из БД)"""
    db = _get_bots_database()

    try:
        return db.is_coin_delisted(symbol)
    except Exception as e:
        logger.error(f"❌ Ошибка проверки делистирования в БД: {e}")
        raise  # Поднимаем исключение - БД обязательна!

# Candles Cache
def save_candles_cache(candles_cache: Dict) -> bool:
    """Сохраняет кэш свечей в БД"""
    # ⚠️ КРИТИЧНО: Проверяем, что это НЕ процесс ai.py
    # ai.py должен использовать ai_database.save_candles(), а не bots_data.db!
    import os
    import sys
    import traceback
    script_name = os.path.basename(sys.argv[0]).lower() if sys.argv else ''
    main_file = None
    try:
        if hasattr(sys.modules.get('__main__', None), '__file__') and sys.modules['__main__'].__file__:
            main_file = str(sys.modules['__main__'].__file__).lower()
    except:
        pass

    # Сначала проверяем, что это НЕ bots.py
    is_bots_process = (
        'bots.py' in script_name or 
        any('bots.py' in str(arg).lower() for arg in sys.argv) or
        (main_file and 'bots.py' in main_file)
    )

    # Если это точно bots.py - разрешаем запись
    if is_bots_process:
        pass  # Разрешаем запись
    else:
        # Проверяем, что это ai.py
        is_ai_process = (
            'ai.py' in script_name or 
            any('ai.py' in str(arg).lower() for arg in sys.argv) or
            (main_file and 'ai.py' in main_file) or
            os.environ.get('INFOBOT_AI_PROCESS', '').lower() == 'true'
        )

        if is_ai_process:
            # Получаем стек вызовов для диагностики
            stack = ''.join(traceback.format_stack()[-8:-1])
            logger.error("=" * 80)
            logger.error("🚫 КРИТИЧЕСКАЯ БЛОКИРОВКА: ai.py пытается записать в bots_data.db через save_candles_cache()!")
            logger.error(f"🚫 script_name={script_name}")
            logger.error(f"🚫 main_file={main_file}")
            logger.error(f"🚫 env INFOBOT_AI_PROCESS={os.environ.get('INFOBOT_AI_PROCESS', 'НЕ УСТАНОВЛЕНО')}")
            logger.error(f"🚫 sys.argv={sys.argv}")
            logger.error(f"🚫 Стек вызовов:\n{stack}")
            logger.error("🚫 Используйте ai_database.save_candles() вместо этого!")
            logger.error("=" * 80)
            return False

    db = _get_bots_database()

    try:
        if db.save_candles_cache(candles_cache):
            return True
        return False
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения кэша свечей в БД: {e}")
        raise  # Поднимаем исключение - БД обязательна!

def load_candles_cache(symbol: Optional[str] = None) -> Dict:
    """Загружает кэш свечей из БД"""
    db = _get_bots_database()

    try:
        cache = db.load_candles_cache(symbol=symbol)
        return cache if cache else {}
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки кэша свечей из БД: {e}")
        raise  # Поднимаем исключение - БД обязательна!

def get_candles_for_symbol(symbol: str) -> Optional[Dict]:
    """Получает свечи для конкретного символа из БД"""
    db = _get_bots_database()

    try:
        return db.get_candles_for_symbol(symbol)
    except Exception as e:
        logger.error(f"❌ Ошибка получения свечей для {symbol} из БД: {e}")
        raise  # Поднимаем исключение - БД обязательна!
