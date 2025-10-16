"""Фоновые воркеры

Включает:
- auto_save_worker - автоматическое сохранение состояния
- auto_bot_worker - проверка сигналов Auto Bot
"""

import time
import logging
import threading
from datetime import datetime

logger = logging.getLogger('BotsService')

# Импортируем глобальные переменные из imports_and_globals
try:
    from bots_modules.imports_and_globals import (
        shutdown_flag, system_initialized, bots_data_lock, bots_data,
        process_state, mature_coins_storage, mature_coins_lock, exchange
    )
except ImportError as e:
    print(f"Warning: Could not import globals in workers: {e}")
    shutdown_flag = threading.Event()
    system_initialized = False
    bots_data_lock = threading.Lock()
    bots_data = {}
    process_state = {}
    mature_coins_storage = {}
    mature_coins_lock = threading.Lock()
    exchange = None

# Константы
BOT_STATUS_UPDATE_INTERVAL = 30
STOP_LOSS_SETUP_INTERVAL = 300
POSITION_SYNC_INTERVAL = 30
INACTIVE_BOT_CLEANUP_INTERVAL = 600

# Импорт функций (будут доступны после импорта)
try:
    from bot_engine.bot_config import SystemConfig
except:
    class SystemConfig:
        AUTO_SAVE_INTERVAL = 60

# Импорт функций из других модулей
try:
    from bots_modules.imports_and_globals import should_log_message
    from bots_modules.sync_and_cache import (
        save_bots_state, update_process_state, save_auto_bot_config,
        update_bots_cache_data, check_missing_stop_losses,
        cleanup_inactive_bots, check_trading_rules_activation
    )
    from bots_modules.maturity import save_mature_coins_storage
    from bots_modules.filters import process_auto_bot_signals
except ImportError as e:
    print(f"Warning: Could not import functions in workers: {e}")
    def should_log_message(category, message, interval_seconds=60):
        return (True, message)
    def save_bots_state():
        return True
    def save_mature_coins_storage():
        pass
    def update_process_state(name, data):
        pass
    def save_auto_bot_config():
        pass
    def update_bots_cache_data():
        pass
    def check_missing_stop_losses():
        pass
    def cleanup_inactive_bots():
        pass
    def check_trading_rules_activation():
        pass
    def process_auto_bot_signals(exchange_obj=None):
        pass

def auto_save_worker():
    """Воркер для автоматического сохранения состояния согласно конфигурации"""
    interval = SystemConfig.AUTO_SAVE_INTERVAL
    logger.info(f"[AUTO_SAVE] 💾 Запуск Auto Save Worker (сохранение каждые {interval} секунд)")
    
    while not shutdown_flag.is_set():
        try:
            # Ждем согласно конфигурации
            if shutdown_flag.wait(interval):
                break
            
            # Сохраняем состояние
            with bots_data_lock:
                bots_count = len(bots_data['bots'])
            
            if bots_count > 0:
                # Логируем только при первом сохранении или если прошло 5 минут
                should_log = (getattr(auto_save_worker, '_last_log_time', 0) + 300 < time.time())
                if should_log:
                    logger.info(f"[AUTO_SAVE] 💾 Автосохранение состояния {bots_count} ботов...")
                    auto_save_worker._last_log_time = time.time()
                save_result = save_bots_state()
                
                # Сохраняем хранилище зрелых монет
                save_mature_coins_storage()
                
                # Обновляем статистику
                update_process_state('auto_save_worker', {
                    'last_save': datetime.now().isoformat(),
                    'save_count': process_state['auto_save_worker']['save_count'] + 1,
                    'last_error': None if save_result else 'Save failed'
                })
            
        except Exception as e:
            logger.error(f"[AUTO_SAVE] ❌ Ошибка автосохранения: {e}")
    
    logger.info("[AUTO_SAVE] 💾 Auto Save Worker остановлен")

def auto_bot_worker():
    """Воркер для регулярной проверки Auto Bot сигналов"""
    logger.info("[AUTO_BOT] 🚫 Auto Bot Worker запущен в режиме ожидания")
    logger.info("[AUTO_BOT] 💡 Автобот НЕ запускается автоматически!")
    logger.info("[AUTO_BOT] 💡 Включите его ВРУЧНУЮ через UI когда будете готовы")
    
    # КРИТИЧЕСКИ ВАЖНО: Принудительно выключаем автобот при запуске!
    with bots_data_lock:
        auto_bot_enabled = bots_data['auto_bot_config']['enabled']
    
    if auto_bot_enabled:
        logger.warning("[AUTO_BOT] ⚠️ Автобот был включен! Принудительно выключаем для безопасности...")
        with bots_data_lock:
            bots_data['auto_bot_config']['enabled'] = False
            save_auto_bot_config()
        logger.warning("[AUTO_BOT] 🔒 Автобот выключен. Включите вручную через UI.")
    
    logger.info("[AUTO_BOT] ✅ Автобот готов к ручному запуску через UI")
    
    # Входим в основной цикл - НО проверяем сигналы ТОЛЬКО если автобот включен вручную
    last_position_update = time.time() - BOT_STATUS_UPDATE_INTERVAL
    last_stop_loss_setup = time.time() - STOP_LOSS_SETUP_INTERVAL
    last_position_sync = time.time() - POSITION_SYNC_INTERVAL
    last_inactive_cleanup = time.time() - INACTIVE_BOT_CLEANUP_INTERVAL
    
    logger.info("[AUTO_BOT] 🔄 Входим в основной цикл (автобот выключен, ждем ручного включения)...")
    while not shutdown_flag.is_set():
        try:
            # Получаем интервал проверки из конфигурации (в секундах)
            with bots_data_lock:
                check_interval_seconds = bots_data['auto_bot_config']['check_interval']
                auto_bot_enabled = bots_data['auto_bot_config']['enabled']
            
            # Ждем согласно конфигурации
            if shutdown_flag.wait(check_interval_seconds):
                break
            
            # Проверяем сигналы только если Auto Bot включен
            if auto_bot_enabled:
                # Подавляем частые сообщения о проверке сигналов
                should_log, log_message = should_log_message(
                    'auto_bot_signals', 
                    f"🔍 Регулярная проверка Auto Bot сигналов (каждые {check_interval_seconds} сек)",
                    interval_seconds=300  # Логируем раз в 5 минут
                )
                if should_log:
                    logger.info(f"[AUTO_BOT] {log_message}")
                
                logger.info(f"[AUTO_BOT] 🚀 Вызываем process_auto_bot_signals...")
                process_auto_bot_signals(exchange_obj=exchange)
                logger.info(f"[AUTO_BOT] ✅ process_auto_bot_signals завершена")
                
                # Обновляем статистику
                current_count = process_state.get('auto_bot_worker', {}).get('check_count', 0)
                update_process_state('auto_bot_worker', {
                    'last_check': datetime.now().isoformat(),
                    'check_count': current_count + 1,
                    'interval_seconds': check_interval_seconds,
                    'enabled': True
                })
            else:
                logger.info(f"[AUTO_BOT] ⏹️ Auto Bot выключен, пропускаем проверку (следующая через {check_interval_seconds} сек)")
                update_process_state('auto_bot_worker', {
                    'last_check': datetime.now().isoformat(),
                    'enabled': False,
                    'interval_seconds': check_interval_seconds
                })
            
            # Обновляем статус позиций каждые BOT_STATUS_UPDATE_INTERVAL секунд (независимо от Auto Bot)
            current_time = time.time()
            time_since_last_update = current_time - last_position_update
            # Подавляем частые сообщения о времени обновления
            should_log_time, log_time_message = should_log_message(
                'position_update_time', 
                f"Время с последнего обновления: {time_since_last_update:.1f}с (нужно {BOT_STATUS_UPDATE_INTERVAL}с)",
                interval_seconds=300  # Логируем раз в 5 минут
            )
            if should_log_time:
                logger.info(f"[POSITION_UPDATE] {log_time_message}")
            
            if time_since_last_update >= BOT_STATUS_UPDATE_INTERVAL:
                # Подавляем частые сообщения об обновлении кэша
                should_log, log_message = should_log_message(
                    'position_update', 
                    f"🔄 Обновление кэшированных данных ботов (каждые {BOT_STATUS_UPDATE_INTERVAL} сек)",
                    interval_seconds=300  # Логируем раз в 5 минут
                )
                if should_log:
                    logger.info(f"[BOTS_CACHE] {log_message}")
                
                update_bots_cache_data()
                last_position_update = current_time
            
            # Устанавливаем недостающие стоп-лоссы каждые STOP_LOSS_SETUP_INTERVAL секунд
            time_since_stop_setup = current_time - last_stop_loss_setup
            if time_since_stop_setup >= STOP_LOSS_SETUP_INTERVAL:
                logger.info(f"[STOP_LOSS_SETUP] 🔧 Установка недостающих стоп-лоссов (каждые {STOP_LOSS_SETUP_INTERVAL//60} мин)")
                check_missing_stop_losses()
                last_stop_loss_setup = current_time
            
            # Умная синхронизация позиций с биржей каждые POSITION_SYNC_INTERVAL секунд - ВРЕМЕННО ОТКЛЮЧЕНА
            # time_since_sync = current_time - last_position_sync
            # if time_since_sync >= POSITION_SYNC_INTERVAL:
            #     logger.info(f"[POSITION_SYNC] 🔄 Синхронизация позиций с биржей (каждые {POSITION_SYNC_INTERVAL//60} мин)")
            #     sync_positions_with_exchange()
            #     last_position_sync = current_time
            
            # Очищаем неактивные боты каждые INACTIVE_BOT_CLEANUP_INTERVAL секунд
            time_since_cleanup = current_time - last_inactive_cleanup
            if time_since_cleanup >= INACTIVE_BOT_CLEANUP_INTERVAL:
                logger.info(f"[INACTIVE_CLEANUP] 🧹 Очистка неактивных ботов (каждые {INACTIVE_BOT_CLEANUP_INTERVAL//60} мин)")
                cleanup_inactive_bots()
                
                # УДАЛЕНО: Очистка зрелых монет - зрелость необратима!
                
                # Активируем правила торговли для зрелых монет
                check_trading_rules_activation()
                
                last_inactive_cleanup = current_time
            
        except Exception as e:
            logger.error(f"[AUTO_BOT] ❌ Ошибка Auto Bot Worker: {e}")
            update_process_state('auto_bot_worker', {
                'last_error': str(e),
                'last_check': datetime.now().isoformat()
            })
    
    logger.info("[AUTO_BOT] 🛑 Auto Bot Worker остановлен")

