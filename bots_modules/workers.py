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

# Константы теперь в SystemConfig

# Импорт функций (будут доступны после импорта)
from bot_engine.bot_config import SystemConfig

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
            # ⚡ БЕЗ БЛОКИРОВКИ: GIL делает чтение атомарным
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
    
    # Проверяем статус Auto Bot
    # ⚡ БЕЗ БЛОКИРОВКИ: GIL делает чтение атомарным
    auto_bot_enabled = bots_data['auto_bot_config']['enabled']
    
    if auto_bot_enabled:
        logger.info("[AUTO_BOT] ✅ Автобот включен и готов к работе")
    else:
        logger.info("[AUTO_BOT] ⏹️ Автобот выключен. Включите через UI при необходимости.")
    
    # Входим в основной цикл - НО проверяем сигналы ТОЛЬКО если автобот включен вручную
    last_position_update = time.time() - SystemConfig.BOT_STATUS_UPDATE_INTERVAL
    last_stop_loss_setup = time.time() - SystemConfig.STOP_LOSS_SETUP_INTERVAL
    last_position_sync = time.time() - SystemConfig.POSITION_SYNC_INTERVAL
    last_inactive_cleanup = time.time() - SystemConfig.INACTIVE_BOT_CLEANUP_INTERVAL
    last_auto_bot_check = time.time()  # Время последней проверки сигналов автобота
    
    logger.info("[AUTO_BOT] 🔄 Входим в основной цикл (автобот выключен, ждем ручного включения)...")
    
    # ✅ КРИТИЧНО: Логируем первый запуск цикла
    cycle_count = 0
    
    while not shutdown_flag.is_set():
        try:
            cycle_count += 1
            logger.info(f"[AUTO_BOT] 🔄 ЦИКЛ #{cycle_count} НАЧАТ")
            
            # Получаем интервал проверки из конфигурации (в секундах)
            # ⚡ БЕЗ БЛОКИРОВКИ: GIL делает чтение атомарным
            check_interval_seconds = bots_data['auto_bot_config']['check_interval']
            auto_bot_enabled = bots_data['auto_bot_config']['enabled']
            
            # ✅ КРИТИЧНО: Обновляем позиции каждую секунду независимо от check_interval!
            # Не ждем check_interval_seconds - это только для проверки сигналов автобота
            logger.info(f"[AUTO_BOT] ⏳ Обновляем позиции каждую секунду...")
            
            # Ждем только 1 секунду для обновления позиций
            if shutdown_flag.wait(1):
                break
            
            # Проверяем сигналы только если Auto Bot включен И прошло достаточно времени
            current_time = time.time()
            time_since_auto_bot_check = current_time - last_auto_bot_check
            
            if auto_bot_enabled and time_since_auto_bot_check >= check_interval_seconds:
                # Подавляем частые сообщения о проверке сигналов
                should_log, log_message = should_log_message(
                    'auto_bot_signals', 
                    f"🔍 Регулярная проверка Auto Bot сигналов (каждые {check_interval_seconds} сек)",
                    interval_seconds=300  # Логируем раз в 5 минут
                )
                if should_log:
                    logger.info(f"[AUTO_BOT] {log_message}")
                
                # 💡 ДАННЫЕ ОБНОВЛЯЮТСЯ НЕПРЕРЫВНЫМ ЗАГРУЗЧИКОМ
                # Автобот просто использует актуальные данные из глобального хранилища
                logger.info(f"[AUTO_BOT] 🚀 Проверяем сигналы на актуальных данных из хранилища...")
                from bots_modules.imports_and_globals import get_exchange
                process_auto_bot_signals(exchange_obj=get_exchange())
                logger.info(f"[AUTO_BOT] ✅ process_auto_bot_signals завершена")
                
                # Обновляем время последней проверки сигналов
                last_auto_bot_check = current_time
                
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
                f"Время с последнего обновления: {time_since_last_update:.1f}с (нужно {SystemConfig.BOT_STATUS_UPDATE_INTERVAL}с)",
                interval_seconds=300  # Логируем раз в 5 минут
            )
            if should_log_time:
                logger.info(f"[POSITION_UPDATE] {log_time_message}")
            
            if time_since_last_update >= SystemConfig.BOT_STATUS_UPDATE_INTERVAL:
                # Подавляем частые сообщения об обновлении кэша
                should_log, log_message = should_log_message(
                    'position_update', 
                    f"🔄 Обновление кэшированных данных ботов (каждые {SystemConfig.BOT_STATUS_UPDATE_INTERVAL} сек)",
                    interval_seconds=300  # Логируем раз в 5 минут
                )
                if should_log:
                    logger.info(f"[BOTS_CACHE] {log_message}")
                
                logger.info(f"[WORKER] 🔄 [1/3] НАЧАЛО: update_bots_cache_data()")
                worker_t_start = time.time()  # time уже импортирован в начале файла
                update_bots_cache_data()
                worker_t_end = time.time()
                execution_time = worker_t_end - worker_t_start
                logger.info(f"[WORKER] ✅ [1/3] КОНЕЦ: update_bots_cache_data() за {execution_time:.1f}с")
                
                # Предупреждение если обновление занимает слишком много времени
                if execution_time > 0.9:  # Если больше 0.9 секунды
                    logger.warning(f"[WORKER] ⚠️ МЕДЛЕННОЕ ОБНОВЛЕНИЕ: {execution_time:.1f}с (может нарушить интервал в 1с)")
                
                last_position_update = current_time
            
            # Устанавливаем недостающие стоп-лоссы каждые SystemConfig.STOP_LOSS_SETUP_INTERVAL секунд
            time_since_stop_setup = current_time - last_stop_loss_setup
            if time_since_stop_setup >= SystemConfig.STOP_LOSS_SETUP_INTERVAL:
                logger.info(f"[WORKER] 🔧 [2/3] НАЧАЛО: check_missing_stop_losses()")
                worker_t_start2 = time.time()
                check_missing_stop_losses()
                logger.info(f"[WORKER] ✅ [2/3] КОНЕЦ: check_missing_stop_losses() за {time.time()-worker_t_start2:.1f}с")
                last_stop_loss_setup = current_time
            
            # Умная синхронизация позиций с биржей каждые SystemConfig.POSITION_SYNC_INTERVAL секунд - ВРЕМЕННО ОТКЛЮЧЕНА
            # time_since_sync = current_time - last_position_sync
            # if time_since_sync >= SystemConfig.POSITION_SYNC_INTERVAL:
            #     logger.info(f"[POSITION_SYNC] 🔄 Синхронизация позиций с биржей (каждые {SystemConfig.POSITION_SYNC_INTERVAL//60} мин)")
            #     sync_positions_with_exchange()
            #     last_position_sync = current_time
            
            # Очищаем неактивные боты каждые SystemConfig.INACTIVE_BOT_CLEANUP_INTERVAL секунд
            time_since_cleanup = current_time - last_inactive_cleanup
            if time_since_cleanup >= SystemConfig.INACTIVE_BOT_CLEANUP_INTERVAL:
                logger.info(f"[WORKER] 🧹 [3a/3] НАЧАЛО: cleanup_inactive_bots()")
                t_start = time.time()
                cleanup_inactive_bots()
                logger.info(f"[WORKER] ✅ [3a/3] КОНЕЦ: cleanup_inactive_bots() за {time.time()-t_start:.1f}с")
                
                # УДАЛЕНО: Очистка зрелых монет - зрелость необратима!
                
                # Активируем правила торговли для зрелых монет
                logger.info(f"[WORKER] 🎯 [3b/3] НАЧАЛО: check_trading_rules_activation()")
                t_start = time.time()
                check_trading_rules_activation()
                logger.info(f"[WORKER] ✅ [3b/3] КОНЕЦ: check_trading_rules_activation() за {time.time()-t_start:.1f}с")
                
                last_inactive_cleanup = current_time
            
        except Exception as e:
            logger.error(f"[AUTO_BOT] ❌ Ошибка Auto Bot Worker: {e}")
            update_process_state('auto_bot_worker', {
                'last_error': str(e),
                'last_check': datetime.now().isoformat()
            })
    
    logger.info("[AUTO_BOT] 🛑 Auto Bot Worker остановлен")


def positions_monitor_worker():
    """
    📊 Мониторинг позиций на бирже (каждую секунду)
    
    Загружает все позиции с биржи и сохраняет в кэш для быстрого доступа.
    Это позволяет определять ручные позиции и избегать конфликтов.
    КРИТИЧНО: Обновляется каждую секунду для быстрой реакции ботов!
    """
    logger.info("[POSITIONS_MONITOR] 🚀 Запуск мониторинга позиций...")
    
    # Создаем глобальный кэш позиций
    global positions_cache
    positions_cache = {
        'positions': [],
        'last_update': None,
        'symbols_with_positions': set()
    }
    
    while not shutdown_flag.is_set():
        try:
            from bots_modules.imports_and_globals import get_exchange
            
            exchange_obj = get_exchange()
            if not exchange_obj:
                logger.warning("[POSITIONS_MONITOR] ⚠️ Exchange не инициализирован")
                time.sleep(5)
                continue
            
            # Загружаем позиции с биржи
            try:
                # Логируем только каждые 30 секунд чтобы не спамить
                should_log = (int(time.time()) % 30 == 0)
                if should_log:
                    logger.info(f"[POSITIONS_MONITOR] 🔄 Загружаем позиции с биржи...")
                
                exchange_positions = exchange_obj.get_positions()
                if isinstance(exchange_positions, tuple):
                    positions_list = exchange_positions[0] if exchange_positions else []
                else:
                    positions_list = exchange_positions if exchange_positions else []
                
                # Обновляем кэш
                symbols_with_positions = set()
                active_positions_log = []
                for pos in positions_list:
                    if abs(float(pos.get('size', 0))) > 0:
                        symbol = pos.get('symbol', '').replace('USDT', '')
                        symbols_with_positions.add(symbol)
                        if should_log:
                            active_positions_log.append(f"{symbol} (размер: {pos.get('size')})")
                
                positions_cache['positions'] = positions_list
                positions_cache['last_update'] = datetime.now().isoformat()
                positions_cache['symbols_with_positions'] = symbols_with_positions
                
                # Логируем только каждые 30 секунд
                if should_log:
                    logger.info(f"[POSITIONS_MONITOR] 📊 Получено {len(positions_list)} позиций с биржи")
                    if active_positions_log:
                        logger.info(f"[POSITIONS_MONITOR] 📈 Активные позиции: {', '.join(active_positions_log)}")
                    logger.info(f"[POSITIONS_MONITOR] ✅ Обновлено: {len(positions_list)} позиций, активных: {len(symbols_with_positions)}")
                
            except Exception as e:
                logger.error(f"[POSITIONS_MONITOR] ❌ Ошибка загрузки позиций: {e}")
                import traceback
                traceback.print_exc()
            
            # Ждем 1 секунду перед следующей проверкой - КАЖДУЮ СЕКУНДУ!
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"[POSITIONS_MONITOR] ❌ Критическая ошибка: {e}")
            time.sleep(10)
    
    logger.info("[POSITIONS_MONITOR] 🛑 Мониторинг позиций остановлен")


# Глобальный кэш позиций
positions_cache = {
    'positions': [],
    'last_update': None,
    'symbols_with_positions': set()
}

