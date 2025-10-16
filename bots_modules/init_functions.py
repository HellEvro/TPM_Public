"""Функции инициализации системы

Включает:
- init_bot_service - инициализация сервиса ботов
- start_async_processor - запуск асинхронного процессора
- stop_async_processor - остановка асинхронного процессора
- create_bot - создание бота
- process_trading_signals_on_candle_close - обработка сигналов
- init_exchange_sync - синхронная инициализация биржи
- ensure_exchange_initialized - проверка инициализации биржи
"""

import os
import time
import logging
import threading
import asyncio
from datetime import datetime

logger = logging.getLogger('BotsService')

# Импортируем глобальные переменные из imports_and_globals
try:
    from bots_modules.imports_and_globals import (
        exchange, smart_rsi_manager, async_processor, async_processor_task,
        system_initialized, shutdown_flag, bots_data_lock, bots_data,
        process_state, mature_coins_storage, ASYNC_AVAILABLE, BOT_STATUS,
        RealTradingBot
    )
    # Импорт optimal_ema_data из модуля
    try:
        from bots_modules.optimal_ema import optimal_ema_data
    except:
        optimal_ema_data = {}
except ImportError as e:
    print(f"Warning: Could not import globals in init_functions: {e}")
    exchange = None
    smart_rsi_manager = None
    async_processor = None
    async_processor_task = None
    system_initialized = False
    shutdown_flag = threading.Event()
    bots_data_lock = threading.Lock()
    bots_data = {}
    process_state = {}
    mature_coins_storage = {}
    ASYNC_AVAILABLE = False
    BOT_STATUS = {}

# Импорт функций
try:
    from exchanges.exchange_factory import ExchangeFactory
    from app.config import EXCHANGES
except:
    pass

try:
    from bot_engine.smart_rsi_manager import SmartRSIManager
except:
    SmartRSIManager = None

try:
    from bot_engine.async_processor import AsyncMainProcessor
except:
    AsyncMainProcessor = None

try:
    from bot_engine.bot_config import SystemConfig
except:
    class SystemConfig:
        ASYNC_PROCESSOR_ENABLED = False

# Импорт функций из других модулей
try:
    from bots_modules.maturity import load_mature_coins_storage
    from bots_modules.optimal_ema import load_optimal_ema_data
    from bots_modules.imports_and_globals import load_auto_bot_config
    from bots_modules.filters import load_all_coins_rsi, process_trading_signals_for_all_bots
    from bots_modules.sync_and_cache import (
        save_default_config, load_system_config,
        load_bots_state, load_process_state, check_startup_position_conflicts,
        sync_bots_with_exchange, update_process_state
    )
except ImportError as e:
    print(f"Warning: Could not import functions in init_functions: {e}")
    # Заглушки если импорт не удался
    def load_mature_coins_storage():
        pass
    def load_optimal_ema_data():
        pass
    def save_default_config():
        pass
    def load_system_config():
        pass
    def load_auto_bot_config():
        pass
    def load_bots_state():
        pass
    def load_process_state():
        pass
    def check_startup_position_conflicts():
        pass
    def sync_bots_with_exchange():
        pass
    def load_all_coins_rsi(exchange_obj=None):
        pass
    def update_process_state(name, data):
        pass

def init_bot_service():
    """Инициализация сервиса ботов с полным восстановлением состояния"""
    try:
        # ✅ Красивый баннер запуска
        logger.info("=" * 80)
        logger.info("🚀 ЗАПУСК СИСТЕМЫ INFOBOT")
        logger.info("=" * 80)
        logger.info(f"📅 Дата: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
        logger.info(f"🔧 Версия: 1.0")
        logger.info("=" * 80)
        
        # 0. Загружаем постоянное хранилище зрелых монет
        load_mature_coins_storage()
        
        # 0.1. Загружаем данные об оптимальных EMA
        load_optimal_ema_data()
        
        # 1. Создаем дефолтную конфигурацию если её нет
        save_default_config()
        
        # 2. Загружаем системные настройки
        load_system_config()
        
        # 3. Загружаем состояние процессов
        load_process_state()
        
        # 4. Загружаем сохраненное состояние ботов
        load_bots_state()
        
        # 5. СНАЧАЛА инициализируем биржу (КРИТИЧЕСКИ ВАЖНО!)
        if init_exchange_sync():
            pass  # Успешно инициализирована
            update_process_state('exchange_connection', {
                'initialized': True,
                'last_sync': datetime.now().isoformat(),
                'connection_count': process_state['exchange_connection']['connection_count'] + 1
            })
            
            # 5.1. Инициализируем загруженных ботов (после инициализации биржи)
            with bots_data_lock:
                # Создаем копию списка ботов для безопасной итерации
                bots_to_init = list(bots_data['bots'].items())
                
            # Инициализируем ботов вне блокировки для избежания deadlock
            bots_to_remove = []
            for symbol, bot_data in bots_to_init:
                try:
                    # Создаем объект бота из сохраненных данных
                    # Получаем настройки из конфига для fallback
                    with bots_data_lock:
                        auto_bot_config = bots_data['auto_bot_config']
                    
                    bot_config = {
                        'volume_mode': bot_data.get('volume_mode', 'usdt'),
                        'volume_value': bot_data.get('volume_value', auto_bot_config['default_position_size']),  # Fallback из конфига для старых ботов
                        'status': bot_data.get('status', 'paused')
                    }
                    
                    trading_bot = RealTradingBot(
                        symbol=bot_data['symbol'],
                        exchange=exchange,
                        config=bot_config
                    )
                    
                    # Восстанавливаем состояние бота
                    trading_bot.status = bot_data.get('status', 'paused')
                    trading_bot.created_at = bot_data.get('created_at', datetime.now().isoformat())
                    trading_bot.entry_price = bot_data.get('entry_price', '')
                    trading_bot.last_price = bot_data.get('last_price', '')
                    trading_bot.last_rsi = bot_data.get('last_rsi', '')
                    trading_bot.last_signal_time = bot_data.get('last_signal_time', '')
                    trading_bot.last_trend = bot_data.get('last_trend', '')
                    trading_bot.position_side = bot_data.get('position_side', '')
                    trading_bot.position_start_time = bot_data.get('position_start_time', '')
                    trading_bot.unrealized_pnl = bot_data.get('unrealized_pnl', 0)
                    trading_bot.max_profit_achieved = bot_data.get('max_profit_achieved', 0)
                    trading_bot.trailing_stop_price = bot_data.get('trailing_stop_price', '')
                    trading_bot.break_even_activated = bot_data.get('break_even_activated', False)
                    trading_bot.rsi_data = bot_data.get('rsi_data', {})
                    
                    # Обновляем данные в bots_data
                    with bots_data_lock:
                        bots_data['bots'][symbol] = trading_bot.to_dict()
                    
                except Exception as e:
                    logger.error(f"[INIT] ❌ Ошибка инициализации бота {symbol}: {e}")
                    # Помечаем бота для удаления
                    bots_to_remove.append(symbol)
            
            # Удаляем некорректных ботов после итерации
            if bots_to_remove:
                with bots_data_lock:
                    for symbol in bots_to_remove:
                        if symbol in bots_data['bots']:
                            bot_data = bots_data['bots'][symbol]
                            
                            # ✅ УДАЛЯЕМ ПОЗИЦИЮ ИЗ РЕЕСТРА ПРИ УДАЛЕНИИ НЕКОРРЕКТНОГО БОТА
                            try:
                                from bots_modules.imports_and_globals import unregister_bot_position
                                position = bot_data.get('position')
                                if position and position.get('order_id'):
                                    order_id = position['order_id']
                                    unregister_bot_position(order_id)
                                    logger.info(f"[INIT] ✅ Позиция удалена из реестра при удалении некорректного бота {symbol}: order_id={order_id}")
                                else:
                                    logger.info(f"[INIT] ℹ️ У некорректного бота {symbol} нет позиции в реестре")
                            except Exception as registry_error:
                                logger.error(f"[INIT] ❌ Ошибка удаления позиции из реестра для бота {symbol}: {registry_error}")
                                # Не блокируем удаление бота из-за ошибки реестра
                            
                            del bots_data['bots'][symbol]
                logger.info(f"[INIT] 🗑️ Удалено {len(bots_to_remove)} некорректных ботов")
            
            # 6. Запускаем Smart RSI Manager (после инициализации биржи)
            global smart_rsi_manager
            smart_rsi_manager = SmartRSIManager(
                rsi_update_callback=load_all_coins_rsi,
                trading_signal_callback=process_trading_signals_on_candle_close,
                exchange_obj=exchange
            )
            smart_rsi_manager.start()
            
            update_process_state('smart_rsi_manager', {
                'active': True,
                'last_update': datetime.now().isoformat()
            })
            
            # 7. Синхронизируем с биржей (после инициализации биржи)
            sync_bots_with_exchange()
            
            # 7.1. КРИТИЧЕСКИ ВАЖНО: Проверяем конфликты позиций при запуске
            check_startup_position_conflicts()
        else:
            logger.error("[INIT] ❌ Не удалось инициализировать биржу")
            update_process_state('exchange_connection', {
                'initialized': False,
                'last_error': 'Initialization failed'
            })
        
        # 8. Воркеры запускаются в main блоке bots.py (после init_bot_service)
        logger.info("[INIT] ✅ Инициализация завершена, воркеры будут запущены из main блока")
        
        # Запускаем асинхронный процессор для улучшения производительности
        if start_async_processor():
            pass  # Успешно запущен
        else:
            logger.warning("[INIT] ⚠️ Асинхронный процессор не запущен, работаем в синхронном режиме")
        
        # КРИТИЧЕСКИ ВАЖНО: Устанавливаем флаг инициализации ПОСЛЕ всех загрузок
        global system_initialized
        system_initialized = True
        
        # КРИТИЧЕСКИ ВАЖНО: Проверяем Auto Bot при старте - он ДОЛЖЕН быть выключен!
        with bots_data_lock:
            auto_bot_enabled = bots_data['auto_bot_config']['enabled']
        auto_bot_config = bots_data['auto_bot_config']
        bots_count = len(bots_data['bots'])
            
        # ПРИНУДИТЕЛЬНО выключаем автобот при старте системы для безопасности!
        if auto_bot_enabled:
            logger.warning("[INIT] ⚠️ Автобот включен при старте! Принудительно выключаем для безопасности...")
            bots_data['auto_bot_config']['enabled'] = False
            auto_bot_enabled = False
            save_auto_bot_config()  # Сохраняем изменение
        
        # ✅ ИТОГОВАЯ ИНФОРМАЦИЯ О ЗАПУСКЕ
        logger.info("=" * 80)
        logger.info("✅ СИСТЕМА УСПЕШНО ЗАПУЩЕНА!")
        logger.info("=" * 80)
        logger.info(f"📊 Статус компонентов:")
        logger.info(f"  🔗 Exchange: {'✅ Инициализирован' if exchange else '❌ Не инициализирован'}")
        logger.info(f"  📊 Smart RSI Manager: {'✅ Запущен' if smart_rsi_manager else '❌ Не запущен'}")
        logger.info(f"  🤖 Auto Bot: {'❌ ВКЛЮЧЕН!' if auto_bot_enabled else '✅ Выключен (безопасно)'}")
        logger.info(f"  💾 Auto Save: ✅ Запущен")
        logger.info(f"  🔄 Async Processor: ✅ Запущен")
        logger.info("")
        logger.info(f"📈 Данные:")
        logger.info(f"  🤖 Загружено ботов: {bots_count}")
        logger.info(f"  ✅ Зрелых монет: {len(mature_coins_storage)}")
        logger.info(f"  📊 Optimal EMA: {len(optimal_ema_data)}")
        logger.info("")
        logger.info(f"⚙️ Конфигурация Auto Bot:")
        logger.info(f"  📊 RSI: LONG≤{auto_bot_config.get('rsi_long_threshold')}, SHORT≥{auto_bot_config.get('rsi_short_threshold')}")
        logger.info(f"  ⏰ RSI Time Filter: {'✅ ON' if auto_bot_config.get('rsi_time_filter_enabled') else '❌ OFF'} ({auto_bot_config.get('rsi_time_filter_candles')} свечей)")
        logger.info(f"  ✅ Maturity Check: {'✅ ON' if auto_bot_config.get('enable_maturity_check') else '❌ OFF'}")
        logger.info(f"  🛡️ Stop-Loss: {auto_bot_config.get('max_loss_percent')}%, Trailing: {auto_bot_config.get('trailing_stop_activation')}%")
        logger.info(f"  👥 Max Concurrent: {auto_bot_config.get('max_concurrent')}")
        logger.info("=" * 80)
        logger.info("🎯 СИСТЕМА ГОТОВА К РАБОТЕ!")
        logger.info("💡 Логи будут показывать только важные события")
        logger.info("=" * 80)
        
        # ✅ ВОССТАНАВЛИВАЕМ ПОТЕРЯННЫХ БОТОВ ИЗ РЕЕСТРА
        try:
            from bots_modules.imports_and_globals import restore_lost_bots
            restored_bots = restore_lost_bots()
            if restored_bots:
                logger.info(f"[INIT] 🎯 Восстановлено {len(restored_bots)} ботов из реестра позиций")
            else:
                logger.info("[INIT] ℹ️ Ботов для восстановления не найдено")
        except Exception as restore_error:
            logger.error(f"[INIT] ❌ Ошибка восстановления ботов: {restore_error}")
            # Не блокируем запуск системы из-за ошибки восстановления
        
        return True
        
    except Exception as e:
        logger.error(f"[INIT] ❌ Ошибка инициализации сервиса: {e}")
        return False

def start_async_processor():
    """Запускает асинхронный процессор"""
    global async_processor, async_processor_task
    
    if not ASYNC_AVAILABLE:
        logger.warning("[ASYNC] ⚠️ Асинхронный процессор недоступен")
        return False
    
    try:
        logger.info("[ASYNC] 🚀 Запуск асинхронного процессора...")
        
        # Конфигурация для асинхронного процессора
        async_config = {
            'max_rsi_requests': 15,  # Увеличиваем количество одновременных запросов
            'max_concurrent_bots': 8,  # Увеличиваем количество ботов
            'max_concurrent_signals': 20,  # Увеличиваем количество сигналов
            'max_concurrent_saves': 5,  # Увеличиваем количество сохранений
            'rsi_update_interval': SystemConfig.RSI_UPDATE_INTERVAL,
            'position_sync_interval': 60,  # Синхронизация позиций каждую минуту
            'bot_processing_interval': 10,  # Обработка ботов каждые 10 секунд
            'signal_processing_interval': 5,  # Обработка сигналов каждые 5 секунд
            'data_saving_interval': 30  # Сохранение данных каждые 30 секунд
        }
        
        # Создаем асинхронный процессор
        # Используем глобальную переменную exchange
        global exchange
        logger.info(f"[ASYNC] 🔍 Проверяем глобальную переменную exchange: {type(exchange)}")
        logger.info(f"[ASYNC] 🔍 exchange is None: {exchange is None}")
        logger.info(f"[ASYNC] 🔍 exchange == None: {exchange == None}")
        
        if exchange is None:
            logger.error("[ASYNC] ❌ Биржа не инициализирована, пропускаем асинхронный процессор")
            return False
        
        logger.info(f"[ASYNC] ✅ Биржа найдена, создаем AsyncMainProcessor с типом: {type(exchange)}")
        async_processor = AsyncMainProcessor(exchange, async_config)
        logger.info(f"[ASYNC] ✅ AsyncMainProcessor создан успешно")
        
        # Запускаем в отдельном потоке
        def run_async_processor():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(async_processor.start())
            except Exception as e:
                logger.error(f"[ASYNC] ❌ Ошибка в асинхронном процессоре: {e}")
            finally:
                loop.close()
        
        async_processor_task = threading.Thread(target=run_async_processor, daemon=True)
        async_processor_task.start()
        
        # Немедленная синхронизация позиций при запуске - ВРЕМЕННО ОТКЛЮЧЕНА
        # logger.info("[ASYNC] 🔄 Немедленная синхронизация позиций при запуске...")
        # try:
        #     result = sync_positions_with_exchange()
        #     logger.info(f"[ASYNC] ✅ Синхронизация завершена: {result}")
        # except Exception as e:
        #     logger.error(f"[ASYNC] ❌ Ошибка синхронизации: {e}")
        
        logger.info("[ASYNC] ✅ Асинхронный процессор запущен")
        return True
        
    except Exception as e:
        logger.error(f"[ASYNC] ❌ Ошибка запуска асинхронного процессора: {e}")
        return False

def stop_async_processor():
    """Останавливает асинхронный процессор"""
    global async_processor, async_processor_task
    
    if async_processor:
        try:
            logger.info("[ASYNC] 🛑 Остановка асинхронного процессора...")
            async_processor.stop()
            async_processor = None
            async_processor_task = None
            logger.info("[ASYNC] ✅ Асинхронный процессор остановлен")
        except Exception as e:
            logger.error(f"[ASYNC] ❌ Ошибка остановки асинхронного процессора: {e}")

def create_bot(symbol, config=None, exchange_obj=None):
    """Создает нового бота для символа"""
    if config is None:
        # Получаем default_position_size из конфигурации Auto Bot
        with bots_data_lock:
            auto_bot_config = bots_data['auto_bot_config']
            default_volume = auto_bot_config['default_position_size']
        
        config = {
            'volume_mode': 'usdt',
            'volume_value': default_volume,
            'status': BOT_STATUS['RUNNING'],
            'entry_price': None,
            'position_side': None,
            'unrealized_pnl': 0.0,
            'created_at': datetime.now().isoformat(),
            'last_signal_time': None
        }
    
    # Применяем настройки из конфигурации Auto Bot как базовые
    with bots_data_lock:
        auto_bot_config = bots_data['auto_bot_config']
        base_config = {
            'volume_mode': 'usdt',
            'volume_value': auto_bot_config['default_position_size'],
            'status': BOT_STATUS['RUNNING'],
            'entry_price': None,
            'position_side': None,
            'unrealized_pnl': 0.0,
            'created_at': datetime.now().isoformat(),
            'last_signal_time': None,
            # Настройки RSI и защитных механизмов
            'rsi_long_threshold': auto_bot_config.get('rsi_long_threshold', 29),
            'rsi_short_threshold': auto_bot_config.get('rsi_short_threshold', 71),
            'rsi_exit_long': auto_bot_config.get('rsi_exit_long', 65),
            'rsi_exit_short': auto_bot_config.get('rsi_exit_short', 35),
            'max_loss_percent': auto_bot_config.get('max_loss_percent', 15.0),
            'trailing_stop_activation': auto_bot_config.get('trailing_stop_activation', 300.0),
            'trailing_stop_distance': auto_bot_config.get('trailing_stop_distance', 150.0),
            'max_position_hours': auto_bot_config.get('max_position_hours', 48),
            'break_even_protection': auto_bot_config.get('break_even_protection', True),
            'break_even_trigger': auto_bot_config.get('break_even_trigger', 100.0),
            'avoid_down_trend': auto_bot_config.get('avoid_down_trend', True),
            'avoid_up_trend': auto_bot_config.get('avoid_up_trend', True),
            'enable_maturity_check': auto_bot_config.get('enable_maturity_check', True)
        }
        
        # Объединяем базовую конфигурацию с переданной (переданная имеет приоритет)
        full_config = {**base_config, **config}
        config = full_config
    
    logger.info(f"[BOT_INIT] Инициализация бота для {symbol}")
    logger.info(f"[BOT_INIT] 🔍 Детальная отладка конфигурации бота:")
    logger.info(f"[BOT_INIT] 🔍 {symbol}: config = {config}")
    logger.info(f"[BOT_INIT] 🔍 {symbol}: volume_mode = {config.get('volume_mode')}")
    logger.info(f"[BOT_INIT] 🔍 {symbol}: volume_value = {config.get('volume_value')}")
    logger.info(f"[BOT_INIT] Объем торговли: {config.get('volume_mode')} = {config.get('volume_value')}")
    logger.info(f"[BOT_INIT] RSI пороги: Long<={config.get('rsi_long_threshold')}, Short>={config.get('rsi_short_threshold')}")
    
    # Создаем экземпляр торгового бота
    logger.info(f"[BOT_INIT] Создание экземпляра TradingBot для {symbol}...")
    # Используем переданную биржу или глобальную переменную
    exchange_to_use = exchange_obj if exchange_obj else exchange
    trading_bot = RealTradingBot(symbol, exchange_to_use, config)
    
    with bots_data_lock:
        # Обновляем существующую запись или создаем новую
        if symbol in bots_data['bots']:
            # Если есть временная запись с статусом 'creating', обновляем её
            if bots_data['bots'][symbol].get('status') == 'creating':
                logger.info(f"[BOT_ACTIVE] 🔄 Обновляем временную запись бота {symbol}")
            else:
                logger.info(f"[BOT_ACTIVE] ⚠️ Бот {symbol} уже существует, перезаписываем")
        
        bots_data['bots'][symbol] = trading_bot.to_dict()
        total_bots = len(bots_data['bots'])
        logger.info(f"[BOT_ACTIVE] ✅ Бот {symbol} добавлен в список активных")
        logger.info(f"[BOT_ACTIVE] Всего активных ботов: {total_bots}")
        logger.info(f"[BOT_ACTIVE] Статус {symbol}: {trading_bot.status}")
    
    # Логируем создание бота в историю
    # log_bot_start(symbol, config)  # TODO: Функция не определена
    
    # Автоматически сохраняем состояние после создания бота
    save_bots_state()
    
    return trading_bot.to_dict()

# Старый rsi_update_worker удален - заменен на SmartRSIManager

def process_trading_signals_on_candle_close(candle_timestamp: int, exchange_obj=None):
    """
    Обрабатывает торговые сигналы при закрытии свечи 6H
    
    Args:
        candle_timestamp: Timestamp закрытой свечи
        exchange_obj: Объект биржи (если None, используется глобальная переменная)
    """
    try:
        logger.info(f"[TRADING] 🎯 Обработка торговых сигналов для свечи {candle_timestamp}")
        
        # КРИТИЧЕСКИ ВАЖНО: Обрабатываем торговые сигналы для всех ботов в основном процессе
        logger.info("[TRADING] 🔄 Вызываем process_trading_signals_for_all_bots...")
        process_trading_signals_for_all_bots(exchange_obj=exchange_obj)
        logger.info("[TRADING] ✅ process_trading_signals_for_all_bots завершен")
        
        # Получаем список активных ботов
        with bots_data_lock:
            active_bots = {symbol: bot for symbol, bot in bots_data['bots'].items() 
                          if bot['status'] not in [BOT_STATUS['IDLE'], BOT_STATUS['PAUSED']]}
        
        if not active_bots:
            logger.info("[TRADING] 📭 Нет активных ботов для обработки сигналов")
            # Но все равно проверяем Auto Bot сигналы!
            logger.info("[TRADING] 🤖 Проверяем Auto Bot сигналы (нет активных ботов)...")
            # process_auto_bot_signals(exchange_obj=exchange_obj)  # ОТКЛЮЧЕНО!
            return
        
        logger.info(f"[TRADING] 🤖 Обработка сигналов для {len(active_bots)} активных ботов")
        
        # Обрабатываем каждого бота
        for symbol, bot_data in active_bots.items():
            try:
                # Получаем актуальные RSI данные для монеты
                with rsi_data_lock:
                    coin_rsi_data = coins_rsi_data['coins'].get(symbol)
                
                if not coin_rsi_data:
                    logger.warning(f"[TRADING] ⚠️ Нет RSI данных для {symbol}")
                    continue
                
                rsi = coin_rsi_data.get('rsi6h')
                trend = coin_rsi_data.get('trend6h', 'NEUTRAL')
                price = coin_rsi_data.get('price', 0)
                
                if not rsi or not price:
                    logger.warning(f"[TRADING] ⚠️ Неполные данные для {symbol}: RSI={rsi}, Price={price}")
                    continue
                
                logger.info(f"[TRADING] 📊 {symbol}: RSI={rsi}, Trend={trend}, Price={price}")
                
                # Создаем объект торгового бота для обработки сигнала
                # Используем переданную биржу или глобальную переменную
                exchange_to_use = exchange_obj if exchange_obj else exchange
                trading_bot = RealTradingBot(symbol, exchange_to_use, bot_data)
                
                # Обрабатываем торговый сигнал при закрытии свечи
                result = trading_bot.process_trading_signals(trend, rsi, price, on_candle_close=True)
                
                if result:
                    logger.info(f"[TRADING] ✅ {symbol}: Обработан сигнал при закрытии свечи")
                    
                    # Обновляем данные бота
                    with bots_data_lock:
                        bots_data['bots'][symbol] = trading_bot.to_dict()
                else:
                    logger.debug(f"[TRADING] 💤 {symbol}: Нет активных сигналов")
                    
            except Exception as bot_error:
                logger.error(f"[TRADING] ❌ Ошибка обработки бота {symbol}: {bot_error}")
        
        # КРИТИЧЕСКИ ВАЖНО: Обрабатываем Auto Bot сигналы при закрытии свечи только если Auto Bot включен
        with bots_data_lock:
            auto_bot_enabled = bots_data['auto_bot_config']['enabled']
        if auto_bot_enabled:
            logger.info("[TRADING]  Проверяем Auto Bot сигналы после обработки существующих ботов...")
            # process_auto_bot_signals(exchange_obj=exchange_obj)  # ОТКЛЮЧЕНО!
        
        # Сохраняем состояние после обработки сигналов
        save_bots_state()
        logger.info(f"[TRADING] 💾 Состояние ботов сохранено после обработки сигналов")
        
    except Exception as e:
        logger.error(f"[TRADING] ❌ Критическая ошибка обработки торговых сигналов: {e}")

# Эта функция удалена - используется основная init_bot_service() выше

def delayed_exchange_init():
    """Отложенная инициализация биржи"""
    global exchange
    
    try:
        logger.info("[INIT] Начало отложенной инициализации биржи...")
        
        # Даем время Flask серверу запуститься
        time.sleep(2)
        
        logger.info("[INIT] Подключение к бирже...")
        logger.info(f"[INIT] Используем ключи: api_key={EXCHANGES['BYBIT']['api_key'][:10]}...")
        
        exchange = ExchangeFactory.create_exchange(
            'BYBIT', 
            EXCHANGES['BYBIT']['api_key'], 
            EXCHANGES['BYBIT']['api_secret']
        )
        
        if not exchange:
            raise Exception("ExchangeFactory вернул None")
        
        logger.info("[INIT] ✅ Биржа подключена успешно!")
        
        # Тестируем подключение
        try:
            account_info = exchange.get_unified_account_info()
            logger.info(f"[INIT] ✅ Тест подключения успешен, баланс: {account_info.get('totalWalletBalance', 'N/A')}")
        except Exception as test_e:
            logger.warning(f"[INIT] ⚠️ Тест подключения не удался: {str(test_e)}")
        
        # RSI Worker теперь запускается через SmartRSIManager в init_bot_service()
        logger.info("[INIT] ✅ Биржа инициализирована")
        
    except Exception as e:
        logger.error(f"[INIT] ❌ Критическая ошибка инициализации биржи: {str(e)}")
        import traceback
        logger.error(f"[INIT] Traceback: {traceback.format_exc()}")

def init_exchange_sync():
    """Синхронная инициализация биржи"""
    global exchange
    
    try:
        # Импортируем set_exchange для обновления во всех модулях
        from bots_modules.imports_and_globals import set_exchange
        
        logger.info("[SYNC] 🔗 Подключение к бирже...")
        
        new_exchange = ExchangeFactory.create_exchange(
            'BYBIT', 
            EXCHANGES['BYBIT']['api_key'], 
            EXCHANGES['BYBIT']['api_secret']
        )
        
        # Устанавливаем биржу ВО ВСЕХ модулях через GlobalState
        exchange = set_exchange(new_exchange)
        
        logger.info(f"[SYNC] 🔍 ExchangeFactory создал биржу: {type(new_exchange)}")
        logger.info(f"[SYNC] 🔍 exchange is None: {new_exchange is None}")
        
        if not new_exchange:
            logger.error("[SYNC] ❌ ExchangeFactory вернул None")
            return False
        
        # Тестируем подключение
        try:
            account_info = new_exchange.get_unified_account_info()
            logger.info(f"[SYNC] ✅ Подключение успешно, баланс: {account_info.get('totalWalletBalance', 'N/A')}")
        except Exception as test_e:
            logger.warning(f"[SYNC] ⚠️ Тест подключения не удался: {str(test_e)}")
        
        logger.info(f"[SYNC] 🔍 В конце init_exchange_sync exchange: {type(new_exchange)}")
        logger.info(f"[SYNC] 🔍 В конце init_exchange_sync exchange is None: {new_exchange is None}")
        
        return True
        
    except Exception as e:
        logger.error(f"[SYNC] ❌ Критическая ошибка инициализации биржи: {str(e)}")
        import traceback
        logger.error(f"[SYNC] Traceback: {traceback.format_exc()}")
        return False
        
def ensure_exchange_initialized():
    """Проверяет что биржа инициализирована"""
    global exchange
    if exchange is None:
        logger.warning("[WARNING] Биржа не инициализирована, попытка переподключения...")
        try:
            logger.info(f"[DEBUG] Создание exchange с ключами: api_key={EXCHANGES['BYBIT']['api_key'][:10]}...")
            exchange = ExchangeFactory.create_exchange(
                'BYBIT', 
                EXCHANGES['BYBIT']['api_key'], 
                EXCHANGES['BYBIT']['api_secret']
            )
            if exchange:
                logger.info("[OK] Биржа переподключена успешно")
                return True
            else:
                logger.error("[ERROR] ExchangeFactory вернул None")
                return False
        except Exception as e:
            logger.error(f"[ERROR] Не удалось переподключиться к бирже: {str(e)}")
            return False
    logger.debug("[DEBUG] Exchange уже инициализирован")
    return True

# API endpoints
