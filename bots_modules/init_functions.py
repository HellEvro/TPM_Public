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
        RealTradingBot, get_individual_coin_settings,
        load_individual_coin_settings
    )
    # ❌ ОТКЛЮЧЕНО: optimal_ema перемещен в backup (используется заглушка из imports_and_globals)
    # # Импорт optimal_ema_data из модуля
    # try:
    #     from bots_modules.optimal_ema import optimal_ema_data
    # except:
    #     optimal_ema_data = {}
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
    def get_individual_coin_settings(symbol):
        return None
    def load_individual_coin_settings():
        return {}

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

from bot_engine.config_loader import SystemConfig

# Импорт функций из других модулей
try:
    from bots_modules.maturity import load_mature_coins_storage
    from bots_modules.imports_and_globals import load_auto_bot_config
    from bots_modules.filters import load_all_coins_rsi, process_trading_signals_for_all_bots, process_auto_bot_signals
    from bots_modules.sync_and_cache import (
        save_default_config, load_system_config,
        load_bots_state, save_bots_state, load_process_state, check_startup_position_conflicts,
        sync_bots_with_exchange, update_process_state, save_auto_bot_config
    )
except ImportError as e:
    print(f"Warning: Could not import functions in init_functions: {e}")
    # Заглушки если импорт не удался
    def load_mature_coins_storage():
        pass
    def save_default_config():
        pass
    def load_system_config():
        pass
    def load_auto_bot_config():
        pass
    def load_bots_state():
        pass
    def save_bots_state():
        pass
    def load_process_state():
        pass
    def check_startup_position_conflicts():
        pass
    def sync_bots_with_exchange():
        pass
    def save_auto_bot_config():
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
        
        # 0. Загружаем постоянное хранилище зрелых монет (с проверкой конфига!)
        load_mature_coins_storage()
        
        # 0.1. 🚀 Загружаем кэш последней проверки зрелости
        from bots_modules.maturity import load_maturity_check_cache
        load_maturity_check_cache()
        
        # 1. Создаем дефолтную конфигурацию если её нет
        save_default_config()
        
        # 2. Загружаем системные настройки
        load_system_config()
        
        # 2.1. Таймфрейм — только из configs/bot_config.py (AutoBotConfig.SYSTEM_TIMEFRAME), не из БД
        try:
            from bot_engine.config_loader import set_current_timeframe, get_current_timeframe
            tf = get_current_timeframe()
            if tf:
                set_current_timeframe(tf)
            logger.info(f"⏱️ Текущий таймфрейм системы (из конфига): {get_current_timeframe()}")
        except Exception as tf_err:
            logger.warning(f"⚠️ Ошибка инициализации таймфрейма из конфига: {tf_err}")

        # 3. Загружаем состояние процессов
        load_process_state()
        
        # 4. Загружаем сохраненное состояние ботов
        load_bots_state()

        # 4.1. Загружаем индивидуальные настройки монет
        load_individual_coin_settings()
        
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
                    
                    default_volume_mode = auto_bot_config.get('default_position_mode', 'usdt')
                    bot_config = {
                        'volume_mode': bot_data.get('volume_mode', default_volume_mode),
                        # ВАЖНО: если в БД volume_value=None, dict.get вернёт None и сломает логику сравнений (None > 0).
                        # Поэтому используем default_position_size как fallback не только при отсутствии ключа, но и при None.
                        'volume_value': (bot_data.get('volume_value') if bot_data.get('volume_value') is not None else auto_bot_config['default_position_size']),
                        'status': bot_data.get('status', 'paused'),
                        'leverage': bot_data.get('leverage', auto_bot_config.get('leverage', 10))  # ✅ Добавляем leverage
                    }
                    bot_config.setdefault('volume_mode', default_volume_mode)
                    
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
                    trading_bot.unrealized_pnl_usdt = bot_data.get('unrealized_pnl_usdt', 0)
                    trading_bot.realized_pnl = bot_data.get('realized_pnl', 0)
                    trading_bot.position_size = bot_data.get('position_size', trading_bot.position_size)
                    trading_bot.position_size_coins = bot_data.get('position_size_coins', trading_bot.position_size_coins)
                    trading_bot.leverage = bot_data.get('leverage', trading_bot.leverage)
                    trading_bot.margin_usdt = bot_data.get('margin_usdt', trading_bot.margin_usdt)
                    trading_bot.max_profit_achieved = bot_data.get('max_profit_achieved', 0)
                    trading_bot.trailing_stop_price = bot_data.get('trailing_stop_price', '')
                    trading_bot.trailing_activation_profit = bot_data.get('trailing_activation_profit', 0)
                    trading_bot.trailing_activation_threshold = bot_data.get('trailing_activation_threshold', 0)
                    trading_bot.trailing_locked_profit = bot_data.get('trailing_locked_profit', 0)
                    trading_bot.trailing_active = bot_data.get('trailing_active', False)
                    trading_bot.trailing_max_profit_usdt = bot_data.get('trailing_max_profit_usdt', 0.0)
                    trading_bot.trailing_step_usdt = bot_data.get('trailing_step_usdt', 0.0)
                    trading_bot.trailing_step_price = bot_data.get('trailing_step_price', 0.0)
                    trading_bot.trailing_steps = bot_data.get('trailing_steps', 0)
                    trading_bot.break_even_activated = bot_data.get('break_even_activated', False)
                    trading_bot.rsi_data = bot_data.get('rsi_data', {})
                    
                    # Обновляем данные в bots_data
                    with bots_data_lock:
                        bots_data['bots'][symbol] = trading_bot.to_dict()
                    
                except Exception as e:
                    logger.error(f" ❌ Ошибка инициализации бота {symbol}: {e}")
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
                                    logger.info(f" ✅ Позиция удалена из реестра при удалении некорректного бота {symbol}: order_id={order_id}")
                                else:
                                    logger.info(f" ℹ️ У некорректного бота {symbol} нет позиции в реестре")
                            except Exception as registry_error:
                                logger.error(f" ❌ Ошибка удаления позиции из реестра для бота {symbol}: {registry_error}")
                                # Не блокируем удаление бота из-за ошибки реестра
                            
                            del bots_data['bots'][symbol]
                logger.info(f" 🗑️ Удалено {len(bots_to_remove)} некорректных ботов")
            
            # 6. ⚠️ Smart RSI Manager ОТКЛЮЧЕН - автобот сам обновляет данные каждые 3 минуты
            # Это избыточно и может привести к конфликтам обновлений
            # global smart_rsi_manager
            # smart_rsi_manager = SmartRSIManager(
            #     rsi_update_callback=load_all_coins_rsi,
            #     trading_signal_callback=process_trading_signals_on_candle_close,
            #     exchange_obj=exchange
            # )
            # smart_rsi_manager.start()
            
            logger.info(" ℹ️ Smart RSI Manager отключен - автобот обновляет данные самостоятельно")
            
            update_process_state('smart_rsi_manager', {
                'active': False,
                'last_update': None,
                'reason': 'Disabled - auto_bot handles updates'
            })
            
            # 7. Синхронизируем с биржей (после инициализации биржи)
            # ⚠️ КРИТИЧНО: Запускаем в отдельном потоке чтобы не блокировать Flask!
            import threading
            def startup_sync():
                try:
                    logger.info(" 🔄 Запуск стартовой синхронизации в фоне...")
                    sync_bots_with_exchange()
                    check_startup_position_conflicts()
                    
                    # ✅ ПРОВЕРКА ДЕЛИСТИНГА ПРИ ЗАПУСКЕ: После загрузки всех монет
                    logger.info(" 🚨 Проверка делистинга при запуске...")
                    try:
                        from bots_modules.sync_and_cache import check_delisting_emergency_close
                        check_delisting_emergency_close()
                        logger.info(" ✅ Проверка делистинга при запуске завершена")
                    except Exception as delisting_error:
                        logger.error(f" ❌ Ошибка проверки делистинга при запуске: {delisting_error}")
                    
                    logger.info(" ✅ Стартовая синхронизация завершена")
                except Exception as e:
                    logger.error(f" ❌ Ошибка стартовой синхронизации: {e}")
            
            sync_thread = threading.Thread(target=startup_sync, daemon=True, name="StartupSync")
            sync_thread.start()
            logger.info(" 🧵 Стартовая синхронизация запущена в фоне")
        else:
            logger.error(" ❌ Не удалось инициализировать биржу")
            update_process_state('exchange_connection', {
                'initialized': False,
                'last_error': 'Initialization failed'
            })
        
        # 8. 🔄 ЗАПУСК НЕПРЕРЫВНОГО ЗАГРУЗЧИКА ДАННЫХ
        # Воркер будет постоянно обновлять все данные по кругу
        # Все остальные модули будут просто читать актуальные данные из хранилища
        logger.info(" 🔄 Запускаем непрерывный загрузчик данных...")
        try:
            from bots_modules.continuous_data_loader import start_continuous_loader
            
            # Запускаем воркер с интервалом 180 сек (3 минуты)
            continuous_loader = start_continuous_loader(exchange_obj=exchange, update_interval=180)
            
            if continuous_loader:
                logger.info(" ✅ Непрерывный загрузчик данных запущен (интервал: 180с)")
                logger.info(" 💡 Все данные будут обновляться автоматически по кругу")
                logger.info(" 💡 Автобот и API будут использовать актуальные данные из хранилища")
            else:
                logger.error(" ❌ Не удалось запустить непрерывный загрузчик")
                
        except Exception as e:
            logger.error(f" ❌ Ошибка запуска непрерывного загрузчика: {e}")
        
        # 9. Воркеры запускаются в main блоке bots.py (после init_bot_service)
        logger.info(" ✅ Инициализация завершена, воркеры будут запущены из main блока")
        
        # КРИТИЧЕСКИ ВАЖНО: Устанавливаем флаг инициализации ПОСЛЕ всех загрузок
        global system_initialized
        system_initialized = True
        
        # Проверяем статус Auto Bot при старте
        with bots_data_lock:
            auto_bot_enabled = bots_data['auto_bot_config']['enabled']
        auto_bot_config = bots_data['auto_bot_config']
        bots_count = len(bots_data['bots'])
            
        # Логируем статус Auto Bot
        if auto_bot_enabled:
            logger.info(" ✅ Автобот включен и готов к работе")
        else:
            logger.info(" ⏹️ Автобот выключен. Включите через UI при необходимости.")
        
        # ✅ ИТОГОВАЯ ИНФОРМАЦИЯ О ЗАПУСКЕ
        logger.info("=" * 80)
        logger.info("✅ СИСТЕМА УСПЕШНО ЗАПУЩЕНА!")
        logger.info("=" * 80)
        logger.info(f"📊 Статус компонентов:")
        logger.info(f"  🔗 Exchange: {'✅ Инициализирован' if exchange else '❌ Не инициализирован'}")
        logger.info(f"  🔄 Continuous Data Loader: ✅ Запущен (обновляет RSI и свечи)")
        logger.info(f"  🤖 Auto Bot: {'❌ ВКЛЮЧЕН!' if auto_bot_enabled else '✅ Выключен (безопасно)'}")
        logger.info(f"  💾 Auto Save: ✅ Запущен")
        logger.info(f"  🔄 Async Processor: ✅ Запущен")
        logger.info("")
        logger.info(f"📈 Данные:")
        logger.info(f"  🤖 Загружено ботов: {bots_count}")
        logger.info(f"  ✅ Зрелых монет: {len(mature_coins_storage)}")
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
                logger.info(f" 🎯 Восстановлено {len(restored_bots)} ботов из реестра позиций")
            else:
                logger.info(" ℹ️ Ботов для восстановления не найдено")
        except Exception as restore_error:
            logger.error(f" ❌ Ошибка восстановления ботов: {restore_error}")
            # Не блокируем запуск системы из-за ошибки восстановления
        
        return True
        
    except Exception as e:
        logger.error(f" ❌ Ошибка инициализации сервиса: {e}")
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
            logger.warning("[ASYNC] 🛑 Остановка асинхронного процессора...")
            async_processor.stop()
            async_processor = None
            async_processor_task = None
            logger.warning("[ASYNC] ✅ Асинхронный процессор остановлен")
        except Exception as e:
            logger.error(f"[ASYNC] ❌ Ошибка остановки асинхронного процессора: {e}")

def create_bot(symbol, config=None, exchange_obj=None):
    """Создает нового бота для символа"""
    with bots_data_lock:
        auto_bot_config = bots_data['auto_bot_config'].copy()

    individual_settings = get_individual_coin_settings(symbol)
    incoming_config = config if isinstance(config, dict) else {}
    
    # ✅ Если передан полный конфиг из сервера (с настройками фильтров), используем его как базу
    # Проверяем, содержит ли incoming_config настройки фильтров (значит это merged конфиг с сервера)
    # Расширяем проверку - если конфиг содержит много настроек (не только volume/leverage), значит это серверный конфиг
    has_server_config = (
        incoming_config and 
        len(incoming_config) > 10 and  # Серверный конфиг содержит много ключей
        any(key in incoming_config for key in ['avoid_up_trend', 'avoid_down_trend', 'rsi_time_filter_enabled', 'rsi_long_threshold', 'rsi_short_threshold'])
    )

    unique_id = f"{symbol}_{int(time.time())}"
    default_volume_mode = auto_bot_config.get('default_position_mode', 'usdt')
    
    # ✅ Если есть серверный конфиг - используем его как базу, иначе собираем из auto_bot_config
    if has_server_config:
        base_config = incoming_config.copy()
        logger.info(f"[BOT_INIT] ✅ Используется серверный merged конфиг для {symbol} (avoid_up_trend={base_config.get('avoid_up_trend')}, avoid_down_trend={base_config.get('avoid_down_trend')})")
        logger.info(f"[BOT_INIT] 🔍 Серверный конфиг содержит {len(base_config)} ключей")
    else:
        logger.warning(f"[BOT_INIT] ⚠️ Серверный конфиг не обнаружен для {symbol}, используется fallback логика (входящий конфиг: {len(incoming_config) if incoming_config else 0} ключей)")
        base_config = {
        'id': unique_id,
        'symbol': symbol,
        'volume_mode': default_volume_mode,
        'volume_value': auto_bot_config.get('default_position_size'),
        'status': BOT_STATUS['RUNNING'],
        'entry_price': None,
        'position_side': None,
        'unrealized_pnl': 0.0,
        'created_at': datetime.now().isoformat(),
        'last_signal_time': None,
        'rsi_long_threshold': auto_bot_config.get('rsi_long_threshold', 29),
        'rsi_short_threshold': auto_bot_config.get('rsi_short_threshold', 71),
        'rsi_exit_long_with_trend': auto_bot_config.get('rsi_exit_long_with_trend', 65),
        'rsi_exit_long_against_trend': auto_bot_config.get('rsi_exit_long_against_trend', 60),
        'rsi_exit_short_with_trend': auto_bot_config.get('rsi_exit_short_with_trend', 35),
        'rsi_exit_short_against_trend': auto_bot_config.get('rsi_exit_short_against_trend', 40),
        'rsi_exit_min_candles': auto_bot_config.get('rsi_exit_min_candles', 0),
        'rsi_exit_min_minutes': auto_bot_config.get('rsi_exit_min_minutes', 0),
        'rsi_exit_min_move_percent': auto_bot_config.get('rsi_exit_min_move_percent', 0),
        'max_loss_percent': auto_bot_config.get('max_loss_percent', 15.0),
        'trailing_stop_activation': auto_bot_config.get('trailing_stop_activation', 20.0),
        'trailing_stop_distance': auto_bot_config.get('trailing_stop_distance', 5.0),
        'trailing_take_distance': auto_bot_config.get('trailing_take_distance', 0.5),
        'trailing_update_interval': auto_bot_config.get('trailing_update_interval', 3.0),
        'max_position_hours': auto_bot_config.get('max_position_hours', 48),
        'break_even_protection': auto_bot_config.get('break_even_protection', True),
        'break_even_trigger': auto_bot_config.get('break_even_trigger', 100.0),
        'leverage': individual_settings.get('leverage') if individual_settings and 'leverage' in individual_settings else auto_bot_config.get('leverage', 10),  # ✅ Из конфиг-файла или индивидуальных настроек
        'break_even_trigger_percent': auto_bot_config.get(
            'break_even_trigger_percent',
            auto_bot_config.get('break_even_trigger', 100.0)
        ),
        'take_profit_percent': auto_bot_config.get('take_profit_percent', 20.0),
        'avoid_down_trend': auto_bot_config.get('avoid_down_trend', False),
        'avoid_up_trend': auto_bot_config.get('avoid_up_trend', False),
        'loss_reentry_protection': auto_bot_config.get('loss_reentry_protection', True),
        'loss_reentry_count': auto_bot_config.get('loss_reentry_count', 1),
        'loss_reentry_candles': auto_bot_config.get('loss_reentry_candles', 3),
        'enable_maturity_check': auto_bot_config.get('enable_maturity_check', True),
        # RSI временной фильтр
        'rsi_time_filter_enabled': auto_bot_config.get('rsi_time_filter_enabled', True),
        'rsi_time_filter_candles': auto_bot_config.get('rsi_time_filter_candles', 8),
        'rsi_time_filter_lower': auto_bot_config.get('rsi_time_filter_lower', 35),
        'rsi_time_filter_upper': auto_bot_config.get('rsi_time_filter_upper', 65),
        # ExitScam фильтр
        'exit_scam_enabled': auto_bot_config.get('exit_scam_enabled', True),
        'exit_scam_candles': auto_bot_config.get('exit_scam_candles', 10),
        'exit_scam_single_candle_percent': auto_bot_config.get('exit_scam_single_candle_percent', 15.0),
        'exit_scam_multi_candle_count': auto_bot_config.get('exit_scam_multi_candle_count', 4),
        'exit_scam_multi_candle_percent': auto_bot_config.get('exit_scam_multi_candle_percent', 50.0),
        # Анализ тренда
        'trend_detection_enabled': auto_bot_config.get('trend_detection_enabled', True),
        'trend_analysis_period': auto_bot_config.get('trend_analysis_period', 30),
        'trend_price_change_threshold': auto_bot_config.get('trend_price_change_threshold', 7),
        'trend_candles_threshold': auto_bot_config.get('trend_candles_threshold', 70),
        # Параметры зрелости
        'min_candles_for_maturity': auto_bot_config.get('min_candles_for_maturity', 400),
        'min_rsi_low': auto_bot_config.get('min_rsi_low', 35),
        'max_rsi_high': auto_bot_config.get('max_rsi_high', 65)
    }

    # ✅ ВАЖНО: Индивидуальные настройки (особенно AI-оптимизированные) применяются ВСЕГДА с приоритетом
    # Это позволяет AI выставлять уникальные улучшенные параметры для каждой монеты
    ai_params_applied = False
    if individual_settings:
        # Определяем, есть ли AI-оптимизированные параметры
        ai_params_keys = ['rsi_long_threshold', 'rsi_short_threshold', 'rsi_exit_long_with_trend', 
                         'rsi_exit_long_against_trend', 'rsi_exit_short_with_trend', 'rsi_exit_short_against_trend']
        has_ai_params = any(key in individual_settings for key in ai_params_keys)
        ai_trained = individual_settings.get('ai_trained', False)
        
        # Применяем индивидуальные настройки с приоритетом (они перезаписывают базовые)
        # Это критично для AI-оптимизированных параметров, которые должны использоваться вместо базовых
        base_config.update(individual_settings)
        
        if has_ai_params or ai_trained:
            ai_params_applied = True
            win_rate = individual_settings.get('ai_win_rate', 0)
            logger.info(
                f"[BOT_INIT] 🤖 Применены AI-оптимизированные параметры для {symbol} "
                f"(Win Rate: {win_rate:.1f}%, Rating: {individual_settings.get('ai_rating', 0):.2f})"
            )
        else:
            pass
    
    # ✅ Обновляем только если НЕ использовали серверный конфиг как базу
    if not has_server_config:
        if incoming_config:
            # Для входящего конфига (без серверных настроек) обновляем только разрешённые поля
            # НО: не перезаписываем AI-параметры из индивидуальных настроек
            allowed_overrides = {'volume_mode', 'volume_value', 'leverage', 'status', 'auto_managed', 'margin_usdt'}
            safe_overrides = {k: v for k, v in incoming_config.items() if k in allowed_overrides}
            if safe_overrides:
                base_config.update(safe_overrides)
    else:
        # Если использовали серверный конфиг - обновляем только разрешённые manual overrides
        # НО: не перезаписываем AI-параметры из индивидуальных настроек
        allowed_manual_overrides = {'volume_mode', 'volume_value', 'leverage', 'status', 'auto_managed', 'margin_usdt'}
        manual_overrides_only = {k: v for k, v in incoming_config.items() if k in allowed_manual_overrides}
        if manual_overrides_only:
            base_config.update(manual_overrides_only)
            logger.info(f"[BOT_INIT] 🔧 Применены manual overrides: {list(manual_overrides_only.keys())}")

    base_config['id'] = unique_id
    base_config['symbol'] = symbol
    base_config['status'] = BOT_STATUS['RUNNING']
    base_config.setdefault('created_at', datetime.now().isoformat())
    base_config.setdefault('volume_mode', default_volume_mode)
    if base_config.get('volume_value') is None:
        base_config['volume_value'] = auto_bot_config.get('default_position_size')

    config = base_config
    
    logger.info(f"[BOT_INIT] Инициализация бота для {symbol}")
    logger.info(f"[BOT_INIT] 🔍 Детальная отладка конфигурации бота:")
    logger.info(f"[BOT_INIT] 🔍 {symbol}: config = {config}")
    logger.info(f"[BOT_INIT] 🔍 {symbol}: volume_mode = {config.get('volume_mode')}")
    logger.info(f"[BOT_INIT] 🔍 {symbol}: volume_value = {config.get('volume_value')}")
    logger.info(f"[BOT_INIT] ⚡ {symbol}: leverage = {config.get('leverage')}x (из конфиг-файла: {auto_bot_config.get('leverage')}x, индивидуальные: {individual_settings.get('leverage') if individual_settings and 'leverage' in individual_settings else None})")
    logger.info(f"[BOT_INIT] Объем торговли: {config.get('volume_mode')} = {config.get('volume_value')}")
    logger.info(f"[BOT_INIT] RSI пороги: Long<={config.get('rsi_long_threshold')}, Short>={config.get('rsi_short_threshold')}")
    logger.info(f"[BOT_INIT] 🛡️ Фильтры трендов: avoid_up_trend={config.get('avoid_up_trend')}, avoid_down_trend={config.get('avoid_down_trend')}")
    
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
        
        # КРИТИЧНО: Логируем BOT_START только после успешного сохранения в bots_data
        # Это гарантирует, что логируются только реальные активные боты
        try:
            from bot_engine.bot_history import log_bot_start
            # Проверяем, что бот действительно сохранен в bots_data
            if symbol in bots_data['bots']:
                # Определяем направление на основе текущей позиции или конфига
                direction = trading_bot.position_side or config.get('position_side') or 'LONG'
                log_bot_start(
                    bot_id=symbol,
                    symbol=symbol,
                    direction=direction,
                    config=config
                )
                logger.info(f"[BOT_HISTORY] ✅ Записано открытие бота {symbol} в историю")
            else:
                pass
        except ImportError as e:
            pass
        except Exception as e:
            logger.error(f"[BOT_HISTORY] ❌ Ошибка логирования запуска бота {symbol}: {e}")
    
    # Автоматически сохраняем состояние после создания бота
    save_bots_state()
    
    return trading_bot.to_dict()

# Старый rsi_update_worker удален - заменен на SmartRSIManager

def process_trading_signals_on_candle_close(candle_timestamp: int, exchange_obj=None):
    """
    Обрабатывает торговые сигналы при закрытии свечи текущего таймфрейма
    
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
            # Проверяем Auto Bot сигналы только если автобот включён (иначе не ищем новые сделки)
            with bots_data_lock:
                auto_bot_enabled = bots_data['auto_bot_config']['enabled']
            if auto_bot_enabled:
                logger.info("[TRADING] 🤖 Проверяем Auto Bot сигналы (нет активных ботов)...")
                process_auto_bot_signals(exchange_obj=exchange_obj)
            return
        
        logger.info(f"[TRADING] 🤖 Обработка сигналов для {len(active_bots)} активных ботов")
        
        # Обрабатываем каждого бота
        for symbol, bot_data in active_bots.items():
            try:
                # ✅ КРИТИЧНО: Закрытие по RSI — по таймфрейму входа бота (entry_timeframe)
                bot_entry_timeframe = bot_data.get('entry_timeframe')
                if bot_entry_timeframe and bot_data.get('status') in [
                    BOT_STATUS.get('IN_POSITION_LONG'),
                    BOT_STATUS.get('IN_POSITION_SHORT')
                ]:
                    timeframe_to_use = bot_entry_timeframe
                else:
                    from bot_engine.config_loader import get_current_timeframe
                    timeframe_to_use = get_current_timeframe()
                
                # Получаем актуальные RSI данные для монеты
                with rsi_data_lock:
                    coin_rsi_data = coins_rsi_data['coins'].get(symbol)
                
                if not coin_rsi_data:
                    logger.warning(f"[TRADING] ⚠️ Нет RSI данных для {symbol}")
                    continue
                
                # Получаем RSI и тренд с учетом таймфрейма бота
                from bot_engine.config_loader import get_rsi_from_coin_data, get_trend_from_coin_data
                rsi = get_rsi_from_coin_data(coin_rsi_data, timeframe=timeframe_to_use)
                trend = get_trend_from_coin_data(coin_rsi_data, timeframe=timeframe_to_use)
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
                    pass
                    
            except Exception as bot_error:
                logger.error(f"[TRADING] ❌ Ошибка обработки бота {symbol}: {bot_error}")
        
        # КРИТИЧЕСКИ ВАЖНО: Обрабатываем Auto Bot сигналы при закрытии свечи только если Auto Bot включен
        with bots_data_lock:
            auto_bot_enabled = bots_data['auto_bot_config']['enabled']
        if auto_bot_enabled:
            logger.info("[TRADING]  Проверяем Auto Bot сигналы после обработки существующих ботов...")
            process_auto_bot_signals(exchange_obj=exchange_obj)  # ВКЛЮЧЕНО!
        
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
        logger.info(" Начало отложенной инициализации биржи...")
        
        # Даем время Flask серверу запуститься
        time.sleep(2)
        
        logger.info(" Подключение к бирже...")
        logger.info(f" Используем ключи: api_key={EXCHANGES['BYBIT']['api_key'][:10]}...")
        
        exchange = ExchangeFactory.create_exchange(
            'BYBIT', 
            EXCHANGES['BYBIT']['api_key'], 
            EXCHANGES['BYBIT']['api_secret']
        )
        
        if not exchange:
            raise Exception("ExchangeFactory вернул None")
        
        logger.info(" ✅ Биржа подключена успешно!")
        
        # Тестируем подключение
        try:
            account_info = exchange.get_unified_account_info()
            logger.info(f" ✅ Тест подключения успешен, баланс: {account_info.get('totalWalletBalance', 'N/A')}")
        except Exception as test_e:
            logger.warning(f" ⚠️ Тест подключения не удался: {str(test_e)}")
        
        # RSI Worker теперь запускается через SmartRSIManager в init_bot_service()
        logger.info(" ✅ Биржа инициализирована")
        
    except Exception as e:
        logger.error(f" ❌ Критическая ошибка инициализации биржи: {str(e)}")
        import traceback
        logger.error(f" Traceback: {traceback.format_exc()}")

def init_exchange_sync():
    """Синхронная инициализация биржи"""
    global exchange
    
    try:
        # Импортируем set_exchange для обновления во всех модулях
        from bots_modules.imports_and_globals import set_exchange
        
        logger.info(" 🔗 Подключение к бирже...")
        
        new_exchange = ExchangeFactory.create_exchange(
            'BYBIT', 
            EXCHANGES['BYBIT']['api_key'], 
            EXCHANGES['BYBIT']['api_secret']
        )
        
        # Устанавливаем биржу ВО ВСЕХ модулях через GlobalState
        exchange = set_exchange(new_exchange)
        
        logger.info(f" 🔍 ExchangeFactory создал биржу: {type(new_exchange)}")
        logger.info(f" 🔍 exchange is None: {new_exchange is None}")
        
        if not new_exchange:
            logger.error(" ❌ ExchangeFactory вернул None")
            return False
        
        # Тестируем подключение
        try:
            account_info = new_exchange.get_unified_account_info()
            logger.info(f" ✅ Подключение успешно, баланс: {account_info.get('totalWalletBalance', 'N/A')}")
        except Exception as test_e:
            logger.warning(f" ⚠️ Тест подключения не удался: {str(test_e)}")
        
        logger.info(f" 🔍 В конце init_exchange_sync exchange: {type(new_exchange)}")
        logger.info(f" 🔍 В конце init_exchange_sync exchange is None: {new_exchange is None}")
        
        return True
        
    except Exception as e:
        logger.error(f" ❌ Критическая ошибка инициализации биржи: {str(e)}")
        import traceback
        logger.error(f" Traceback: {traceback.format_exc()}")
        return False
        
def ensure_exchange_initialized():
    """Проверяет что биржа инициализирована"""
    global exchange
    from bots_modules.imports_and_globals import set_exchange, get_exchange
    
    # Проверяем глобальное состояние
    current_exchange = get_exchange()
    if current_exchange is None:
        logger.warning("[WARNING] Биржа не инициализирована, попытка переподключения...")
        try:
            logger.info(f"[DEBUG] Создание exchange с ключами: api_key={EXCHANGES['BYBIT']['api_key'][:10]}...")
            new_exchange = ExchangeFactory.create_exchange(
                'BYBIT', 
                EXCHANGES['BYBIT']['api_key'], 
                EXCHANGES['BYBIT']['api_secret']
            )
            if new_exchange:
                # ✅ ИСПРАВЛЕНИЕ: Обновляем глобальное состояние
                exchange = set_exchange(new_exchange)
                logger.info("[OK] Биржа переподключена успешно и обновлена в GlobalState")
                return True
            else:
                logger.error(" ExchangeFactory вернул None")
                return False
        except Exception as e:
            logger.error(f" Не удалось переподключиться к бирже: {str(e)}")
            return False
    else:
        # Обновляем локальную переменную
        exchange = current_exchange
        return True

# API endpoints
