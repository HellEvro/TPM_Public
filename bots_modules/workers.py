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
        process_state, mature_coins_storage, mature_coins_lock, get_exchange
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
    def get_exchange():
        return None

# Константы теперь в SystemConfig

# Импорт функций (будут доступны после импорта)
from bot_engine.config_loader import SystemConfig

try:
    from utils.memory_utils import force_collect_full
except ImportError:
    def force_collect_full():
        pass

# Импорт функций из других модулей
try:
    from bots_modules.imports_and_globals import should_log_message
    from bots_modules.sync_and_cache import (
        save_bots_state, update_process_state, save_auto_bot_config,
        update_bots_cache_data, check_missing_stop_losses,
        cleanup_inactive_bots, check_trading_rules_activation,
        check_delisting_emergency_close, sync_positions_with_exchange
    )
    from bots_modules.maturity import save_mature_coins_storage
    from bots_modules.filters import process_auto_bot_signals, process_trading_signals_for_all_bots
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
    def process_trading_signals_for_all_bots(exchange_obj=None):
        pass
    def sync_positions_with_exchange():
        pass

def log_system_status(cycle_count, auto_bot_enabled, check_interval_seconds):
    """Логирует компактный статус системы с ключевой информацией"""
    try:
        from bots_modules.imports_and_globals import mature_coins_storage, bots_data_lock, service_start_time

        with bots_data_lock:
            # Подсчитываем ботов
            total_bots = len(bots_data.get('bots', {}))
            active_bots = sum(1 for bot in bots_data['bots'].values() 
                            if bot.get('status') not in ['paused', 'idle'])
            in_position = sum(1 for bot in bots_data['bots'].values() 
                            if bot.get('status') in ['in_position_long', 'in_position_short'])

            # Зрелые монеты
            mature_count = len(mature_coins_storage)

            # AI Status
            try:
                from bot_engine.ai.risk_manager import DynamicRiskManager
                ai_status = "✅ AI доступен"
            except:
                ai_status = "❌ AI недоступен"

            # Exchange: актуальное состояние через get_exchange(); в первые 30 с — «подключение», не «не подключена»
            exch = get_exchange()
            if exch:
                exchange_status = "✅ Подключена"
            elif (time.time() - service_start_time) < 30:
                exchange_status = "⏳ Подключение..."
            else:
                exchange_status = "❌ Не подключена"

            # Компактный статус
            logger.info("=" * 80)
            logger.info("📊 СТАТУС СИСТЕМЫ")
            logger.info("=" * 80)
            logger.info(f"🤖 Боты: {total_bots} всего | {active_bots} активных | {in_position} в позиции")
            logger.info(f"💰 Зрелые монеты: {mature_count}")
            logger.info(f"{'🎯' if auto_bot_enabled else '⏹️'}  AutoBot: {'ON' if auto_bot_enabled else 'OFF'} (интервал: {check_interval_seconds}s)")
            logger.info(f"💡 AI: {ai_status}")
            logger.info(f"🌐 Биржа: {exchange_status}")
            logger.info("=" * 80)

    except Exception as e:
                pass

def auto_save_worker():
    """Воркер для автоматического сохранения состояния согласно конфигурации"""
    interval = SystemConfig.AUTO_SAVE_INTERVAL
    logger.info(f" 💾 Запуск Auto Save Worker (сохранение каждые {interval} секунд)")

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
                    logger.info(f" 💾 Автосохранение состояния {bots_count} ботов...")
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

                force_collect_full()

        except Exception as e:
            logger.error(f" ❌ Ошибка автосохранения: {e}")

    logger.warning(" 💾 Auto Save Worker остановлен")

def auto_bot_worker():
    """Воркер для регулярной проверки Auto Bot сигналов"""
    logger.info(" 🚫 Auto Bot Worker запущен в режиме ожидания")
    logger.info(" 💡 Автобот НЕ запускается автоматически!")
    logger.info(" 💡 Включите его ВРУЧНУЮ через UI когда будете готовы")

    # Проверяем статус Auto Bot
    # ⚡ БЕЗ БЛОКИРОВКИ: GIL делает чтение атомарным
    auto_bot_enabled = bots_data['auto_bot_config']['enabled']

    if auto_bot_enabled:
        logger.info(" ✅ Автобот включен и готов к работе")
    else:
        logger.info(" ⏹️ Автобот выключен. Включите через UI при необходимости.")

    # Входим в основной цикл - НО проверяем сигналы ТОЛЬКО если автобот включен вручную
    last_position_update = time.time() - SystemConfig.BOT_STATUS_UPDATE_INTERVAL
    last_stop_loss_setup = time.time() - SystemConfig.STOP_LOSS_SETUP_INTERVAL
    last_position_sync = time.time() - SystemConfig.POSITION_SYNC_INTERVAL
    last_inactive_cleanup = time.time() - SystemConfig.INACTIVE_BOT_CLEANUP_INTERVAL
    last_auto_bot_check = time.time()  # Время последней проверки сигналов автобота
    last_trading_signals_check = time.time()  # Время последней проверки торговых сигналов для всех ботов
    last_delisting_check = time.time() - 600  # Время последней проверки делистинга (10 минут назад для первого запуска)

    logger.info(" 🔄 Входим в основной цикл (автобот выключен, ждем ручного включения)...")

    # ✅ КРИТИЧНО: Логируем первый запуск цикла
    cycle_count = 0

    while not shutdown_flag.is_set():
        try:
            cycle_count += 1

            # Получаем интервал проверки из конфигурации (в секундах)
            # ⚡ БЕЗ БЛОКИРОВКИ: GIL делает чтение атомарным
            check_interval_seconds = bots_data['auto_bot_config']['check_interval']
            auto_bot_enabled = bots_data['auto_bot_config']['enabled']

            # Логируем статус раз в 5 минут с важной информацией
            if cycle_count % 300 == 1:
                log_system_status(cycle_count, auto_bot_enabled, check_interval_seconds)

            # Сборка мусора раз в ~60 сек (цикл ~1 сек)
            if cycle_count % 60 == 0:
                force_collect_full()

            # Ждем только 1 секунду для обновления позиций
            if shutdown_flag.wait(1):
                break

            # Проверяем сигналы только если Auto Bot включен И прошло достаточно времени
            current_time = time.time()
            time_since_auto_bot_check = current_time - last_auto_bot_check

            if auto_bot_enabled and time_since_auto_bot_check >= check_interval_seconds:
                from bots_modules.imports_and_globals import get_exchange, coins_rsi_data
                # ✅ Блокировка только до первой загрузки: после first_round_complete ожидание не используется
                if not coins_rsi_data.get('first_round_complete'):
                    last_auto_bot_check = current_time
                    if cycle_count % 30 == 0:  # раз в ~30 сек
                        logger.info(" ⏳ Ожидание первой загрузки свечей и расчёта RSI — автобот запустится после этого...")
                    continue
                process_auto_bot_signals(exchange_obj=get_exchange())

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
                update_process_state('auto_bot_worker', {
                    'last_check': datetime.now().isoformat(),
                    'enabled': False,
                    'interval_seconds': check_interval_seconds
                })

            # Примечание: Проверка закрытия по RSI и решения по стопам — в positions_monitor_worker и sync_positions_with_exchange()
            # по интервалу «Синхронизация позиций» (POSITION_SYNC_INTERVAL): раз в N сек — свечи, RSI, закрыть/стопы

            # Обновляем статус позиций каждые BOT_STATUS_UPDATE_INTERVAL секунд (независимо от Auto Bot)
            current_time = time.time()
            time_since_last_update = current_time - last_position_update

            if time_since_last_update >= SystemConfig.BOT_STATUS_UPDATE_INTERVAL:
                # Логируем только при медленном обновлении (проблема!)
                worker_t_start = time.time()
                update_bots_cache_data()
                execution_time = time.time() - worker_t_start

                last_position_update = current_time

            # Устанавливаем недостающие стоп-лоссы каждые SystemConfig.STOP_LOSS_SETUP_INTERVAL секунд
            time_since_stop_setup = current_time - last_stop_loss_setup
            if time_since_stop_setup >= SystemConfig.STOP_LOSS_SETUP_INTERVAL:
                check_missing_stop_losses()
                last_stop_loss_setup = current_time

            # Очищаем неактивные боты каждые SystemConfig.INACTIVE_BOT_CLEANUP_INTERVAL секунд
            time_since_cleanup = current_time - last_inactive_cleanup
            if time_since_cleanup >= SystemConfig.INACTIVE_BOT_CLEANUP_INTERVAL:
                cleanup_inactive_bots()
                check_trading_rules_activation()
                last_inactive_cleanup = current_time

            # ✅ Синхронизация позиций с биржей каждые POSITION_SYNC_INTERVAL секунд (настройка «Синхронизация позиций»)
            time_since_position_sync = current_time - last_position_sync
            if time_since_position_sync >= SystemConfig.POSITION_SYNC_INTERVAL:
                try:
                    sync_positions_with_exchange()
                except Exception as sync_err:
                    logger.debug(f" Синхронизация позиций: {sync_err}")
                last_position_sync = current_time

            # ✅ ПРОВЕРКА ДЕЛИСТИНГА: Каждые 10 минут проверяем делистинг и закрываем позиции
            current_time = time.time()
            time_since_delisting_check = current_time - last_delisting_check

            if time_since_delisting_check >= 600:  # 10 минут = 600 секунд
                check_delisting_emergency_close()
                last_delisting_check = current_time

        except Exception as e:
            logger.error(f" ❌ Ошибка Auto Bot Worker: {e}")
            update_process_state('auto_bot_worker', {
                'last_error': str(e),
                'last_check': datetime.now().isoformat()
            })

    logger.warning(" 🛑 Auto Bot Worker остановлен")

def positions_monitor_worker():
    """
    📊 Мониторинг позиций на бирже и проверка закрытия по RSI

    Загружает все позиции с биржи и сохраняет в кэш для быстрого доступа.
    Интервал расчёта RSI и решений (закрыть/стопы) = POSITION_SYNC_INTERVAL («Синхронизация позиций»).
    Каждые N сек: для ботов в позиции — 20+ свечей → RSI → решение закрыть или нет; стопы/трейлинг — в sync_positions_with_exchange().
    """
    logger.info(" 🚀 Запуск мониторинга позиций...")

    # Создаем глобальный кэш позиций
    global positions_cache
    positions_cache = {
        'positions': [],
        'last_update': None,
        'symbols_with_positions': set()
    }

    # ✅ КРИТИЧНО: Флаг первого запуска - ждем первую загрузку RSI
    first_startup = True
    rsi_data_loaded_once = False

    # Время последней проверки закрытия позиций по RSI (интервал = «Синхронизация позиций», чтобы раз в 1–2 сек пересчитывать RSI и решения)
    last_rsi_close_check = time.time() - SystemConfig.POSITION_SYNC_INTERVAL  # Сразу при первом запуске

    # Время начала ожидания инициализации биржи
    exchange_init_wait_start = time.time()
    exchange_init_warning_shown = False
    _gc_ticks = 0

    while not shutdown_flag.is_set():
        try:
            from bots_modules.imports_and_globals import get_exchange

            exchange_obj = get_exchange()
            if not exchange_obj:
                # Показываем предупреждение только если:
                # 1. Система еще не инициализирована (normal wait) - не показываем
                # 2. Система уже инициализирована, но биржа не инициализирована (error) - показываем
                # 3. Прошло больше 30 секунд ожидания (timeout) - показываем
                wait_time = time.time() - exchange_init_wait_start

                if system_initialized:
                    # Система инициализирована, но биржа не инициализирована - это проблема
                    if not exchange_init_warning_shown:
                        logger.warning(" ⚠️ Exchange не инициализирован (система уже запущена)")
                        exchange_init_warning_shown = True
                elif wait_time > 30:
                    # Прошло больше 30 секунд - показываем предупреждение о задержке
                    if not exchange_init_warning_shown:
                        logger.warning(f" ⚠️ Exchange все еще не инициализирован (ожидание: {int(wait_time)}с)")
                        exchange_init_warning_shown = True

                time.sleep(5)
                continue

            # Биржа инициализирована - сбрасываем флаги
            if exchange_init_warning_shown:
                logger.info(" ✅ Exchange инициализирован, мониторинг позиций возобновлен")
                exchange_init_warning_shown = False
            exchange_init_wait_start = time.time()  # Сбрасываем таймер

            # Загружаем позиции с биржи
            try:
                # Логируем только каждые 30 секунд чтобы не спамить
                should_log = (int(time.time()) % 30 == 0)
                if should_log:
                    logger.info(f" 🔄 Загружаем позиции с биржи...")

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
                    logger.info(f" 📊 Получено {len(positions_list)} позиций с биржи")
                    if active_positions_log:
                        logger.info(f" 📈 Активные позиции: {', '.join(active_positions_log)}")
                    logger.info(f" ✅ Обновлено: {len(positions_list)} позиций, активных: {len(symbols_with_positions)}")

            except Exception as e:
                logger.error(f" ❌ Ошибка загрузки позиций: {e}")
                import traceback
                traceback.print_exc()

            # ✅ Интервал = «Синхронизация позиций» (POSITION_SYNC_INTERVAL): раз в 1–2 сек — свечи, RSI, решение закрыть/нет; стопы — в sync_positions_with_exchange()
            current_time = time.time()
            time_since_rsi_check = current_time - last_rsi_close_check
            position_sync_interval = SystemConfig.POSITION_SYNC_INTERVAL

            if time_since_rsi_check >= position_sync_interval:
                try:
                    # ✅ Перечитываем конфиг с диска — пороги RSI выхода из UI учитываются при закрытии
                    try:
                        from bots_modules.imports_and_globals import load_auto_bot_config
                        if hasattr(load_auto_bot_config, '_last_mtime'):
                            load_auto_bot_config._last_mtime = 0
                        load_auto_bot_config()
                    except Exception:
                        pass
                    # ✅ КРИТИЧНО: Проверяем, загружены ли RSI данные перед проверкой
                    from bots_modules.imports_and_globals import bots_data, bots_data_lock, coins_rsi_data
                    from bots_modules.bot_class import NewTradingBot

                    # ✅ Блокировка только до первой загрузки: проверки по RSI — только после first_round_complete; далее не ждём
                    rsi_data_available = (
                        coins_rsi_data.get('first_round_complete') and
                        coins_rsi_data.get('coins') is not None and
                        len(coins_rsi_data.get('coins', {})) > 0
                    )

                    # При первом запуске ждём загрузки RSI; после первой загрузки блокировка не используется
                    if first_startup:
                        if rsi_data_available:
                            first_startup = False
                            rsi_data_loaded_once = True
                        else:
                            last_rsi_close_check = current_time
                            continue
                    else:
                        if not rsi_data_available:
                            last_rsi_close_check = current_time
                            continue

                    # ✅ RSI данные загружены - выполняем проверку закрытия
                    with bots_data_lock:
                        # Получаем только ботов в позиции
                        bots_in_position = {
                            symbol: bot_data for symbol, bot_data in bots_data.get('bots', {}).items()
                            if bot_data.get('status') in ['in_position_long', 'in_position_short']
                        }

                    if bots_in_position:
                        for symbol, bot_data in bots_in_position.items():
                            try:
                                position_side = bot_data.get('position_side')

                                # ✅ КРИТИЧНО: Закрытие по RSI — по таймфрейму ВХОДА бота (entry_timeframe). 1m-бот закрывается по 1m RSI.
                                bot_entry_timeframe = bot_data.get('entry_timeframe')
                                if not bot_entry_timeframe:
                                    from bot_engine.config_loader import get_current_timeframe
                                    bot_entry_timeframe = get_current_timeframe()

                                rsi_data = coins_rsi_data.get('coins', {}).get(symbol)
                                from bot_engine.config_loader import get_rsi_from_coin_data
                                current_rsi = get_rsi_from_coin_data(rsi_data, timeframe=bot_entry_timeframe) if rsi_data else None
                                current_price = rsi_data.get('price') if rsi_data else None

                                # ✅ Боты в позиции: при отсутствии RSI в общем кэше — загружаем только последние 20 свечей и считаем RSI (достаточно для RSI(14), без лишней нагрузки на API)
                                # Таймфрейм свечей = таймфрейм бота (entry_timeframe).
                                if current_rsi is None or current_price is None:
                                    try:
                                        try:
                                            chart_response = exchange_obj.get_chart_data(
                                                symbol, bot_entry_timeframe, '1w',
                                                bulk_mode=True, bulk_limit=20
                                            )
                                        except TypeError:
                                            chart_response = exchange_obj.get_chart_data(symbol, bot_entry_timeframe, '1w')
                                        if chart_response and chart_response.get('success'):
                                            candles = chart_response.get('data', {}).get('candles', [])
                                            if len(candles) >= 15:
                                                from bots_modules.calculations import calculate_rsi
                                                closes = [float(c.get('close', 0)) for c in candles]
                                                current_rsi = calculate_rsi(closes, 14)
                                                current_price = candles[-1].get('close') if candles else None
                                    except Exception as fetch_err:
                                        logger.debug(f" Монитор позиций: RSI для {symbol} по свечам: {fetch_err}")

                                if current_rsi is None or current_price is None:
                                    continue

                                # ✅ ОПТИМИЗАЦИЯ: Используем статический метод без создания объекта бота
                                rsi_should_close, rsi_reason = NewTradingBot.check_should_close_by_rsi(symbol, current_rsi, position_side)
                                should_close, reason = NewTradingBot.check_exit_with_breakeven_wait(
                                    symbol, bot_data, current_price, position_side, rsi_should_close, rsi_reason
                                )

                                if should_close:
                                    logger.info(f" 🔴 {symbol}: Закрываем {position_side} (RSI={current_rsi:.2f}, reason={reason})")
                                    trading_bot = NewTradingBot(symbol, bot_data, exchange_obj)
                                    close_result = trading_bot._close_position_on_exchange(reason)
                                    if close_result:
                                        logger.info(f" ✅ {symbol}: Позиция закрыта")
                                    else:
                                        logger.error(f" ❌ {symbol}: Ошибка закрытия!")

                            except Exception as bot_error:
                                logger.error(f" ❌ {symbol}: {bot_error}")
                                import traceback
                                logger.error(f" ❌ Traceback: {traceback.format_exc()}")

                    last_rsi_close_check = current_time

                except Exception as e:
                    logger.error(f" ❌ Ошибка проверки закрытия позиций по RSI: {e}")
                    import traceback
                    logger.error(f" ❌ Traceback: {traceback.format_exc()}")

            _gc_ticks += 1
            if _gc_ticks >= 60:
                force_collect_full()
                _gc_ticks = 0

            # Ждем 1 секунду перед следующей проверкой - КАЖДУЮ СЕКУНДУ!
            time.sleep(1)

        except Exception as e:
            logger.error(f" ❌ Критическая ошибка: {e}")
            time.sleep(10)

    logger.warning(" 🛑 Мониторинг позиций остановлен")

# Глобальный кэш позиций
positions_cache = {
    'positions': [],
    'last_update': None,
    'symbols_with_positions': set()
}
