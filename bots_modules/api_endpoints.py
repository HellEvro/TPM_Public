"""Flask API endpoints для сервиса ботов

Все API endpoints для управления ботами, конфигурацией, позициями и т.д.
"""

import logging
import json
import os
import time
import threading
import sys
import importlib
from datetime import datetime
from flask import Flask, request, jsonify

logger = logging.getLogger('BotsService')

# Импорт SystemConfig
from bot_engine.bot_config import SystemConfig

# Импорт Flask приложения и глобальных переменных из imports_and_globals
from bots_modules.imports_and_globals import (
    bots_app, exchange, smart_rsi_manager, async_processor,
    bots_data_lock, bots_data, rsi_data_lock, coins_rsi_data,
    bots_cache_data, bots_cache_lock, process_state,
    system_initialized, shutdown_flag, mature_coins_storage,
    mature_coins_lock, optimal_ema_data, coin_processing_locks,
    BOT_STATUS, ASYNC_AVAILABLE, RSI_CACHE_FILE, bot_history_manager,
    get_exchange
)
import bots_modules.imports_and_globals as globals_module

# Импорт RSI констант из bot_config
# Enhanced RSI константы теперь в SystemConfig

# Импорт констант интервалов
try:
    from bots_modules.sync_and_cache import SYSTEM_CONFIG_FILE
except ImportError:
    SYSTEM_CONFIG_FILE = 'data/system_config.json'

# Импорт функций из других модулей
try:
    from bots_modules.sync_and_cache import (
        update_bots_cache_data, save_system_config, load_system_config,
        save_auto_bot_config, save_bots_state,
        save_optimal_ema_periods,
        restore_default_config, load_default_config
    )
    from bots_modules.init_functions import ensure_exchange_initialized, create_bot
    from bots_modules.maturity import (
        save_mature_coins_storage, load_mature_coins_storage,
        remove_mature_coin_from_storage, check_coin_maturity_with_storage
    )
    from bots_modules.optimal_ema import (
        load_optimal_ema_data, update_optimal_ema_data
    )
    from bots_modules.filters import (
        get_effective_signal, check_auto_bot_filters,
        process_auto_bot_signals, test_exit_scam_filter, test_rsi_time_filter,
        process_trading_signals_for_all_bots
    )
    # Для clear_mature_coins_storage может быть в разных модулях
    try:
        from bots_modules.maturity import clear_mature_coins_storage as clear_mature_storage
    except:
        def clear_mature_storage():
            pass
except ImportError as e:
    print(f"Warning: Could not import functions in api_endpoints: {e}")
    # Заглушки если импорт не удался
    def update_bots_cache_data():
        pass
    def save_system_config(config):
        pass
    def load_system_config():
        return {}
    def save_auto_bot_config():
        pass
    def save_bots_state():
        pass
    def get_effective_signal(coin):
        # Используем настоящую функцию из filters.py
        from bots_modules.filters import get_effective_signal as real_get_effective_signal
        return real_get_effective_signal(coin)
    def check_auto_bot_filters(symbol):
        return {'allowed': True}
    def process_auto_bot_signals(exchange_obj=None):
        pass
    def test_exit_scam_filter(symbol):
        pass
    def test_rsi_time_filter(symbol):
        pass

# Функция для очистки данных для JSON
def clean_data_for_json(data):
    """Очищает данные для безопасной JSON сериализации"""
    if data is None:
        return None
    if isinstance(data, (str, int, float, bool)):
        return data
    if isinstance(data, dict):
        return {k: clean_data_for_json(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [clean_data_for_json(item) for item in data]
    return str(data)

# Дополнительные импорты из sync_and_cache
try:
    from bots_modules.sync_and_cache import (
        sync_positions_with_exchange, cleanup_inactive_bots,
        remove_mature_coins, check_trading_rules_activation
    )
except:
    def sync_positions_with_exchange():
        pass
    def cleanup_inactive_bots():
        pass
    def remove_mature_coins(symbols):
        pass
    def check_trading_rules_activation():
        pass

try:
    from bots_modules.optimal_ema import get_optimal_ema_periods
except:
    def get_optimal_ema_periods(symbol):
        return {}

def start_async_processor():
    pass

def stop_async_processor():
    pass

def health_check():
    """Проверка состояния сервиса"""
    try:
        logger.info(f"[HEALTH_CHECK] ✅ Flask работает, запрос обработан")
        return jsonify({
            'status': 'ok',
            'service': 'bots',
            'timestamp': datetime.now().isoformat(),
            'exchange_connected': exchange is not None,
            'coins_loaded': len(coins_rsi_data.get('coins', {})),
            'bots_active': len(bots_data.get('bots', {}))
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'service': 'bots',
            'error': str(e)
        }), 500

@bots_app.route('/api/bots/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервиса"""
    try:
        return jsonify({
            'success': True,
            'status': 'healthy',
            'service': 'bots',
            'timestamp': datetime.now().isoformat(),
            'uptime': time.time() - globals_module.service_start_time if hasattr(globals_module, 'service_start_time') else 0,
            'components': {
                'exchange': exchange is not None,
                'coins_loaded': len(coins_rsi_data.get('coins', {})) > 0,
                'bots_active': len(bots_data.get('bots', {}))
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@bots_app.route('/api/bots/status', methods=['GET'])
def get_service_status():
    """Получить статус сервиса ботов"""
    try:
        return jsonify({
            'success': True,
            'status': 'running',
            'service': 'bots',
            'timestamp': datetime.now().isoformat(),
            'last_update': coins_rsi_data.get('last_update'),
            'update_in_progress': coins_rsi_data.get('update_in_progress', False),
            'coins_loaded': len(coins_rsi_data.get('coins', {})),
            'total_coins': coins_rsi_data.get('total_coins', 0),
            'successful_coins': coins_rsi_data.get('successful_coins', 0),
            'failed_coins': coins_rsi_data.get('failed_coins', 0),
            'bots': {
                'total': len(bots_data.get('bots', {})),
                'active': len([b for b in bots_data.get('bots', {}).values() if b.get('status') not in ['paused', 'idle']])
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bots_app.route('/api/bots/pairs', methods=['GET'])
def get_trading_pairs():
    """Получить список торговых пар"""
    try:
        # Получаем список монет из RSI данных
        coins = list(coins_rsi_data.get('coins', {}).keys())
        
        return jsonify({
            'success': True,
            'pairs': [f"{coin}USDT" for coin in coins],
            'total': len(coins),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bots_app.route('/api/bots/async-status', methods=['GET'])
def get_async_status():
    """Получает статус асинхронного процессора"""
    try:
        global async_processor, async_processor_task
        
        status = {
            'available': ASYNC_AVAILABLE,
            'running': async_processor is not None and async_processor.is_running,
            'task_active': async_processor_task is not None and async_processor_task.is_alive(),
            'last_update': async_processor.last_update if async_processor else 0,
            'active_tasks': len(async_processor.active_tasks) if async_processor else 0
        }
        
        return jsonify({
            'success': True,
            'async_status': status
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bots_app.route('/api/bots/async-control', methods=['POST'])
def control_async_processor():
    """Управляет асинхронным процессором"""
    try:
        data = request.get_json()
        action = data.get('action')
        
        if action == 'start':
            if async_processor is None:
                success = start_async_processor()
                return jsonify({
                    'success': success,
                    'message': 'Асинхронный процессор запущен' if success else 'Ошибка запуска'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Асинхронный процессор уже запущен'
                })
        
        elif action == 'stop':
            if async_processor is not None:
                stop_async_processor()
                return jsonify({
                    'success': True,
                    'message': 'Асинхронный процессор остановлен'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Асинхронный процессор не запущен'
                })
        
        elif action == 'restart':
            stop_async_processor()
            time.sleep(1)  # Небольшая пауза
            success = start_async_processor()
            return jsonify({
                'success': success,
                'message': 'Асинхронный процессор перезапущен' if success else 'Ошибка перезапуска'
            })
        
        else:
            return jsonify({
                'success': False,
                'error': 'Неизвестное действие'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bots_app.route('/api/bots/account-info', methods=['GET'])
def get_account_info():
    """Получает информацию о едином торговом счете (напрямую с биржи)"""
    try:
        # Получаем данные напрямую с биржи (без кэширования)
        if not ensure_exchange_initialized():
            return jsonify({
                'success': False,
                'error': 'Exchange not initialized'
            }), 500
        
        # Получаем актуальные данные с биржи
        current_exchange = get_exchange()
        if not current_exchange:
            return jsonify({
                'success': False,
                'error': 'Exchange not initialized'
            }), 500
        account_info = current_exchange.get_unified_account_info()
        if not account_info.get("success"):
            account_info = {
                'success': False,
                'error': 'Failed to get account info from exchange'
            }
        
        # Добавляем информацию о ботах из актуальных данных
        # ⚡ БЕЗ БЛОКИРОВКИ: чтение словаря - атомарная операция
        bots_list = list(bots_data['bots'].values())
        account_info["bots_count"] = len(bots_list)
        account_info["active_bots"] = sum(1 for bot in bots_list 
                                        if bot.get('status') not in ['paused'])
        
        response = jsonify(account_info)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except Exception as e:
        logger.error(f"[ERROR] Ошибка получения информации о счете: {str(e)}")
        response = jsonify({
            "success": False,
            "error": str(e)
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500

@bots_app.route('/api/bots/manual-positions/refresh', methods=['POST'])
def refresh_manual_positions():
    """Обновить список монет с ручными позициями на бирже (позиции БЕЗ ботов)"""
    try:
        manual_positions = []
        
        # Получаем exchange объект
        try:
            exchange = get_exchange()
        except ImportError:
            exchange = None
        
        if exchange:
            exchange_positions = exchange.get_positions()
            
            if isinstance(exchange_positions, tuple):
                positions_list = exchange_positions[0] if exchange_positions else []
            else:
                positions_list = exchange_positions if exchange_positions else []
            
            # Получаем список символов с ботами (включая сохраненных)
            with bots_data_lock:
                active_bot_symbols = set(bots_data['bots'].keys())
            
            # Также загружаем сохраненных ботов из файла
            saved_bot_symbols = set()
            try:
                import json
                bots_state_file = 'data/bots_state.json'
                if os.path.exists(bots_state_file):
                    with open(bots_state_file, 'r', encoding='utf-8') as f:
                        saved_data = json.load(f)
                        if 'bots' in saved_data:
                            saved_bot_symbols = set(saved_data['bots'].keys())
            except Exception as e:
                logger.warning(f"[MANUAL_POSITIONS] ⚠️ Не удалось загрузить сохраненных ботов: {e}")
            
            # Объединяем активных и сохраненных ботов
            system_bot_symbols = active_bot_symbols.union(saved_bot_symbols)
            
            # Извлекаем символы с активными позициями, для которых НЕТ бота в системе
            for pos in positions_list:
                if abs(float(pos.get('size', 0))) > 0:
                    symbol = pos.get('symbol', '')
                    # Убираем USDT из символа для сопоставления с coins_rsi_data
                    clean_symbol = symbol.replace('USDT', '') if symbol else ''
                    
                    # ✅ РУЧНЫЕ ПОЗИЦИИ = позиции на бирже БЕЗ бота в системе
                    if clean_symbol and clean_symbol not in system_bot_symbols:
                        if clean_symbol not in manual_positions:
                            manual_positions.append(clean_symbol)
            
            
        return jsonify({
            'success': True,
            'count': len(manual_positions),
            'positions': manual_positions
        })
        
    except Exception as e:
        logger.error(f"[MANUAL_POSITIONS] ❌ Ошибка обновления ручных позиций: {str(e)}")
        import traceback
        logger.error(f"[MANUAL_POSITIONS] ❌ Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/coins-with-rsi', methods=['GET'])
def get_coins_with_rsi():
    """Получить все монеты с RSI 6H данными"""
    try:
        # Проверяем параметр refresh_symbol для обновления конкретной монеты
        refresh_symbol = request.args.get('refresh_symbol')
        if refresh_symbol:
            logger.info(f"[API] 🔄 Запрос на обновление RSI данных для {refresh_symbol}")
            try:
                if ensure_exchange_initialized():
                    coin_data = get_coin_rsi_data(refresh_symbol, get_exchange())
                    if coin_data:
                        # ⚡ БЕЗ БЛОКИРОВКИ: GIL делает запись атомарной
                        coins_rsi_data['coins'][refresh_symbol] = coin_data
                        logger.info(f"[API] ✅ RSI данные для {refresh_symbol} обновлены")
                    else:
                        logger.warning(f"[API] ⚠️ Не удалось обновить RSI данные для {refresh_symbol}")
            except Exception as e:
                logger.error(f"[API] ❌ Ошибка обновления RSI для {refresh_symbol}: {e}")
        
        # ⚡ БЕЗ БЛОКИРОВКИ: чтение словаря - атомарная операция
        # Проверяем возраст кэша
        cache_age = None
        if os.path.exists(RSI_CACHE_FILE):
            try:
                cache_stat = os.path.getmtime(RSI_CACHE_FILE)
                cache_age = (time.time() - cache_stat) / 60  # в минутах
            except:
                cache_age = None
        
        # Очищаем данные от несериализуемых объектов
        cleaned_coins = {}
        for symbol, coin_data in coins_rsi_data['coins'].items():
            # ✅ ИСПРАВЛЕНИЕ: НЕ фильтруем монеты по зрелости для UI!
            # Фильтр зрелости применяется в get_coin_rsi_data() через изменение сигнала на WAIT
            # Здесь показываем ВСЕ монеты, независимо от зрелости
                
            cleaned_coin = coin_data.copy()
            
            # Очищаем enhanced_rsi от numpy типов и других несериализуемых объектов
            if 'enhanced_rsi' in cleaned_coin and cleaned_coin['enhanced_rsi']:
                enhanced_rsi = cleaned_coin['enhanced_rsi'].copy()
                
                # Конвертируем numpy типы в Python типы
                if 'confirmations' in enhanced_rsi:
                    confirmations = enhanced_rsi['confirmations'].copy()
                    for key, value in confirmations.items():
                        if hasattr(value, 'item'):  # numpy scalar
                            confirmations[key] = value.item()
                        elif value is None:
                            confirmations[key] = None
                    enhanced_rsi['confirmations'] = confirmations
                
                # Конвертируем adaptive_levels если это tuple
                if 'adaptive_levels' in enhanced_rsi and enhanced_rsi['adaptive_levels']:
                    if isinstance(enhanced_rsi['adaptive_levels'], tuple):
                        enhanced_rsi['adaptive_levels'] = list(enhanced_rsi['adaptive_levels'])
                
                cleaned_coin['enhanced_rsi'] = enhanced_rsi
            
            # Добавляем эффективный сигнал для единообразия с фронтендом
            # Вычисляем эффективный сигнал после очистки от numpy типов
            effective_signal = get_effective_signal(cleaned_coin)
            cleaned_coin['effective_signal'] = effective_signal
            
            # ✅ ИСПРАВЛЕНИЕ: Копируем Stochastic RSI из enhanced_rsi в основные поля
            if 'enhanced_rsi' in cleaned_coin and cleaned_coin['enhanced_rsi']:
                enhanced_rsi = cleaned_coin['enhanced_rsi']
                if 'confirmations' in enhanced_rsi:
                    confirmations = enhanced_rsi['confirmations']
                    # Копируем Stochastic RSI данные в основные поля для совместимости с UI
                    cleaned_coin['stoch_rsi_k'] = confirmations.get('stoch_rsi_k')
                    cleaned_coin['stoch_rsi_d'] = confirmations.get('stoch_rsi_d')
            
            # ✅ ИСПРАВЛЕНИЕ: Добавляем количество свечей из данных зрелых монет
            try:
                from bots_modules.imports_and_globals import mature_coins_storage
                if symbol in mature_coins_storage:
                    maturity_data = mature_coins_storage[symbol].get('maturity_data', {})
                    details = maturity_data.get('details', {})
                    cleaned_coin['candles_count'] = details.get('candles_count')
                else:
                    cleaned_coin['candles_count'] = None
            except Exception as e:
                logger.debug(f"[API] Ошибка получения candles_count для {symbol}: {e}")
                cleaned_coin['candles_count'] = None
            
            cleaned_coins[symbol] = cleaned_coin
        
        # Получаем список монет с ручными позициями на бирже (позиции БЕЗ ботов)
        manual_positions = []
        try:
            # Получаем exchange объект
            try:
                exchange = get_exchange()
            except ImportError:
                exchange = None
            
            if exchange:
                exchange_positions = exchange.get_positions()
                if isinstance(exchange_positions, tuple):
                    positions_list = exchange_positions[0] if exchange_positions else []
                else:
                    positions_list = exchange_positions if exchange_positions else []
                
                # Получаем список символов с ботами (включая сохраненных)
                # ⚡ БЕЗ БЛОКИРОВКИ: чтение словаря - атомарная операция
                active_bot_symbols = set(bots_data['bots'].keys())
                
                # Также загружаем сохраненных ботов из файла
                saved_bot_symbols = set()
                try:
                    bots_state_file = 'data/bots_state.json'
                    if os.path.exists(bots_state_file):
                        with open(bots_state_file, 'r', encoding='utf-8') as f:
                            saved_data = json.load(f)
                            if 'bots' in saved_data:
                                saved_bot_symbols = set(saved_data['bots'].keys())
                except Exception as e:
                    logger.warning(f"[MANUAL_POSITIONS] ⚠️ Не удалось загрузить сохраненных ботов: {e}")
                
                # Объединяем активных и сохраненных ботов
                system_bot_symbols = active_bot_symbols.union(saved_bot_symbols)
                
                # Извлекаем символы с активными позициями, для которых НЕТ бота в системе
                for pos in positions_list:
                    if abs(float(pos.get('size', 0))) > 0:
                        symbol = pos.get('symbol', '')
                        # Убираем USDT из символа для сопоставления с coins_rsi_data
                        clean_symbol = symbol.replace('USDT', '') if symbol else ''
                        
                        # ✅ РУЧНЫЕ ПОЗИЦИИ = позиции на бирже БЕЗ бота в системе
                        if clean_symbol and clean_symbol not in system_bot_symbols:
                            if clean_symbol not in manual_positions:
                                manual_positions.append(clean_symbol)
                
                # ✅ Детальное логирование для отладки
                logger.debug(f"[MANUAL_POSITIONS] Найдено {len(manual_positions)} ручных позиций: {manual_positions}")
        except Exception as e:
            logger.error(f"[ERROR] Ошибка получения ручных позиций: {str(e)}")
        
        result = {
            'success': True,
            'coins': cleaned_coins,
            'total': len(cleaned_coins),
            'last_update': coins_rsi_data['last_update'],
            'update_in_progress': coins_rsi_data['update_in_progress'],
            'data_version': coins_rsi_data.get('data_version', 0),  # ✅ Версия данных для оптимизации UI
            'manual_positions': manual_positions,  # Добавляем список ручных позиций
            'cache_info': {
                'cache_exists': os.path.exists(RSI_CACHE_FILE),
                'cache_age_minutes': round(cache_age, 1) if cache_age else None,
                'data_source': 'cache' if cache_age and cache_age < 360 else 'live'  # 6 часов
            },
            'stats': {
                'total_coins': coins_rsi_data['total_coins'],
                'successful_coins': coins_rsi_data['successful_coins'],
                'failed_coins': coins_rsi_data['failed_coins']
            }
        }
        
        # Убираем спам-лог, только в debug режиме
        if SystemConfig.DEBUG_MODE:
            logger.debug(f"[API] Возврат RSI данных для {len(result['coins'])} монет")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"[ERROR] Ошибка получения монет с RSI: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

def clean_data_for_json(data):
    """Очищает данные от numpy типов для JSON сериализации"""
    if data is None:
        return None
    elif isinstance(data, dict):
        return {k: clean_data_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_data_for_json(item) for item in data]
    elif hasattr(data, 'tolist'):  # numpy array
        return data.tolist()
    elif hasattr(data, 'item'):  # numpy scalar
        return data.item()
    elif hasattr(data, 'dtype'):  # numpy тип
        # Обрабатываем все numpy типы
        if data.dtype.kind == 'b':  # boolean
            return bool(data)
        elif data.dtype.kind in ['i', 'u']:  # integer
            return int(data)
        elif data.dtype.kind == 'f':  # float
            return float(data)
        else:
            return str(data)
    else:
        return data

@bots_app.route('/api/bots/list', methods=['GET'])
def get_bots_list():
    """Получить список всех ботов (использует bots_data напрямую)"""
    try:
        # Используем bots_data напрямую для актуальности
        # ⚡ БЕЗ БЛОКИРОВКИ: GIL делает чтение атомарным
        bots_list = list(bots_data['bots'].values())
        auto_bot_enabled = bots_data.get('auto_bot_config', {}).get('enabled', False)
        
        # Получаем время последнего обновления
        last_update_raw = bots_data.get('last_update')
        if last_update_raw:
            try:
                # Если это datetime объект, форматируем его
                if hasattr(last_update_raw, 'isoformat'):
                    last_update = last_update_raw.isoformat()
                else:
                    # Если это строка, используем как есть
                    last_update = str(last_update_raw)
            except:
                last_update = 'Ошибка форматирования'
        else:
            last_update = 'Никогда не обновлялся'
        
        # Добавляем расчет времени работы и last_update для каждого бота
        current_time = datetime.now()
        for bot in bots_list:
            created_at_str = bot.get('created_at')
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str)
                    work_duration = current_time - created_at
                    work_minutes = int(work_duration.total_seconds() / 60)
                    work_seconds = int(work_duration.total_seconds() % 60)
                    
                    # Форматируем время работы
                    if work_minutes > 0:
                        bot['work_time'] = f"{work_minutes}м {work_seconds}с"
                    else:
                        bot['work_time'] = f"{work_seconds}с"
                except (ValueError, TypeError):
                    bot['work_time'] = "0с"
            else:
                bot['work_time'] = "0с"
            
            # Добавляем last_update для каждого бота
            bot_last_update = bot.get('last_update')
            if bot_last_update:
                try:
                    if hasattr(bot_last_update, 'isoformat'):
                        bot['last_update'] = bot_last_update.isoformat()
                    else:
                        bot['last_update'] = str(bot_last_update)
                except:
                    bot['last_update'] = 'Ошибка'
            else:
                bot['last_update'] = 'Никогда'
            
            # Добавляем статус бота для UI
            bot_status = bot.get('status', 'unknown')
            if bot_status in ['in_position_long', 'in_position_short']:
                bot['bot_status'] = 'Активен'
            elif bot_status == 'idle':
                bot['bot_status'] = 'Ожидает'
            elif bot_status == 'paused':
                bot['bot_status'] = 'Приостановлен'
            else:
                bot['bot_status'] = 'Неизвестно'
        
        # Подсчитываем статистику (idle боты считаются активными для UI)
        active_bots = sum(1 for bot in bots_list if bot.get('status') not in ['paused'])
        
        response_data = {
            'success': True,
            'bots': bots_list,
            'count': len(bots_list),
            'auto_bot_enabled': auto_bot_enabled,
            'last_update': last_update,
            'stats': {
                'active_bots': active_bots,
                'total_bots': len(bots_list)
            }
        }
        
        # ✅ Не логируем частые запросы списка ботов
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"[API] ❌ Ошибка получения списка ботов: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'bots': [],
            'count': 0
        }), 500

@bots_app.route('/api/bots/create', methods=['POST'])
def create_bot_endpoint():
    """Создать нового бота"""
    try:
        # Проверяем что биржа инициализирована
        if not ensure_exchange_initialized():
            return jsonify({
                'success': False, 
                'error': 'Биржа не инициализирована. Попробуйте позже.'
            }), 503
        
        data = request.get_json()
        if not data or not data.get('symbol'):
            return jsonify({'success': False, 'error': 'Symbol required'}), 400
        
        symbol = data['symbol']
        config = data.get('config', {})
        
        logger.info(f"[BOT_CREATE] Запрос на создание бота для {symbol}")
        logger.info(f"[BOT_CREATE] Конфигурация: {config}")
        
        # Проверяем зрелость монеты (если включена проверка для этой монеты)
        enable_maturity_check_coin = config.get('enable_maturity_check', True)
        if enable_maturity_check_coin:
            # Получаем данные свечей для проверки зрелости
            current_exchange = get_exchange()
            if not current_exchange:
                return jsonify({
                    'success': False,
                    'error': 'Exchange not initialized'
                }), 503
            chart_response = current_exchange.get_chart_data(symbol, '6h', '30d')
            if chart_response and chart_response.get('success'):
                candles = chart_response['data']['candles']
                if candles and len(candles) >= 15:
                    maturity_check = check_coin_maturity_with_storage(symbol, candles)
                    if not maturity_check['is_mature']:
                        logger.warning(f"[BOT_CREATE] {symbol}: Монета не прошла проверку зрелости - {maturity_check['reason']}")
                        return jsonify({
                            'success': False, 
                            'error': f'Монета {symbol} не прошла проверку зрелости: {maturity_check["reason"]}',
                            'maturity_details': maturity_check['details']
                        }), 400
                else:
                    logger.warning(f"[BOT_CREATE] {symbol}: Недостаточно данных для проверки зрелости")
                    return jsonify({
                        'success': False, 
                        'error': f'Недостаточно данных для проверки зрелости монеты {symbol}'
                    }), 400
            else:
                logger.warning(f"[BOT_CREATE] {symbol}: Не удалось получить данные для проверки зрелости")
                return jsonify({
                    'success': False, 
                    'error': f'Не удалось получить данные для проверки зрелости монеты {symbol}'
                }), 400
        
        # Создаем бота
        bot_config = create_bot(symbol, config, exchange_obj=get_exchange())
        
        # ✅ Проверяем: есть ли уже позиция на бирже для этой монеты?
        has_existing_position = False
        try:
            # Проверяем через exchange напрямую (более надежно)
            current_exchange = get_exchange()
            if current_exchange:
                positions_response = current_exchange.get_positions()
                if isinstance(positions_response, tuple):
                    positions_list = positions_response[0] if positions_response else []
                else:
                    positions_list = positions_response if positions_response else []
                
                # Ищем позицию для этой монеты
                for pos in positions_list:
                    pos_symbol = pos.get('symbol', '').replace('USDT', '')
                    if pos_symbol == symbol and abs(float(pos.get('size', 0))) > 0:
                        has_existing_position = True
                        logger.info(f"[BOT_CREATE] 🔍 {symbol}: Обнаружена существующая позиция на бирже (размер: {pos.get('size')})")
                        break
        except Exception as e:
            logger.debug(f"[BOT_CREATE] Не удалось проверить существующую позицию: {e}")
        
        # ✅ Возвращаем ответ БЫСТРО
        logger.info(f"[BOT_CREATE] ✅ Бот для {symbol} создан")
        
        # ✅ Запускаем вход в позицию АСИНХРОННО (только если НЕТ существующей позиции!)
        if not has_existing_position:
            def enter_position_async():
                try:
                    with rsi_data_lock:
                        coin_data = coins_rsi_data['coins'].get(symbol)
                        if coin_data and coin_data.get('signal') in ['ENTER_LONG', 'ENTER_SHORT']:
                            signal = coin_data.get('signal')
                            direction = 'LONG' if signal == 'ENTER_LONG' else 'SHORT'
                            
                            logger.info(f"[BOT_CREATE_ASYNC] 🚀 Входим в {direction} позицию для {symbol}")
                            
                            from bots_modules.bot_class import NewTradingBot
                            bot_instance = NewTradingBot(symbol, bot_config, get_exchange())
                            
                            result = bot_instance.enter_position(direction)
                            if result:
                                logger.info(f"[BOT_CREATE_ASYNC] ✅ Успешно вошли в {direction} позицию для {symbol}")
                                with bots_data_lock:
                                    bots_data['bots'][symbol] = bot_instance.to_dict()
                            else:
                                logger.error(f"[BOT_CREATE_ASYNC] ❌ НЕ УДАЛОСЬ войти в {direction} позицию для {symbol}")
                        else:
                            logger.info(f"[BOT_CREATE_ASYNC] ℹ️ Нет активного сигнала для {symbol}, бот будет ждать")
                except Exception as e:
                    logger.error(f"[BOT_CREATE_ASYNC] ❌ Ошибка входа в позицию: {e}")
            
            # Запускаем асинхронно
            thread = threading.Thread(target=enter_position_async)
            thread.daemon = True
            thread.start()
        else:
            # ✅ Для существующей позиции - просто запускаем синхронизацию
            logger.info(f"[BOT_CREATE] 🔄 {symbol}: Запускаем синхронизацию существующей позиции...")
            
            def sync_existing_position():
                try:
                    from bots_modules.sync_and_cache import sync_bots_with_exchange
                    sync_bots_with_exchange()
                    logger.info(f"[BOT_CREATE_ASYNC] ✅ Синхронизация позиции {symbol} завершена")
                except Exception as e:
                    logger.error(f"[BOT_CREATE_ASYNC] ❌ Ошибка синхронизации: {e}")
            
            thread = threading.Thread(target=sync_existing_position)
            thread.daemon = True
            thread.start()
        
        return jsonify({
            'success': True,
            'message': f'Бот для {symbol} создан успешно',
            'bot': bot_config,
            'existing_position': has_existing_position
        })
        
    except Exception as e:
        logger.error(f"[ERROR] Ошибка создания бота: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/start', methods=['POST'])
def start_bot_endpoint():
    """Запустить бота"""
    try:
        data = request.get_json()
        if not data or not data.get('symbol'):
            return jsonify({'success': False, 'error': 'Symbol required'}), 400
        
        symbol = data['symbol']
        
        with bots_data_lock:
            if symbol not in bots_data['bots']:
                return jsonify({'success': False, 'error': 'Bot not found'}), 404
            
            bot_data = bots_data['bots'][symbol]
            if bot_data['status'] in [BOT_STATUS['PAUSED'], BOT_STATUS['IDLE']]:
                bot_data['status'] = BOT_STATUS['RUNNING']
                logger.info(f"[BOT] {symbol}: Бот запущен (снята пауза)")
            else:
                logger.info(f"[BOT] {symbol}: Бот уже активен")
        
        return jsonify({
            'success': True,
            'message': f'Бот для {symbol} запущен'
        })
            
    except Exception as e:
        logger.error(f"[ERROR] Ошибка запуска бота: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/stop', methods=['POST'])
def stop_bot_endpoint():
    """Остановить бота"""
    try:
        logger.info(f"[API] 📥 Получен запрос остановки бота: {request.get_data()}")
        
        # Пробуем разные способы получения данных
        try:
            data = request.get_json()
        except Exception as json_error:
            logger.error(f"[API] ❌ Ошибка парсинга JSON: {json_error}")
            # Пробуем получить данные как form data
            data = request.form.to_dict()
            if not data:
                # Пробуем получить данные из args
                data = request.args.to_dict()
        
        logger.info(f"[API] 📊 Распарсенные данные: {data}")
        
        if not data or not data.get('symbol'):
            logger.error(f"[API] ❌ Отсутствует symbol в данных: {data}")
            return jsonify({'success': False, 'error': 'Symbol required'}), 400
        
        symbol = data['symbol']
        reason = data.get('reason', 'Остановлен пользователем')
        
        # Проверяем, есть ли открытая позиция у бота
        # ⚡ БЕЗ БЛОКИРОВКИ: чтение и простое присваивание - атомарные операции
        position_to_close = None
        
        logger.info(f"[API] 🔍 Проверяем наличие бота {symbol}...")
        if symbol not in bots_data['bots']:
            logger.error(f"[API] ❌ Бот {symbol} не найден!")
            return jsonify({'success': False, 'error': 'Bot not found'}), 404
        
        logger.info(f"[API] ✅ Бот {symbol} найден, останавливаем...")
        
        # ⚡ КРИТИЧНО: Используем блокировку для атомарной операции!
        with bots_data_lock:
            bot_data = bots_data['bots'][symbol]
            old_status = bot_data['status']
            logger.info(f"[API] 📊 Старый статус: {old_status}")
            
            # Проверяем, есть ли открытая позиция
            if bot_data.get('position_side') in ['LONG', 'SHORT']:
                position_to_close = bot_data['position_side']
                logger.info(f"[BOT] {symbol}: Найдена открытая позиция {position_to_close}, будет закрыта при остановке")
            
            bot_data['status'] = BOT_STATUS['PAUSED']
            # НЕ сбрасываем entry_price для возможности возобновления
            # НЕ сбрасываем position_side - оставляем для отображения в UI
            # bot_data['position_side'] = None
            # bot_data['unrealized_pnl'] = 0.0
            logger.info(f"[BOT] {symbol}: Бот остановлен, новый статус: {bot_data['status']}")
            
            # Обновляем глобальную статистику
            bots_data['global_stats']['active_bots'] = len([bot for bot in bots_data['bots'].values() if bot.get('status') in ['running', 'idle']])
            bots_data['global_stats']['bots_in_position'] = len([bot for bot in bots_data['bots'].values() if bot.get('position_side')])
        
        # ⚠️ НЕ ЗАКРЫВАЕМ ПОЗИЦИЮ АВТОМАТИЧЕСКИ - это вызывает зависание!
        # Позиция останется на бирже и закроется при следующей проверке
        if position_to_close:
            logger.info(f"[BOT] {symbol}: ⚠️ Позиция {position_to_close} осталась открытой на бирже (закроется автоматически)")
        
        # Логируем остановку бота в историю
        # log_bot_stop(symbol, reason)  # TODO: Функция не определена
        
        # Сохраняем состояние после остановки
        save_bots_state()
        
        # ⚠️ НЕ обновляем кэш - он перезапишет статус!
        # update_bots_cache_data()
        
        # ⚡ ПРИНУДИТЕЛЬНАЯ ПРОВЕРКА: убеждаемся что статус сохранен
        with bots_data_lock:
            final_status = bots_data['bots'][symbol]['status']
            if final_status != BOT_STATUS['PAUSED']:
                logger.error(f"[BOT] {symbol}: ❌ КРИТИЧНАЯ ОШИБКА! Статус НЕ изменен: {final_status}")
                bots_data['bots'][symbol]['status'] = BOT_STATUS['PAUSED']
                save_bots_state()
                logger.error(f"[BOT] {symbol}: ✅ Статус принудительно исправлен на PAUSED")
            else:
                logger.info(f"[BOT] {symbol}: ✅ Статус корректно установлен: {final_status}")
        
        logger.info(f"[BOT] {symbol}: ✅ Кэш НЕ обновлен (статус PAUSED сохранен)")
        
        return jsonify({
            'success': True, 
            'message': f'Бот для {symbol} остановлен'
        })
        
    except Exception as e:
        logger.error(f"[ERROR] Ошибка остановки бота: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/pause', methods=['POST'])
def pause_bot_endpoint():
    """Приостановить бота"""
    try:
        data = request.get_json()
        if not data or not data.get('symbol'):
            return jsonify({'success': False, 'error': 'Symbol required'}), 400
        
        symbol = data['symbol']
        
        # Проверяем, есть ли открытая позиция у бота
        position_to_close = None
        with bots_data_lock:
            if symbol not in bots_data['bots']:
                return jsonify({'success': False, 'error': 'Bot not found'}), 404
            
            bot_data = bots_data['bots'][symbol]
            old_status = bot_data['status']
            
            # Проверяем, есть ли открытая позиция
            if bot_data.get('position_side') in ['LONG', 'SHORT']:
                position_to_close = bot_data['position_side']
                logger.info(f"[BOT] {symbol}: Найдена открытая позиция {position_to_close}, будет закрыта при приостановке")
            
            bot_data['status'] = BOT_STATUS['PAUSED']
            logger.info(f"[BOT] {symbol}: Бот приостановлен (был: {old_status})")
        
        # ⚠️ НЕ ЗАКРЫВАЕМ ПОЗИЦИЮ АВТОМАТИЧЕСКИ - это вызывает зависание!
        if position_to_close:
            logger.info(f"[BOT] {symbol}: ⚠️ Позиция {position_to_close} осталась открытой на бирже (закроется автоматически)")
        
        return jsonify({
            'success': True,
            'message': f'Бот для {symbol} приостановлен'
        })
        
    except Exception as e:
        logger.error(f"[ERROR] Ошибка приостановки бота: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/delete', methods=['POST'])
def delete_bot_endpoint():
    """Удалить бота"""
    try:
        logger.info(f"[API] 📥 Получен запрос удаления бота: {request.get_data()}")
        
        # Пробуем разные способы получения данных
        try:
            data = request.get_json()
        except Exception as json_error:
            logger.error(f"[API] ❌ Ошибка парсинга JSON: {json_error}")
            # Пробуем получить данные как form data
            data = request.form.to_dict()
            if not data:
                # Пробуем получить данные из args
                data = request.args.to_dict()
        
        logger.info(f"[API] 📊 Распарсенные данные: {data}")
        
        if not data or not data.get('symbol'):
            logger.error(f"[API] ❌ Отсутствует symbol в данных: {data}")
            return jsonify({'success': False, 'error': 'Symbol required'}), 400
        
        symbol = data['symbol']
        reason = data.get('reason', 'Удален пользователем')
        
        # ⚡ БЕЗ БЛОКИРОВКИ: операции с dict атомарны в Python
        logger.info(f"[API] 🔍 Ищем бота {symbol} в bots_data. Доступные боты: {list(bots_data['bots'].keys())}")
        if symbol not in bots_data['bots']:
            logger.error(f"[API] ❌ Бот {symbol} не найден в bots_data")
            return jsonify({'success': False, 'error': 'Bot not found'}), 404
        
        # ✅ ТУПО УДАЛЯЕМ БОТА ИЗ ФАЙЛА!
        del bots_data['bots'][symbol]
        logger.info(f"[BOT] {symbol}: Бот удален из файла")
        
        # Обновляем глобальную статистику
        bots_data['global_stats']['active_bots'] = len([bot for bot in bots_data['bots'].values() if bot.get('status') in ['running', 'idle']])
        bots_data['global_stats']['bots_in_position'] = len([bot for bot in bots_data['bots'].values() if bot.get('position_side')])
        
        # Сохраняем состояние после удаления
        save_bots_state()
        
        # Обновляем кэш после удаления (ЧТОБЫ БОТ НЕ ВИСЕЛ!)
        update_bots_cache_data()
        
        return jsonify({
            'success': True,
            'message': f'Бот для {symbol} удален'
        })
        
    except Exception as e:
        logger.error(f"[ERROR] Ошибка удаления бота: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/close-position', methods=['POST'])
def close_position_endpoint():
    """Принудительно закрыть позицию бота"""
    try:
        logger.info(f"[API] 📥 Получен запрос закрытия позиции: {request.get_data()}")
        
        # Пробуем разные способы получения данных
        try:
            data = request.get_json()
        except Exception as json_error:
            logger.error(f"[API] ❌ Ошибка парсинга JSON: {json_error}")
            # Пробуем получить данные как form data
            data = request.form.to_dict()
            if not data:
                # Пробуем получить данные из args
                data = request.args.to_dict()
        
        logger.info(f"[API] 📊 Распарсенные данные: {data}")
        
        if not data or not data.get('symbol'):
            logger.error(f"[API] ❌ Отсутствует symbol в данных: {data}")
            return jsonify({'success': False, 'error': 'Symbol required'}), 400
        
        symbol = data['symbol']
        force_close = data.get('force', False)  # Принудительное закрытие даже если бот не в позиции
        
        current_exchange = get_exchange()
        if not current_exchange:
            logger.error(f"[API] ❌ Биржа не инициализирована")
            return jsonify({'success': False, 'error': 'Exchange not initialized'}), 500
        
        # ⚡ ИСПРАВЛЕНИЕ: Получаем актуальные позиции с биржи вместо кэша
        try:
            positions_response = current_exchange.get_positions()
            if isinstance(positions_response, tuple):
                positions = positions_response[0] if positions_response else []
            else:
                positions = positions_response if positions_response else []
        except Exception as e:
            logger.error(f"[API] ❌ Ошибка получения позиций с биржи: {e}")
            positions = []
        
        # Ищем позиции для данного символа
        symbol_positions = []
        for pos in positions:
            if pos['symbol'] == f"{symbol}USDT" and float(pos.get('size', 0)) > 0:
                symbol_positions.append(pos)
        
        if not symbol_positions:
            logger.warning(f"[API] ⚠️ Позиции для {symbol} не найдены на бирже")
            return jsonify({
                'success': False, 
                'message': f'Позиции для {symbol} не найдены на бирже'
            }), 404
        
        # Закрываем все найденные позиции
        closed_positions = []
        errors = []
        
        for pos in symbol_positions:
            try:
                position_side = 'LONG' if pos['side'] == 'Buy' else 'SHORT'
                position_size = float(pos['size'])
                
                logger.info(f"[API] 🔄 Закрываем позицию {position_side} размером {position_size} для {symbol}")
                
                close_result = current_exchange.close_position(
                    symbol=symbol,
                    size=position_size,
                    side=position_side,
                    order_type="Market"
                )
                
                if close_result and close_result.get('success'):
                    closed_positions.append({
                        'side': position_side,
                        'size': position_size,
                        'order_id': close_result.get('order_id')
                    })
                    logger.info(f"[API] ✅ Позиция {position_side} для {symbol} успешно закрыта")
                else:
                    error_msg = close_result.get('message', 'Unknown error') if close_result else 'No response'
                    errors.append(f"Позиция {position_side}: {error_msg}")
                    logger.error(f"[API] ❌ Ошибка закрытия позиции {position_side} для {symbol}: {error_msg}")
                    
            except Exception as e:
                error_msg = f"Позиция {pos['side']}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"[API] ❌ Исключение при закрытии позиции {pos['side']} для {symbol}: {str(e)}")
        
        # Обновляем данные бота, если он существует
        with bots_data_lock:
            if symbol in bots_data['bots']:
                bot_data = bots_data['bots'][symbol]
                if closed_positions:
                    bot_data['position_side'] = None
                    bot_data['unrealized_pnl'] = 0.0
                    bot_data['status'] = BOT_STATUS['IDLE']
                    logger.info(f"[API] 🔄 Обновлены данные бота {symbol} после закрытия позиций")
                
                # Обновляем глобальную статистику
                bots_data['global_stats']['bots_in_position'] = len([bot for bot in bots_data['bots'].values() if bot.get('position_side')])
        
        # Сохраняем состояние
        save_bots_state()
        update_bots_cache_data()
        
        if closed_positions:
            return jsonify({
                'success': True,
                'message': f'Закрыто {len(closed_positions)} позиций для {symbol}',
                'closed_positions': closed_positions,
                'errors': errors if errors else None
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Не удалось закрыть позиции для {symbol}',
                'errors': errors
            }), 500
            
    except Exception as e:
        logger.error(f"[ERROR] Ошибка закрытия позиций: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Словарь человекочитаемых названий параметров конфигурации
CONFIG_NAMES = {
    # Auto Bot Configuration
    'enabled': 'Auto Bot включен',
    'max_concurrent': 'Максимум одновременных ботов',
    'default_position_size': 'Размер позиции по умолчанию (USDT)',
    'rsi_long_threshold': 'RSI порог для LONG',
    'rsi_short_threshold': 'RSI порог для SHORT',
    'rsi_time_filter_enabled': 'RSI Time Filter (фильтр по времени)',
    'rsi_time_filter_candles': 'RSI Time Filter - количество свечей',
    'avoid_down_trend': 'Фильтр DOWN тренда для LONG',
    'avoid_up_trend': 'Фильтр UP тренда для SHORT',
    'trend_detection_enabled': 'Определение тренда включено',
    'min_candles_for_maturity': 'Минимум свечей для зрелости',
    'min_rsi_low': 'Минимальный RSI Low для зрелости',
    'max_rsi_high': 'Максимальный RSI High для зрелости',
    'tp_percent': 'Take Profit (%)',
    'sl_percent': 'Stop Loss (%)',
    'leverage': 'Плечо',
    
    # System Configuration
    'rsi_update_interval': 'Интервал обновления RSI (сек)',
    'auto_save_interval': 'Интервал автосохранения (сек)',
    'debug_mode': 'Режим отладки',
    'auto_refresh_ui': 'Автообновление UI',
    'refresh_interval': 'Интервал обновления UI (сек)',
    'position_sync_interval': 'Интервал синхронизации позиций (сек)',
    'inactive_bot_cleanup_interval': 'Интервал очистки неактивных ботов (сек)',
    'inactive_bot_timeout': 'Таймаут неактивного бота (сек)',
    'stop_loss_setup_interval': 'Интервал установки Stop Loss (сек)',
    'enhanced_rsi_enabled': 'Улучшенная система RSI',
    'enhanced_rsi_require_volume_confirmation': 'Подтверждение объемом',
    'enhanced_rsi_require_divergence_confirmation': 'Строгий режим (дивергенции)',
    'enhanced_rsi_use_stoch_rsi': 'Использовать Stochastic RSI',
}

def log_config_change(key, old_value, new_value, description=""):
    """Логирует изменение конфигурации только если значение изменилось"""
    if old_value != new_value:
        arrow = '→'
        # Используем понятное название из словаря или техническое название
        display_name = description or CONFIG_NAMES.get(key, key)
        # Используем print напрямую для ANSI кодов, чтобы обойти логгер
        print(f"\033[92m[CONFIG] ✓ {display_name}: {old_value} {arrow} {new_value}\033[0m")
        return True
    return False

@bots_app.route('/api/bots/system-config', methods=['GET', 'POST'])
def system_config():
    """Получить или обновить системные настройки"""
    # Константы теперь в SystemConfig
    try:
        if request.method == 'GET':
            return jsonify({
                'success': True,
                'config': {
                    'rsi_update_interval': SystemConfig.RSI_UPDATE_INTERVAL,
                    'auto_save_interval': SystemConfig.AUTO_SAVE_INTERVAL,
                    'debug_mode': SystemConfig.DEBUG_MODE,
                    'auto_refresh_ui': SystemConfig.AUTO_REFRESH_UI,
                    'refresh_interval': SystemConfig.UI_REFRESH_INTERVAL,
                    # Интервалы синхронизации и очистки
                    'position_sync_interval': SystemConfig.POSITION_SYNC_INTERVAL,
                    'inactive_bot_cleanup_interval': SystemConfig.INACTIVE_BOT_CLEANUP_INTERVAL,
                    'inactive_bot_timeout': SystemConfig.INACTIVE_BOT_TIMEOUT,
                    'stop_loss_setup_interval': SystemConfig.STOP_LOSS_SETUP_INTERVAL,
                    # Настройки улучшенного RSI
                    'enhanced_rsi_enabled': SystemConfig.ENHANCED_RSI_ENABLED,
                    'enhanced_rsi_require_volume_confirmation': SystemConfig.ENHANCED_RSI_REQUIRE_VOLUME_CONFIRMATION,
                    'enhanced_rsi_require_divergence_confirmation': SystemConfig.ENHANCED_RSI_REQUIRE_DIVERGENCE_CONFIRMATION,
                    'enhanced_rsi_use_stoch_rsi': SystemConfig.ENHANCED_RSI_USE_STOCH_RSI,
                    'rsi_extreme_zone_timeout': SystemConfig.RSI_EXTREME_ZONE_TIMEOUT,
                    'rsi_extreme_oversold': SystemConfig.RSI_EXTREME_OVERSOLD,
                    'rsi_extreme_overbought': SystemConfig.RSI_EXTREME_OVERBOUGHT,
                    'rsi_volume_confirmation_multiplier': SystemConfig.RSI_VOLUME_CONFIRMATION_MULTIPLIER,
                    'rsi_divergence_lookback': SystemConfig.RSI_DIVERGENCE_LOOKBACK,
                    # Параметры определения тренда
                    'trend_confirmation_bars': SystemConfig.TREND_CONFIRMATION_BARS,
                    'trend_min_confirmations': SystemConfig.TREND_MIN_CONFIRMATIONS,
                    'trend_require_slope': SystemConfig.TREND_REQUIRE_SLOPE,
                    'trend_require_price': SystemConfig.TREND_REQUIRE_PRICE,
                    'trend_require_candles': SystemConfig.TREND_REQUIRE_CANDLES
                }
            })
        
        elif request.method == 'POST':
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'error': 'No data provided'}), 400
            
            # Счетчик изменений
            changes_count = 0
            system_changes_count = 0
            
            # Обновляем настройки
            if 'rsi_update_interval' in data:
                old_value = SystemConfig.RSI_UPDATE_INTERVAL
                new_value = int(data['rsi_update_interval'])
                if log_config_change('rsi_update_interval', old_value, new_value):
                    SystemConfig.RSI_UPDATE_INTERVAL = new_value
                    changes_count += 1
                    # Обновляем интервал в SmartRSIManager если он активен
                    if 'smart_rsi_manager' in globals() and smart_rsi_manager:
                        smart_rsi_manager.update_monitoring_interval(SystemConfig.RSI_UPDATE_INTERVAL)
            
            if 'auto_save_interval' in data:
                old_value = SystemConfig.AUTO_SAVE_INTERVAL
                new_value = int(data['auto_save_interval'])
                if log_config_change('auto_save_interval', old_value, new_value):
                    SystemConfig.AUTO_SAVE_INTERVAL = new_value
                    changes_count += 1
            
            if 'debug_mode' in data:
                old_value = SystemConfig.DEBUG_MODE
                new_value = bool(data['debug_mode'])
                if log_config_change('debug_mode', old_value, new_value):
                    SystemConfig.DEBUG_MODE = new_value
                    changes_count += 1
            
            if 'auto_refresh_ui' in data:
                old_value = SystemConfig.AUTO_REFRESH_UI
                new_value = bool(data['auto_refresh_ui'])
                if log_config_change('auto_refresh_ui', old_value, new_value):
                    SystemConfig.AUTO_REFRESH_UI = new_value
                    changes_count += 1
            
            if 'refresh_interval' in data:
                old_value = SystemConfig.UI_REFRESH_INTERVAL
                new_value = int(data['refresh_interval'])
                if log_config_change('refresh_interval', old_value, new_value):
                    SystemConfig.UI_REFRESH_INTERVAL = new_value
                    changes_count += 1
            
            # Интервалы синхронизации и очистки
            if 'stop_loss_setup_interval' in data:
                old_value = SystemConfig.STOP_LOSS_SETUP_INTERVAL
                new_value = int(data['stop_loss_setup_interval'])
                if log_config_change('stop_loss_setup_interval', old_value, new_value):
                    SystemConfig.STOP_LOSS_SETUP_INTERVAL = new_value
                    system_changes_count += 1
            
            if 'position_sync_interval' in data:
                old_value = SystemConfig.POSITION_SYNC_INTERVAL
                new_value = int(data['position_sync_interval'])
                if log_config_change('position_sync_interval', old_value, new_value):
                    SystemConfig.POSITION_SYNC_INTERVAL = new_value
                    system_changes_count += 1
            
            if 'inactive_bot_cleanup_interval' in data:
                old_value = SystemConfig.INACTIVE_BOT_CLEANUP_INTERVAL
                new_value = int(data['inactive_bot_cleanup_interval'])
                if log_config_change('inactive_bot_cleanup_interval', old_value, new_value):
                    SystemConfig.INACTIVE_BOT_CLEANUP_INTERVAL = new_value
                    system_changes_count += 1
            
            if 'inactive_bot_timeout' in data:
                old_value = SystemConfig.INACTIVE_BOT_TIMEOUT
                new_value = int(data['inactive_bot_timeout'])
                if log_config_change('inactive_bot_timeout', old_value, new_value):
                    SystemConfig.INACTIVE_BOT_TIMEOUT = new_value
                    changes_count += 1
            
        # Настройки улучшенного RSI
        if 'enhanced_rsi_enabled' in data:
            old_value = SystemConfig.ENHANCED_RSI_ENABLED
            new_value = bool(data['enhanced_rsi_enabled'])
            log_config_change('enhanced_rsi_enabled', old_value, new_value)
            SystemConfig.ENHANCED_RSI_ENABLED = new_value
            system_changes_count += 1
        
        if 'enhanced_rsi_require_volume_confirmation' in data:
            old_value = SystemConfig.ENHANCED_RSI_REQUIRE_VOLUME_CONFIRMATION
            new_value = bool(data['enhanced_rsi_require_volume_confirmation'])
            log_config_change('enhanced_rsi_require_volume_confirmation', old_value, new_value)
            SystemConfig.ENHANCED_RSI_REQUIRE_VOLUME_CONFIRMATION = new_value
            system_changes_count += 1
        
        if 'enhanced_rsi_require_divergence_confirmation' in data:
            old_value = SystemConfig.ENHANCED_RSI_REQUIRE_DIVERGENCE_CONFIRMATION
            new_value = bool(data['enhanced_rsi_require_divergence_confirmation'])
            log_config_change('enhanced_rsi_require_divergence_confirmation', old_value, new_value)
            SystemConfig.ENHANCED_RSI_REQUIRE_DIVERGENCE_CONFIRMATION = new_value
            system_changes_count += 1
        
        if 'enhanced_rsi_use_stoch_rsi' in data:
            old_value = SystemConfig.ENHANCED_RSI_USE_STOCH_RSI
            new_value = bool(data['enhanced_rsi_use_stoch_rsi'])
            log_config_change('enhanced_rsi_use_stoch_rsi', old_value, new_value)
            SystemConfig.ENHANCED_RSI_USE_STOCH_RSI = new_value
            system_changes_count += 1
        
        if 'rsi_extreme_zone_timeout' in data:
            old_value = SystemConfig.RSI_EXTREME_ZONE_TIMEOUT
            new_value = int(data['rsi_extreme_zone_timeout'])
            log_config_change('rsi_extreme_zone_timeout', old_value, new_value)
            SystemConfig.RSI_EXTREME_ZONE_TIMEOUT = new_value
            system_changes_count += 1
        
        if 'rsi_extreme_oversold' in data:
            old_value = SystemConfig.RSI_EXTREME_OVERSOLD
            new_value = int(data['rsi_extreme_oversold'])
            log_config_change('rsi_extreme_oversold', old_value, new_value)
            SystemConfig.RSI_EXTREME_OVERSOLD = new_value
            system_changes_count += 1
        
        if 'rsi_extreme_overbought' in data:
            old_value = SystemConfig.RSI_EXTREME_OVERBOUGHT
            new_value = int(data['rsi_extreme_overbought'])
            log_config_change('rsi_extreme_overbought', old_value, new_value)
            SystemConfig.RSI_EXTREME_OVERBOUGHT = new_value
            system_changes_count += 1
        
        if 'rsi_volume_confirmation_multiplier' in data:
            old_value = SystemConfig.RSI_VOLUME_CONFIRMATION_MULTIPLIER
            new_value = float(data['rsi_volume_confirmation_multiplier'])
            log_config_change('rsi_volume_confirmation_multiplier', old_value, new_value)
            SystemConfig.RSI_VOLUME_CONFIRMATION_MULTIPLIER = new_value
            system_changes_count += 1
        
        if 'rsi_divergence_lookback' in data:
            old_value = SystemConfig.RSI_DIVERGENCE_LOOKBACK
            new_value = int(data['rsi_divergence_lookback'])
            log_config_change('rsi_divergence_lookback', old_value, new_value)
            SystemConfig.RSI_DIVERGENCE_LOOKBACK = new_value
            system_changes_count += 1
        
        # Параметры определения тренда
        if 'trend_confirmation_bars' in data:
            old_value = SystemConfig.TREND_CONFIRMATION_BARS
            new_value = int(data['trend_confirmation_bars'])
            log_config_change('trend_confirmation_bars', old_value, new_value)
            SystemConfig.TREND_CONFIRMATION_BARS = new_value
            system_changes_count += 1
        
        if 'trend_min_confirmations' in data:
            old_value = SystemConfig.TREND_MIN_CONFIRMATIONS
            new_value = int(data['trend_min_confirmations'])
            log_config_change('trend_min_confirmations', old_value, new_value)
            SystemConfig.TREND_MIN_CONFIRMATIONS = new_value
            system_changes_count += 1
        
        if 'trend_require_slope' in data:
            old_value = SystemConfig.TREND_REQUIRE_SLOPE
            new_value = bool(data['trend_require_slope'])
            log_config_change('trend_require_slope', old_value, new_value)
            SystemConfig.TREND_REQUIRE_SLOPE = new_value
            system_changes_count += 1
        
        if 'trend_require_price' in data:
            old_value = SystemConfig.TREND_REQUIRE_PRICE
            new_value = bool(data['trend_require_price'])
            log_config_change('trend_require_price', old_value, new_value)
            SystemConfig.TREND_REQUIRE_PRICE = new_value
            system_changes_count += 1
        
        if 'trend_require_candles' in data:
            old_value = SystemConfig.TREND_REQUIRE_CANDLES
            new_value = bool(data['trend_require_candles'])
            log_config_change('trend_require_candles', old_value, new_value)
            SystemConfig.TREND_REQUIRE_CANDLES = new_value
            system_changes_count += 1
        
            # КРИТИЧЕСКИ ВАЖНО: Сохраняем системные настройки в файл
            # Сначала загружаем существующие настройки, чтобы не потерять другие поля
            existing_config = {}
            if os.path.exists(SYSTEM_CONFIG_FILE):
                try:
                    with open(SYSTEM_CONFIG_FILE, 'r', encoding='utf-8') as f:
                        existing_config = json.load(f)
                except Exception as e:
                    logger.warning(f"[CONFIG] ⚠️ Не удалось загрузить существующую конфигурацию: {e}")
            
            # Обновляем только измененные поля
            system_config_data = existing_config.copy()
            system_config_data.update({
                'rsi_update_interval': SystemConfig.RSI_UPDATE_INTERVAL,
                'auto_save_interval': SystemConfig.AUTO_SAVE_INTERVAL,
                'debug_mode': SystemConfig.DEBUG_MODE,
                'auto_refresh_ui': SystemConfig.AUTO_REFRESH_UI,
                'refresh_interval': SystemConfig.UI_REFRESH_INTERVAL,
                # Интервалы синхронизации и очистки
                'position_sync_interval': SystemConfig.POSITION_SYNC_INTERVAL,
                'inactive_bot_cleanup_interval': SystemConfig.INACTIVE_BOT_CLEANUP_INTERVAL,
                'inactive_bot_timeout': SystemConfig.INACTIVE_BOT_TIMEOUT,
                'stop_loss_setup_interval': SystemConfig.STOP_LOSS_SETUP_INTERVAL,
                # Настройки улучшенного RSI
                'enhanced_rsi_enabled': SystemConfig.ENHANCED_RSI_ENABLED,
                'enhanced_rsi_require_volume_confirmation': SystemConfig.ENHANCED_RSI_REQUIRE_VOLUME_CONFIRMATION,
                'enhanced_rsi_require_divergence_confirmation': SystemConfig.ENHANCED_RSI_REQUIRE_DIVERGENCE_CONFIRMATION,
                'enhanced_rsi_use_stoch_rsi': SystemConfig.ENHANCED_RSI_USE_STOCH_RSI,
                'rsi_extreme_zone_timeout': SystemConfig.RSI_EXTREME_ZONE_TIMEOUT,
                'rsi_extreme_oversold': SystemConfig.RSI_EXTREME_OVERSOLD,
                'rsi_extreme_overbought': SystemConfig.RSI_EXTREME_OVERBOUGHT,
                'rsi_volume_confirmation_multiplier': SystemConfig.RSI_VOLUME_CONFIRMATION_MULTIPLIER,
                'rsi_divergence_lookback': SystemConfig.RSI_DIVERGENCE_LOOKBACK,
                # Параметры определения тренда
                'trend_confirmation_bars': SystemConfig.TREND_CONFIRMATION_BARS,
                'trend_min_confirmations': SystemConfig.TREND_MIN_CONFIRMATIONS,
                'trend_require_slope': SystemConfig.TREND_REQUIRE_SLOPE,
                'trend_require_price': SystemConfig.TREND_REQUIRE_PRICE,
                'trend_require_candles': SystemConfig.TREND_REQUIRE_CANDLES
            })
            
            saved_to_file = save_system_config(system_config_data)
            
            # Выводим итоговое сообщение
            if changes_count > 0:
                print(f"\033[92m[CONFIG] ✅ Изменено параметров: {changes_count}, конфигурация сохранена\033[0m")
            else:
                logger.info("[CONFIG] ℹ️  Изменений не обнаружено")
            
            # Выводим сообщение для system config
            if system_changes_count > 0:
                print(f"\033[92m[CONFIG] ✅ System config: изменено параметров: {system_changes_count}, конфигурация сохранена\033[0m")
            else:
                logger.info("[CONFIG] ℹ️  System config: изменений не обнаружено")
            
            if saved_to_file and (changes_count > 0 or system_changes_count > 0):
                # Перезагружаем конфигурацию, чтобы применить изменения
                load_system_config()
        
        return jsonify({
            'success': True,
                'message': 'Системные настройки обновлены и сохранены',
                'config': system_config_data,
                'saved_to_file': saved_to_file
        })
        
    except Exception as e:
        logger.error(f"[ERROR] Ошибка настройки системы: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/sync-positions', methods=['GET', 'POST'])
def sync_positions_manual():
    """Принудительная синхронизация позиций с биржей (работает с GET и POST)"""
    try:
        # ✅ Не логируем частые вызовы (только результаты)
        result = sync_positions_with_exchange()
        
        if result:
            return jsonify({
                'success': True,
                'message': 'Синхронизация позиций выполнена успешно',
                'synced': True
            })
        else:
            return jsonify({
                'success': True,
                'message': 'Синхронизация не потребовалась - все позиции актуальны',
                'synced': False
            })
            
    except Exception as e:
        logger.error(f"[MANUAL_SYNC] ❌ Ошибка принудительной синхронизации: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bots_app.route('/api/bots/cleanup-inactive', methods=['POST'])
def cleanup_inactive_manual():
    """Принудительная очистка неактивных ботов"""
    try:
        logger.info("[MANUAL_CLEANUP] 🧹 Запуск принудительной очистки неактивных ботов")
        result = cleanup_inactive_bots()
        
        if result:
            return jsonify({
                'success': True,
                'message': 'Очистка неактивных ботов выполнена успешно',
                'cleaned': True
            })
        else:
            return jsonify({
                'success': True,
                'message': 'Неактивных ботов для удаления не найдено',
                'cleaned': False
            })
            
    except Exception as e:
        logger.error(f"[MANUAL_CLEANUP] ❌ Ошибка принудительной очистки: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# УДАЛЕНО: API endpoint cleanup-mature
# Зрелость монеты необратима - нет смысла в API для удаления зрелых монет

@bots_app.route('/api/bots/mature-coins-list', methods=['GET'])
def get_mature_coins_list():
    """Получить список всех зрелых монет"""
    try:
        # ✅ ИСПРАВЛЕНИЕ: Читаем данные напрямую из файла, а не из памяти
        import json
        import os
        
        mature_coins_file = 'data/mature_coins.json'
        if os.path.exists(mature_coins_file):
            with open(mature_coins_file, 'r', encoding='utf-8') as f:
                mature_coins_data = json.load(f)
            mature_coins_list = list(mature_coins_data.keys())
        else:
            mature_coins_list = []
        
        return jsonify({
            'success': True,
            'mature_coins': mature_coins_list,
            'total_count': len(mature_coins_list)
        })
        
    except Exception as e:
        logger.error(f"[API_MATURE_LIST] ❌ Ошибка получения списка зрелых монет: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bots_app.route('/api/bots/remove-mature-coins', methods=['POST'])
def remove_mature_coins_api():
    """API для удаления конкретных монет из зрелых"""
    try:
        data = request.get_json()
        if not data or 'coins' not in data:
            return jsonify({
                'success': False,
                'error': 'Не указаны монеты для удаления'
            }), 400
        
        coins_to_remove = data['coins']
        if not isinstance(coins_to_remove, list):
            return jsonify({
                'success': False,
                'error': 'Параметр coins должен быть массивом'
            }), 400
        
        result = remove_mature_coins(coins_to_remove)
        
        if result['success']:
            return jsonify({
                'success': True,
                'message': result['message'],
                'removed_count': result['removed_count'],
                'removed_coins': result['removed_coins'],
                'not_found': result['not_found']
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500
            
    except Exception as e:
        logger.error(f"[API_REMOVE_MATURE] ❌ Ошибка API удаления монет: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bots_app.route('/api/bots/smart-rsi-status', methods=['GET'])
def get_smart_rsi_status():
    """Получить статус Smart RSI Manager"""
    try:
        global smart_rsi_manager
        if not smart_rsi_manager:
            return jsonify({
                'success': False,
                'error': 'Smart RSI Manager не инициализирован'
            }), 500
        
        status = smart_rsi_manager.get_status()
        return jsonify({
            'success': True,
            'status': status
        })
        
    except Exception as e:
        logger.error(f"[API] ❌ Ошибка получения статуса Smart RSI Manager: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bots_app.route('/api/bots/force-rsi-update', methods=['POST'])
def force_rsi_update():
    """Принудительно обновить RSI данные"""
    try:
        logger.info("[API] 🔄 Принудительное обновление RSI данных...")
        
        # Запускаем обновление RSI данных в отдельном потоке
        import threading
        def update_rsi():
            try:
                load_all_coins_rsi()
                logger.info("[API] ✅ RSI данные обновлены принудительно")
            except Exception as e:
                logger.error(f"[API] ❌ Ошибка принудительного обновления RSI: {e}")
        
        thread = threading.Thread(target=update_rsi)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Обновление RSI данных запущено'
        })
        
    except Exception as e:
        logger.error(f"[ERROR] Ошибка принудительного обновления RSI: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/test-exit-scam/<symbol>', methods=['GET'])
def test_exit_scam_endpoint(symbol):
    """Тестирует ExitScam фильтр для конкретной монеты"""
    try:
        test_exit_scam_filter(symbol)
        return jsonify({'success': True, 'message': f'Тест ExitScam фильтра для {symbol} выполнен'})
    except Exception as e:
        logger.error(f"[API] Ошибка тестирования ExitScam фильтра для {symbol}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Алиас для обратной совместимости
@bots_app.route('/api/bots/test-anti-pump/<symbol>', methods=['GET'])
def test_anti_pump_endpoint(symbol):
    """Алиас для test_exit_scam_endpoint (обратная совместимость)"""
    return test_exit_scam_endpoint(symbol)

@bots_app.route('/api/bots/test-rsi-time-filter/<symbol>', methods=['GET'])
def test_rsi_time_filter_endpoint(symbol):
    """Тестирует RSI временной фильтр для конкретной монеты"""
    try:
        test_rsi_time_filter(symbol)
        return jsonify({'success': True, 'message': f'Тест RSI временного фильтра для {symbol} выполнен'})
    except Exception as e:
        logger.error(f"[API] Ошибка тестирования RSI временного фильтра для {symbol}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/reload-modules', methods=['POST'])
def reload_modules_endpoint():
    """Перезагружает модули без перезапуска сервера"""
    try:
        import importlib
        import sys
        
        # Объявляем глобальные переменные в начале функции
        global exchange, system_initialized
        
        logger.info("[HOT_RELOAD] 🔄 Начинаем перезагрузку модулей...")
        
        # Сохраняем важные глобальные переменные
        saved_exchange = exchange
        saved_system_initialized = system_initialized
        
        # Список модулей для перезагрузки
        modules_to_reload = []
        
        # Находим все модули, которые содержат 'bot' в имени
        for module_name in sys.modules.keys():
            if 'bot' in module_name.lower() and not module_name.startswith('_'):
                modules_to_reload.append(module_name)
        
        reloaded_count = 0
        for module_name in modules_to_reload:
            try:
                if module_name in sys.modules:
                    importlib.reload(sys.modules[module_name])
                    reloaded_count += 1
                    logger.info(f"[HOT_RELOAD] Перезагружен модуль: {module_name}")
            except Exception as e:
                logger.warning(f"[HOT_RELOAD] Не удалось перезагрузить {module_name}: {e}")
        
        # Восстанавливаем важные переменные
        if saved_exchange:
            exchange = saved_exchange
            logger.info("[HOT_RELOAD] ✅ Восстановлена переменная exchange")
        
        if saved_system_initialized:
            system_initialized = saved_system_initialized
            logger.info("[HOT_RELOAD] ✅ Восстановлен флаг system_initialized")
        
        logger.info(f"[HOT_RELOAD] ✅ Перезагружено {reloaded_count} модулей")
        
        return jsonify({
            'success': True, 
            'message': f'Перезагружено {reloaded_count} модулей',
            'reloaded_modules': reloaded_count
        })
        
    except Exception as e:
        logger.error(f"[API] Ошибка перезагрузки модулей: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/refresh-rsi/<symbol>', methods=['POST'])
def refresh_rsi_for_coin(symbol):
    """Обновляет RSI данные для конкретной монеты (применяет новую логику)"""
    try:
        global coins_rsi_data
        
        logger.info(f"[HOT_RELOAD] 🔄 Обновление RSI данных для {symbol}...")
        
        # Проверяем биржу
        if not ensure_exchange_initialized():
            return jsonify({'success': False, 'error': 'Биржа не инициализирована'}), 500
        
        # Получаем новые данные монеты
        coin_data = get_coin_rsi_data(symbol, get_exchange())
        
        if coin_data:
            # Обновляем в глобальном кэше
            with rsi_data_lock:
                coins_rsi_data['coins'][symbol] = coin_data
            
            logger.info(f"[HOT_RELOAD] ✅ RSI данные для {symbol} обновлены")
            
            return jsonify({
                'success': True,
                'message': f'RSI данные для {symbol} обновлены',
                'coin_data': coin_data
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Не удалось получить RSI данные для {symbol}'
            }), 500
            
    except Exception as e:
        logger.error(f"[API] Ошибка обновления RSI для {symbol}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/refresh-rsi-all', methods=['POST'])
def refresh_rsi_for_all_coins():
    """Обновляет RSI данные для всех монет (применяет новую логику)"""
    try:
        global coins_rsi_data
        
        logger.info("[HOT_RELOAD] 🔄 Обновление RSI данных для всех монет...")
        
        # Проверяем биржу
        if not ensure_exchange_initialized():
            return jsonify({'success': False, 'error': 'Биржа не инициализирована'}), 500
        
        with rsi_data_lock:
            existing_symbols = list(coins_rsi_data['coins'].keys())
        
        updated_count = 0
        failed_count = 0
        
        current_exchange = get_exchange()
        
        for symbol in existing_symbols:
            try:
                coin_data = get_coin_rsi_data(symbol, current_exchange)
                if coin_data:
                    with rsi_data_lock:
                        coins_rsi_data['coins'][symbol] = coin_data
                    updated_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logger.warning(f"[HOT_RELOAD] Ошибка обновления {symbol}: {e}")
                failed_count += 1
        
        logger.info(f"[HOT_RELOAD] ✅ Обновлено {updated_count} монет, ошибок: {failed_count}")
        
        return jsonify({
            'success': True,
            'message': f'RSI данные обновлены для {updated_count} монет',
            'updated_count': updated_count,
            'failed_count': failed_count
        })
        
    except Exception as e:
        logger.error(f"[API] Ошибка обновления всех RSI данных: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/restart-service', methods=['POST'])
def restart_service_endpoint():
    """Перезапускает сервис ботов (только основные компоненты)"""
    try:
        logger.info("[HOT_RELOAD] 🔄 Перезапуск сервиса ботов...")
        
        # Перезагружаем глобальные переменные
        global exchange, coins_rsi_data, bots_data
        global system_initialized
        
        # Сбрасываем флаг инициализации
        system_initialized = False
        logger.info("[HOT_RELOAD] 🔄 Сброшен флаг инициализации")
        
        # Перезагружаем конфигурацию (БЕЗ принудительного выключения автобота)
        load_auto_bot_config(force_disable=False)
        load_system_config()
        logger.info("[HOT_RELOAD] 🔄 Перезагружена конфигурация")
        
        # Перезагружаем состояние ботов
        load_bots_state()
        logger.info("[HOT_RELOAD] 🔄 Перезагружено состояние ботов")
        
        # НЕ сбрасываем RSI данные! Используем существующий кэш
        # RSI данные обновятся автоматически по расписанию
        logger.info("[HOT_RELOAD] ⏭️  RSI данные сохранены (используется кэш)")
        
        # Восстанавливаем флаг инициализации
        system_initialized = True
        logger.info("[HOT_RELOAD] ✅ Флаг инициализации восстановлен")
        
        logger.info("[HOT_RELOAD] ✅ Сервис ботов перезапущен (RSI кэш сохранен)")
        
        return jsonify({
            'success': True, 
            'message': 'Сервис ботов перезапущен успешно'
        })
        
    except Exception as e:
        logger.error(f"[API] Ошибка перезапуска сервиса: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/process-trading-signals', methods=['POST'])
def process_trading_signals_endpoint():
    """Принудительно обработать торговые сигналы для всех ботов"""
    try:
        logger.info("[API] 🔄 Принудительная обработка торговых сигналов...")
        
        # Вызываем process_trading_signals_for_all_bots в основном процессе
        process_trading_signals_for_all_bots(exchange_obj=get_exchange())
        
        # Получаем количество активных ботов для отчета
        with bots_data_lock:
            active_bots = {symbol: bot for symbol, bot in bots_data['bots'].items() 
                          if bot['status'] not in [BOT_STATUS['IDLE'], BOT_STATUS['PAUSED']]}
        
        logger.info(f"[API] ✅ Обработка торговых сигналов завершена для {len(active_bots)} ботов")
        
        return jsonify({
            'success': True,
            'message': f'Обработка торговых сигналов завершена для {len(active_bots)} ботов',
            'active_bots_count': len(active_bots)
        })
        
    except Exception as e:
        logger.error(f"[API] ❌ Ошибка обработки торговых сигналов: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/reset-update-flag', methods=['POST'])
def reset_update_flag():
    """Принудительно сбросить флаг update_in_progress"""
    try:
        with rsi_data_lock:
            was_in_progress = coins_rsi_data['update_in_progress']
            coins_rsi_data['update_in_progress'] = False
            
        logger.info(f"[API] 🔄 Флаг update_in_progress сброшен (был: {was_in_progress})")
        return jsonify({
            'success': True,
            'message': 'Флаг update_in_progress сброшен',
            'was_in_progress': was_in_progress
        })
        
    except Exception as e:
        logger.error(f"[API] ❌ Ошибка сброса флага update_in_progress: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bots_app.route('/api/bots/test-stop', methods=['POST'])
def test_stop_bot():
    """Тестовый endpoint для остановки бота"""
    try:
        logger.info(f"[API] 🧪 Тестовый запрос остановки бота")
        logger.info(f"[API] 📥 Raw data: {request.get_data()}")
        logger.info(f"[API] 📥 Headers: {dict(request.headers)}")
        
        # Пробуем получить данные разными способами
        json_data = None
        form_data = None
        args_data = None
        
        try:
            json_data = request.get_json()
            logger.info(f"[API] 📊 JSON data: {json_data}")
        except Exception as e:
            logger.error(f"[API] ❌ JSON error: {e}")
        
        try:
            form_data = request.form.to_dict()
            logger.info(f"[API] 📊 Form data: {form_data}")
        except Exception as e:
            logger.error(f"[API] ❌ Form error: {e}")
        
        try:
            args_data = request.args.to_dict()
            logger.info(f"[API] 📊 Args data: {args_data}")
        except Exception as e:
            logger.error(f"[API] ❌ Args error: {e}")
        
        return jsonify({
            'success': True,
            'message': 'Тестовый запрос получен',
            'json_data': json_data,
            'form_data': form_data,
            'args_data': args_data
        })
        
    except Exception as e:
        logger.error(f"[API] ❌ Ошибка тестового запроса: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bots_app.route('/api/bots/activate-trading-rules', methods=['POST'])
def activate_trading_rules_manual():
    """Активация правил торговли для зрелых монет"""
    try:
        logger.info("[MANUAL_CLEANUP] 🎯 Запуск активации правил торговли")
        result = check_trading_rules_activation()
        
        if result:
            return jsonify({
                'success': True,
                'message': 'Правила торговли активированы успешно',
                'activated': True
            })
        else:
            return jsonify({
                'success': True,
                'message': 'Нет зрелых монет для активации правил торговли',
                'activated': False
            })
            
    except Exception as e:
        logger.error(f"[MANUAL_CLEANUP] ❌ Ошибка активации правил торговли: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bots_app.route('/api/bots/auto-bot', methods=['GET', 'POST'])
def auto_bot_config():
    """Получить или обновить конфигурацию Auto Bot"""
    try:
        # ✅ Логируем только POST (изменения), GET не логируем (слишком часто)
        if request.method == 'POST':
            logger.info(f"[CONFIG_API] 📝 Изменение конфигурации Auto Bot")
        
        if request.method == 'GET':
            with bots_data_lock:
                config = bots_data['auto_bot_config'].copy()
                return jsonify({
                    'success': True,
                    'config': config
                })
        
        elif request.method == 'POST':
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'error': 'No data provided'}), 400
            
            # Проверяем изменение критериев зрелости
            maturity_params_changed = False
            maturity_keys = ['min_candles_for_maturity', 'min_rsi_low', 'max_rsi_high']
            changes_count = 0
            
            # ✅ Сохраняем старую конфигурацию для сравнения
            with bots_data_lock:
                old_config = bots_data['auto_bot_config'].copy()
            
            # ✅ Сначала проверяем какие изменения будут
            for key in maturity_keys:
                if key in data and data[key] != old_config.get(key):
                    maturity_params_changed = True
                    logger.warning(f"[MATURITY] ⚠️ Изменен критерий зрелости: {key} ({old_config.get(key)} → {data[key]})")
            
            for key, value in data.items():
                if key in old_config:
                    old_value = old_config[key]
                    
                    # Проверяем реальное изменение
                    if old_value != value:
                        changes_count += 1
                        
                        # Используем log_config_change с названием из словаря
                        log_config_change(key, old_value, value)
            
            # ✅ Обновляем bots_data новыми значениями
            with bots_data_lock:
                for key, value in data.items():
                    if key in bots_data['auto_bot_config']:
                        bots_data['auto_bot_config'][key] = value
            
            # КРИТИЧЕСКИ ВАЖНО: Сохраняем конфигурацию в файл (с перезагрузкой модуля)
            save_result = save_auto_bot_config()
            
            # Выводим итоговое сообщение
            if changes_count > 0:
                print(f"\033[92m[CONFIG] ✅ Auto Bot: изменено параметров: {changes_count}, конфигурация сохранена и перезагружена\033[0m")
            else:
                logger.info("[CONFIG] ℹ️  Auto Bot: изменений не обнаружено")
            
            # ✅ АВТОМАТИЧЕСКАЯ ОЧИСТКА при изменении критериев зрелости
            if maturity_params_changed:
                logger.warning("=" * 80)
                logger.warning("[MATURITY] 🔄 КРИТЕРИИ ЗРЕЛОСТИ ИЗМЕНЕНЫ!")
                logger.warning("[MATURITY] 🗑️ Очистка файла зрелых монет...")
                logger.warning("=" * 80)
                
                try:
                    # Очищаем хранилище зрелых монет
                    clear_mature_coins_storage()
                    logger.info("[MATURITY] ✅ Файл зрелых монет очищен")
                    logger.info("[MATURITY] 🔄 Монеты будут перепроверены при следующей загрузке RSI")
                except Exception as e:
                    logger.error(f"[MATURITY] ❌ Ошибка очистки файла зрелых монет: {e}")
            
            # КРИТИЧЕСКИ ВАЖНО: При включении Auto Bot запускаем немедленную проверку
            # Показываем блок только если enabled реально изменился с False на True
            if 'enabled' in data and old_config.get('enabled') == False and data['enabled'] == True:
                # ✅ ЯРКИЙ ЛОГ ВКЛЮЧЕНИЯ (ЗЕЛЕНЫЙ)
                logger.info("=" * 80)
                print("\033[92m🟢 AUTO BOT ВКЛЮЧЕН! 🟢\033[0m")
                logger.info("=" * 80)
                logger.info("⚠️  ВНИМАНИЕ: Автобот будет автоматически создавать ботов!")
                logger.info(f"⚙️  Макс. одновременных ботов: {bots_data['auto_bot_config'].get('max_concurrent', 5)}")
                logger.info(f"📊 RSI пороги: LONG≤{bots_data['auto_bot_config'].get('rsi_long_threshold')}, SHORT≥{bots_data['auto_bot_config'].get('rsi_short_threshold')}")
                logger.info(f"⏰ RSI Time Filter: {'ON' if bots_data['auto_bot_config'].get('rsi_time_filter_enabled') else 'OFF'} ({bots_data['auto_bot_config'].get('rsi_time_filter_candles')} свечей)")
                logger.info("=" * 80)
                
                try:
                    # process_auto_bot_signals(exchange_obj=exchange)  # ОТКЛЮЧЕНО!
                    logger.info("[CONFIG] ✅ Немедленная проверка Auto Bot завершена")
                except Exception as e:
                    logger.error(f"[CONFIG] ❌ Ошибка немедленной проверки Auto Bot: {e}")
            
            # КРИТИЧЕСКИ ВАЖНО: При отключении Auto Bot НЕ удаляем ботов!
            # Показываем блок только если enabled реально изменился с True на False
            if 'enabled' in data and old_config.get('enabled') == True and data['enabled'] == False:
                # ✅ ЯРКИЙ ЛОГ ВЫКЛЮЧЕНИЯ (КРАСНЫЙ)
                logger.info("=" * 80)
                print("\033[91m🔴 AUTO BOT ВЫКЛЮЧЕН! 🔴\033[0m")
                logger.info("=" * 80)
                
                with bots_data_lock:
                    bots_count = len(bots_data['bots'])
                    bots_in_position = sum(1 for bot in bots_data['bots'].values() 
                                          if bot.get('status') in ['IN_POSITION_LONG', 'IN_POSITION_SHORT'])
                
                if bots_count > 0:
                    logger.info(f"💾 Сохранено {bots_count} ботов:")
                    logger.info(f"   📊 В позиции: {bots_in_position}")
                    logger.info(f"   🔄 Остальные: {bots_count - bots_in_position}")
                    logger.info("")
                    logger.info("✅ ЧТО БУДЕТ ДАЛЬШЕ:")
                    logger.info("   🔄 Существующие боты продолжат работать")
                    logger.info("   🛡️ Защитные механизмы активны (стоп-лосс, RSI выход)")
                    logger.info("   ❌ Новые боты НЕ будут создаваться")
                    logger.info("   🗑️ Для удаления используйте кнопку 'Удалить всё'")
                else:
                    logger.info("ℹ️  Нет активных ботов")
                
                logger.info("=" * 80)
                logger.info("✅ АВТОБОТ ОСТАНОВЛЕН (боты сохранены)")
                logger.info("=" * 80)
        
        return jsonify({
            'success': True,
            'message': 'Конфигурация Auto Bot обновлена и сохранена',
            'config': bots_data['auto_bot_config'].copy(),
            'saved_to_file': save_result
        })
        
    except Exception as e:
        logger.error(f"[ERROR] Ошибка конфигурации Auto Bot: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/auto-bot/restore-defaults', methods=['POST'])
def restore_auto_bot_defaults():
    """Восстанавливает дефолтную конфигурацию Auto Bot"""
    try:
        logger.info("[API] 🔄 Запрос на восстановление дефолтной конфигурации Auto Bot")
        
        # Восстанавливаем дефолтные настройки
        result = restore_default_config()
        
        if result:
            with bots_data_lock:
                current_config = bots_data['auto_bot_config'].copy()
            
            return jsonify({
                'success': True,
                'message': 'Дефолтная конфигурация Auto Bot восстановлена',
                'config': current_config,
                'restored_to_defaults': True
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Ошибка восстановления дефолтной конфигурации'
            }), 500
            
    except Exception as e:
        logger.error(f"[ERROR] Ошибка восстановления дефолтной конфигурации: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/debug-init', methods=['GET'])
def debug_init_status():
    """Отладочный эндпоинт для проверки инициализации"""
    try:
        return jsonify({
            'success': True,
            'init_bot_service_called': 'init_bot_service' in globals(),
            'smart_rsi_manager_exists': smart_rsi_manager is not None,
            'exchange_exists': exchange is not None,
            'bots_data_keys': list(bots_data.keys()) if 'bots_data' in globals() else 'not_initialized'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@bots_app.route('/api/bots/process-state', methods=['GET'])
def get_process_state():
    """Получить состояние всех процессов системы"""
    try:
        return jsonify({
            'success': True,
            'process_state': process_state.copy(),
            'system_info': {
                'smart_rsi_manager_running': smart_rsi_manager is not None and not smart_rsi_manager.shutdown_flag.is_set(),
                'exchange_initialized': exchange is not None,
                'total_bots': len(bots_data['bots']),
                'auto_bot_enabled': bots_data['auto_bot_config']['enabled'],
                'mature_coins_storage_size': len(mature_coins_storage),
                'optimal_ema_count': len(optimal_ema_data)
            }
                })
        
    except Exception as e:
        logger.error(f"[ERROR] Ошибка получения состояния процессов: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/mature-coins', methods=['GET'])
def get_mature_coins():
    """Получение списка зрелых монет из постоянного хранилища"""
    try:
        return jsonify({
            'success': True,
            'data': {
                'mature_coins': list(mature_coins_storage.keys()),
                'count': len(mature_coins_storage),
                'storage_details': mature_coins_storage
            }
        })
    except Exception as e:
        logger.error(f"[API] Ошибка получения зрелых монет: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/mature-coins/reload', methods=['POST'])
def reload_mature_coins():
    """Перезагрузить список зрелых монет из файла"""
    try:
        load_mature_coins_storage()
        logger.info(f"[MATURITY_STORAGE] Перезагружено {len(mature_coins_storage)} зрелых монет")
        return jsonify({
            'success': True,
            'message': f'Перезагружено {len(mature_coins_storage)} зрелых монет',
            'data': {
                'mature_coins': list(mature_coins_storage.keys()),
                'count': len(mature_coins_storage)
            }
        })
    except Exception as e:
        logger.error(f"[ERROR] Ошибка перезагрузки зрелых монет: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/mature-coins/<symbol>', methods=['DELETE'])
def remove_mature_coin(symbol):
    """Удаление монеты из постоянного хранилища зрелых монет"""
    try:
        if symbol in mature_coins_storage:
            remove_mature_coin_from_storage(symbol)
            return jsonify({
                'success': True,
                'message': f'Монета {symbol} удалена из постоянного хранилища зрелых монет'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Монета {symbol} не найдена в постоянном хранилище зрелых монет'
            }), 404
    except Exception as e:
        logger.error(f"[API] Ошибка удаления монеты {symbol} из хранилища: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/mature-coins/clear', methods=['POST'])
def clear_mature_coins_storage():
    """Очистка всего постоянного хранилища зрелых монет"""
    try:
        global mature_coins_storage
        mature_coins_storage = {}
        save_mature_coins_storage()
        logger.info("[API] Постоянное хранилище зрелых монет очищено")
        return jsonify({
            'success': True,
            'message': 'Постоянное хранилище зрелых монет очищено'
        })
    except Exception as e:
        logger.error(f"[API] Ошибка очистки хранилища зрелых монет: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/optimal-ema', methods=['GET'])
def get_optimal_ema():
    """Получение списка оптимальных EMA из хранилища"""
    try:
        return jsonify({
            'success': True,
            'data': {
                'optimal_ema': optimal_ema_data,
                'count': len(optimal_ema_data)
            }
        })
    except Exception as e:
        logger.error(f"[API] Ошибка получения оптимальных EMA: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/optimal-ema/<symbol>', methods=['GET'])
def get_optimal_ema_for_symbol(symbol):
    """Получение оптимальных EMA для конкретной монеты"""
    try:
        if symbol in optimal_ema_data:
            return jsonify({
                'success': True,
                'data': optimal_ema_data[symbol]
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Оптимальные EMA для {symbol} не найдены'
            }), 404
    except Exception as e:
        logger.error(f"[API] Ошибка получения оптимальных EMA для {symbol}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/optimal-ema/<symbol>/rescan', methods=['POST'])
def rescan_optimal_ema(symbol):
    """Принудительное пересканирование оптимальных EMA для монеты"""
    try:
        # Здесь можно добавить логику для запуска пересканирования
        # Пока просто возвращаем сообщение
        return jsonify({
            'success': True,
            'message': f'Запущено пересканирование оптимальных EMA для {symbol}. Используйте скрипт scripts/sync/optimal_ema.py для выполнения.'
        })
    except Exception as e:
        logger.error(f"[API] Ошибка пересканирования EMA для {symbol}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/optimal-ema-worker/status', methods=['GET'])
def get_optimal_ema_worker_status():
    """Получает статус воркера оптимальных EMA"""
    try:
        from bot_engine.optimal_ema_worker import get_optimal_ema_worker
        
        worker = get_optimal_ema_worker()
        if worker:
            status = worker.get_status()
            return jsonify({
                'success': True,
                'data': status
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Воркер оптимальных EMA не инициализирован'
            }), 404
    except Exception as e:
        logger.error(f"[API] Ошибка получения статуса воркера: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/optimal-ema-worker/force-update', methods=['POST'])
def force_optimal_ema_update():
    """Принудительно запускает обновление оптимальных EMA"""
    try:
        from bot_engine.optimal_ema_worker import get_optimal_ema_worker
        
        worker = get_optimal_ema_worker()
        if worker:
            success = worker.force_update()
            if success:
                return jsonify({
                    'success': True,
                    'message': 'Принудительное обновление оптимальных EMA запущено'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Обновление уже выполняется'
                }), 409
        else:
            return jsonify({
                'success': False,
                'error': 'Воркер оптимальных EMA не инициализирован'
            }), 404
    except Exception as e:
        logger.error(f"[API] Ошибка принудительного обновления: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/optimal-ema-worker/set-interval', methods=['POST'])
def set_optimal_ema_interval():
    """Устанавливает интервал обновления воркера оптимальных EMA"""
    try:
        from bot_engine.optimal_ema_worker import get_optimal_ema_worker
        
        data = request.get_json()
        if not data or 'interval' not in data:
            return jsonify({
                'success': False,
                'error': 'Не указан интервал обновления'
            }), 400
        
        interval = int(data['interval'])
        if interval < 300:  # Минимум 5 минут
            return jsonify({
                'success': False,
                'error': 'Интервал не может быть меньше 300 секунд (5 минут)'
            }), 400
        
        worker = get_optimal_ema_worker()
        if worker:
            success = worker.set_update_interval(interval)
            if success:
                return jsonify({
                    'success': True,
                    'message': f'Интервал обновления изменен на {interval} секунд'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Не удалось изменить интервал'
                })
        else:
            return jsonify({
                'success': False,
                'error': 'Воркер оптимальных EMA не инициализирован'
            }), 404
    except Exception as e:
        logger.error(f"[API] Ошибка изменения интервала: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/default-config', methods=['GET'])
def get_default_config():
    """Получить дефолтную конфигурацию Auto Bot"""
    try:
        default_config = load_default_config()
        
        return jsonify({
            'success': True,
            'default_config': default_config,
            'message': 'Дефолтная конфигурация загружена'
        })
        
    except Exception as e:
        logger.error(f"[ERROR] Ошибка загрузки дефолтной конфигурации: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/auto-bot/test-signals', methods=['POST'])
def test_auto_bot_signals():
    """Тестовый эндпоинт для принудительной обработки Auto Bot сигналов - УДАЛЕНО!"""
    return jsonify({'success': False, 'message': 'Auto Bot отключен!'})
    try:
        logger.info("[TEST] 🧪 Принудительная обработка Auto Bot сигналов...")
        
        # Принудительно вызываем обработку сигналов
        # process_auto_bot_signals(exchange_obj=exchange)  # ОТКЛЮЧЕНО!
        
        # Получаем статистику
        with bots_data_lock:
            auto_bot_enabled = bots_data['auto_bot_config']['enabled']
            total_bots = len(bots_data['bots'])
            max_concurrent = bots_data['auto_bot_config']['max_concurrent']
            
        with rsi_data_lock:
            signals = [c for c in coins_rsi_data['coins'].values() 
                      if c['signal'] in ['ENTER_LONG', 'ENTER_SHORT']]
        
        return jsonify({
            'success': True,
            'message': 'Auto Bot сигналы обработаны принудительно',
            'stats': {
                'auto_bot_enabled': auto_bot_enabled,
                'available_signals': len(signals),
                'current_bots': total_bots,
                'max_concurrent': max_concurrent,
                'signals_details': signals[:5]  # Первые 5 для примера
            }
        })
        
    except Exception as e:
        logger.error(f"[ERROR] Ошибка тестирования Auto Bot: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@bots_app.errorhandler(500)
def internal_error(error):
    logger.error(f"[ERROR] Внутренняя ошибка сервера: {str(error)}")
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

def signal_handler(signum, frame):
    """Обработчик сигналов завершения с принудительным завершением"""
    global graceful_shutdown
    print(f"\n[SHUTDOWN] 🛑 Получен сигнал {signum}, начинаем graceful shutdown...")
    logger.info(f"[SHUTDOWN] 🛑 Получен сигнал {signum}, начинаем graceful shutdown...")
    graceful_shutdown = True
    shutdown_flag.set()
    
    # Запускаем принудительное завершение через таймер
    def force_exit():
        time.sleep(2.0)  # Даём 2 секунды на graceful shutdown
        print("[SHUTDOWN] ⏱️ Таймаут graceful shutdown, принудительное завершение...")
        logger.info("[SHUTDOWN] ⏱️ Таймаут graceful shutdown, принудительное завершение...")
        os._exit(0)
    
    force_exit_thread = threading.Thread(target=force_exit, daemon=True)
    force_exit_thread.start()
    
    # Пытаемся выполнить graceful shutdown
    try:
        cleanup_bot_service()
        print("[SHUTDOWN] ✅ Graceful shutdown завершен")
        logger.info("[SHUTDOWN] ✅ Graceful shutdown завершен")
        sys.exit(0)
    except Exception as e:
        print(f"[SHUTDOWN] ⚠️ Ошибка при graceful shutdown: {e}")
        logger.error(f"[SHUTDOWN] ⚠️ Ошибка при graceful shutdown: {e}")
        os._exit(1)

@bots_app.route('/api/system/reload-modules', methods=['POST'])
def reload_modules():
    """Умная горячая перезагрузка модулей с поддержкой Flask"""
    try:
        import importlib
        import sys
        import os
        import threading
        import time
        
        # Определяем модули для перезагрузки в порядке зависимостей
        modules_to_reload = [
            'bot_engine.bot_config',
            'bot_engine.indicators',
            'bots_modules.maturity',
            'bots_modules.sync_and_cache',
            'bots_modules.calculations',
            'bots_modules.filters',
        ]
        
        # Модули которые требуют перезапуска Flask сервера
        flask_restart_modules = [
            'bots_modules.api_endpoints',
            'bots_modules.init_functions',
        ]
        
        reloaded = []
        failed = []
        flask_restart_required = False
        
        logger.info("[HOT_RELOAD] 🔄 Начинаем умную горячую перезагрузку...")
        
        # Этап 1: Перезагружаем безопасные модули
        for module_name in modules_to_reload:
            try:
                if module_name in sys.modules:
                    logger.info(f"[HOT_RELOAD] 🔄 Перезагрузка модуля {module_name}...")
                    module = sys.modules[module_name]
                    importlib.reload(module)
                    reloaded.append(module_name)
                    logger.info(f"[HOT_RELOAD] ✅ Модуль {module_name} перезагружен")
                else:
                    logger.warning(f"[HOT_RELOAD] ⚠️ Модуль {module_name} не был загружен")
            except Exception as e:
                logger.error(f"[HOT_RELOAD] ❌ Ошибка перезагрузки {module_name}: {e}")
                failed.append({'module': module_name, 'error': str(e)})
        
        # Этап 1.5: Проверяем состояние Flask после перезагрузки
        try:
            logger.info("[HOT_RELOAD] 🔍 Проверка состояния Flask после перезагрузки...")
            # Простая проверка что Flask все еще работает
            if hasattr(request, 'method') and hasattr(request, 'get_json'):
                logger.info("[HOT_RELOAD] ✅ Flask состояние корректно")
            else:
                logger.warning("[HOT_RELOAD] ⚠️ Flask состояние может быть нарушено")
        except Exception as e:
            logger.error(f"[HOT_RELOAD] ❌ Ошибка проверки Flask: {e}")
        
        # Этап 2: Проверяем нужен ли перезапуск Flask сервера
        try:
            request_data = request.get_json() or {}
            force_flask_restart = request_data.get('force_flask_restart', False)
            logger.info(f"[HOT_RELOAD] 📋 Данные запроса: {request_data}")
        except Exception as e:
            logger.error(f"[HOT_RELOAD] ❌ Ошибка парсинга JSON запроса: {e}")
            request_data = {}
            force_flask_restart = False
        
        if force_flask_restart or any(module in sys.modules for module in flask_restart_modules):
            flask_restart_required = True
            logger.info("[HOT_RELOAD] 🔄 Требуется перезапуск Flask сервера...")
            
            # Сохраняем состояние перед перезапуском
            save_bots_state()
            logger.info("[HOT_RELOAD] 💾 Состояние ботов сохранено")
            
            # Запускаем перезапуск сервера в отдельном потоке
            def restart_server():
                time.sleep(2)  # Даем время для ответа клиенту
                logger.info("[HOT_RELOAD] 🔄 Перезапуск Flask сервера...")
                os._exit(42)  # Специальный код для перезапуска
            
            restart_thread = threading.Thread(target=restart_server, daemon=True)
            restart_thread.start()
        
        # Этап 3: Перезагружаем конфигурацию
        try:
            from bots_modules.imports_and_globals import load_auto_bot_config
            load_auto_bot_config()
            logger.info("[HOT_RELOAD] ✅ Конфигурация Auto Bot перезагружена")
        except Exception as e:
            logger.error(f"[HOT_RELOAD] ❌ Ошибка перезагрузки конфигурации: {e}")
        
        # Формируем ответ
        response_data = {
            'success': True,
            'reloaded': reloaded,
            'failed': failed,
            'flask_restart_required': flask_restart_required,
            'message': f'Перезагружено {len(reloaded)} модулей'
        }
        
        if flask_restart_required:
            response_data['message'] += '. Сервер будет перезапущен через 2 секунды...'
            response_data['restart_in_seconds'] = 2
        
        logger.info(f"[HOT_RELOAD] ✅ Горячая перезагрузка завершена: {len(reloaded)} модулей")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"[HOT_RELOAD] ❌ Общая ошибка горячей перезагрузки: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bots_app.route('/api/bots/delisted-coins', methods=['GET'])
def get_delisted_coins_api():
    """API для получения списка делистинговых монет"""
    try:
        from bots_modules.sync_and_cache import load_delisted_coins
        
        delisted_data = load_delisted_coins()
        
        return jsonify({
            'success': True,
            'delisted_coins': delisted_data.get('delisted_coins', {}),
            'last_scan': delisted_data.get('last_scan'),
            'scan_enabled': delisted_data.get('scan_enabled', True),
            'total_count': len(delisted_data.get('delisted_coins', {}))
        })
        
    except Exception as e:
        logger.error(f"[API_DELISTED_COINS] ❌ Ошибка получения делистинговых монет: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bots_app.route('/api/bots/delisted-coins/force-scan', methods=['POST'])
def force_delisting_scan_api():
    """API для принудительного сканирования делистинговых монет"""
    try:
        from bots_modules.sync_and_cache import scan_all_coins_for_delisting
        
        logger.info("[API_DELISTING_SCAN] 🔍 Принудительное сканирование делистинговых монет...")
        scan_all_coins_for_delisting()
        
        # Получаем обновленные данные
        from bots_modules.sync_and_cache import load_delisted_coins
        delisted_data = load_delisted_coins()
        
        return jsonify({
            'success': True,
            'message': 'Сканирование делистинговых монет завершено',
            'delisted_coins': delisted_data.get('delisted_coins', {}),
            'last_scan': delisted_data.get('last_scan'),
            'total_count': len(delisted_data.get('delisted_coins', {}))
        })
        
    except Exception as e:
        logger.error(f"[API_DELISTING_SCAN] ❌ Ошибка принудительного сканирования: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def cleanup_bot_service():
    """Очистка ресурсов при завершении сервиса"""
    global smart_rsi_manager, system_initialized
    
    # КРИТИЧЕСКИ ВАЖНО: Сбрасываем флаг, чтобы остановить торговлю
    system_initialized = False
    logger.info("[CLEANUP] 🛑 Флаг system_initialized сброшен - торговля остановлена")
    
    try:
        logger.info("[CLEANUP] 🧹 Очистка ресурсов сервиса ботов...")
        
        # Останавливаем асинхронный процессор
        stop_async_processor()
        
        # Останавливаем умный менеджер RSI
        if smart_rsi_manager:
            logger.info("[CLEANUP] 🛑 Остановка Smart RSI Manager...")
            smart_rsi_manager.stop()
            smart_rsi_manager = None
        
        # Останавливаем воркер оптимальных EMA
        try:
            from bot_engine.optimal_ema_worker import stop_optimal_ema_worker
            stop_optimal_ema_worker()
            logger.info("[CLEANUP] 🛑 Остановка воркера оптимальных EMA...")
        except Exception as e:
            logger.error(f"[CLEANUP] Ошибка остановки воркера оптимальных EMA: {e}")
        
        # Сохраняем все важные данные
        logger.info("[CLEANUP] 💾 Финальное сохранение всех данных...")
        
        # 1. Сохраняем состояние ботов
        logger.info("[CLEANUP] 📊 Сохранение состояния ботов...")
        save_bots_state()
        
        # 2. Сохраняем конфигурацию автобота
        logger.info("[CLEANUP] ⚙️ Сохранение конфигурации автобота...")
        save_auto_bot_config()
        
        # 3. Сохраняем системную конфигурацию
        logger.info("[CLEANUP] 🔧 Сохранение системной конфигурации...")
        system_config_data = {
            'bot_status_update_interval': SystemConfig.BOT_STATUS_UPDATE_INTERVAL,
            'position_sync_interval': SystemConfig.POSITION_SYNC_INTERVAL,
            'inactive_bot_cleanup_interval': SystemConfig.INACTIVE_BOT_CLEANUP_INTERVAL,
            'inactive_bot_timeout': SystemConfig.INACTIVE_BOT_TIMEOUT,
            'stop_loss_setup_interval': SystemConfig.STOP_LOSS_SETUP_INTERVAL
        }
        save_system_config(system_config_data)
        
        # 4. Сохраняем кэш RSI данных
        logger.info("[CLEANUP] 📈 Сохранение кэша RSI данных...")
        save_rsi_cache()
        
        # 5. Сохраняем состояние процессов
        logger.info("[CLEANUP] 🔄 Сохранение состояния процессов...")
        save_process_state()
        
        # 6. Сохраняем данные о зрелости монет
        logger.info("[CLEANUP] 🪙 Сохранение данных о зрелости монет...")
        save_mature_coins_storage()
        
        # 7. Сохраняем оптимальные EMA периоды
        logger.info("[CLEANUP] 📊 Сохранение оптимальных EMA периодов...")
        save_optimal_ema_periods()
        
        logger.info("[CLEANUP] ✅ Все данные сохранены, очистка завершена")
        
    except Exception as e:
        logger.error(f"[CLEANUP] ❌ Ошибка при очистке: {e}")
        import traceback
        logger.error(f"[CLEANUP] Traceback: {traceback.format_exc()}")

def run_bots_service():
    """Запуск сервиса ботов"""
    print("[RUN_SERVICE] 🚀 Запуск run_bots_service...")
    try:
        # Создаем директорию для логов
        os.makedirs('logs', exist_ok=True)
        print("[RUN_SERVICE] 📁 Директория логов создана")
        
        # Очищаем старые логи при запуске
        log_files = ['logs/bots.log', 'logs/app.log', 'logs/error.log']
        for log_file in log_files:
            if os.path.exists(log_file):
                file_size = os.path.getsize(log_file)
                if file_size > 2 * 1024 * 1024:  # 2MB
                    print(f"[RUN_SERVICE] 🗑️ Очищаем большой лог файл: {log_file} ({file_size / 1024 / 1024:.1f}MB)")
                    with open(log_file, 'w', encoding='utf-8') as f:
                        f.write(f"# Лог файл очищен при запуске - {datetime.now().isoformat()}\n")
                else:
                    print(f"[RUN_SERVICE] 📝 Лог файл в порядке: {log_file} ({file_size / 1024:.1f}KB)")
        
        # Временно отключаем обработчики сигналов до полной инициализации
        # signal.signal(signal.SIGINT, signal_handler)
        # signal.signal(signal.SIGTERM, signal_handler)
        
        # Регистрируем функцию очистки для автоматического вызова при завершении
        atexit.register(cleanup_bot_service)
        
        logger.info(f"🌐 Запуск Flask сервера для ботов на {SystemConfig.BOTS_SERVICE_HOST}:{SystemConfig.BOTS_SERVICE_PORT}...")
        logger.info("📋 Этот сервис предоставляет API для торговых ботов")
        
        # Регистрируем AI endpoints
        try:
            from bot_engine.api.endpoints_ai import register_ai_endpoints
            register_ai_endpoints(bots_app)
            logger.info("✅ AI endpoints зарегистрированы")
        except ImportError as e:
            logger.warning(f"⚠️ AI endpoints недоступны: {e}")
        except Exception as e:
            logger.error(f"❌ Ошибка регистрации AI endpoints: {e}")
        
        # Запускаем Flask сервер в отдельном потоке СРАЗУ
        def run_flask_server():
            try:
                logger.info("🚀 Запуск Flask сервера в отдельном потоке...")
                bots_app.run(
                    debug=SystemConfig.DEBUG_MODE,
                    host=SystemConfig.BOTS_SERVICE_HOST,
                    port=SystemConfig.BOTS_SERVICE_PORT,
                    use_reloader=False,
                    threaded=True
                )
            except Exception as e:
                logger.error(f"❌ Ошибка запуска Flask сервера: {e}")
        
        flask_thread = threading.Thread(target=run_flask_server, daemon=True)
        flask_thread.start()
        
        # Ждем, пока Flask сервер запустится
        import time
        time.sleep(3)
        logger.info("✅ Flask сервер запущен в фоновом режиме")
        
        # Теперь инициализируем сервис в отдельном потоке
        def init_service_async():
            try:
                logger.info("[INIT_THREAD] 🚀 Запуск инициализации в отдельном потоке...")
                result = init_bot_service()
                if result:
                    logger.info("[INIT_THREAD] ✅ Инициализация завершена успешно")
                    return True
                else:
                    logger.error("[INIT_THREAD] ❌ Инициализация завершена с ошибкой")
                    return False
            except Exception as e:
                logger.error(f"[INIT_THREAD] ❌ Исключение при инициализации: {e}")
                import traceback
                logger.error(f"[INIT_THREAD] Traceback: {traceback.format_exc()}")
                return False
        
        service_thread = threading.Thread(target=init_service_async, daemon=True)
        service_thread.start()
        
        # Ждем завершения инициализации сервиса
        logger.info("⏳ Ожидание инициализации сервиса ботов...")
        service_thread.join(timeout=30)  # Ждем максимум 30 секунд
        
        if service_thread.is_alive():
            logger.warning("⚠️ Инициализация сервиса ботов занимает больше времени, продолжаем...")
        else:
            logger.info("✅ Сервис ботов инициализирован")
        
        # ДОПОЛНИТЕЛЬНО: Ждем установки флага system_initialized
        logger.info("⏳ Ожидание установки флага system_initialized...")
        max_wait_time = 60  # Максимум 60 секунд
        wait_start = time.time()
        
        while not system_initialized and (time.time() - wait_start) < max_wait_time:
            time.sleep(1)
            if int(time.time() - wait_start) % 10 == 0:  # Каждые 10 секунд
                logger.info(f"⏳ Ожидание system_initialized... ({int(time.time() - wait_start)}s)")
        
        if system_initialized:
            logger.info("✅ Флаг system_initialized установлен - система готова к работе")
        else:
            logger.error("❌ Флаг system_initialized не установлен за {max_wait_time}s - возможны проблемы")
        
        # Теперь настраиваем обработчики сигналов после полной инициализации
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        logger.info("✅ Обработчики сигналов настроены")
        
        # Запускаем воркер для обновления оптимальных EMA
        try:
            from bot_engine.optimal_ema_worker import start_optimal_ema_worker
            optimal_ema_worker = start_optimal_ema_worker(update_interval=21600)  # 6 часов
            if optimal_ema_worker:
                logger.info("✅ Воркер оптимальных EMA запущен")
            else:
                logger.warning("⚠️ Не удалось запустить воркер оптимальных EMA")
        except Exception as e:
            logger.error(f"❌ Ошибка запуска воркера оптимальных EMA: {e}")
        
        # Основной поток
        logger.info("🔄 Сервис ботов запущен и работает...")
        last_bot_processing = 0
        bot_processing_interval = 30  # Обрабатываем ботов каждые 30 секунд
        
        while True:
            try:
                current_time = time.time()
                
                # Обрабатываем ботов каждые 30 секунд
                if current_time - last_bot_processing >= bot_processing_interval:
                    logger.info("[MAIN_LOOP] 🤖 Обработка ботов...")
                    process_trading_signals_for_all_bots(exchange_obj=get_exchange())
                    last_bot_processing = current_time
                    logger.info("[MAIN_LOOP] ✅ Обработка ботов завершена")
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"[MAIN_LOOP] ❌ Ошибка в основном цикле: {e}")
                time.sleep(5)  # Ждем 5 секунд при ошибке
        
    except KeyboardInterrupt:
        logger.info("[STOP] Получен сигнал прерывания...")
        cleanup_bot_service()
        os._exit(0)
    except Exception as e:
        logger.error(f"[ERROR] Ошибка запуска сервиса ботов: {str(e)}")
        cleanup_bot_service()
        os._exit(1)
    finally:
        cleanup_bot_service()

@bots_app.route('/api/bots/active-detailed', methods=['GET'])
def get_active_bots_detailed():
    """Получает детальную информацию о активных ботах для мониторинга"""
    try:
        with bots_data_lock:
            active_bots = []
            for symbol, bot_data in bots_data['bots'].items():
                if bot_data.get('status') in ['in_position_long', 'in_position_short']:
                    # Получаем текущую цену из RSI данных
                    current_price = None
                    with rsi_data_lock:
                        coin_data = coins_rsi_data['coins'].get(symbol)
                        if coin_data:
                            current_price = coin_data.get('price')
                    
                    # Определяем направление позиции
                    position_side = None
                    if bot_data.get('status') in ['in_position_long']:
                        position_side = 'Long'
                    elif bot_data.get('status') in ['in_position_short']:
                        position_side = 'Short'
                    
                    # Получаем настройки бота
                    config = bot_data.get('config', {})
                    
                    # Рассчитываем потенциальный убыток по стоп-лоссу
                    stop_loss_pnl = 0
                    if current_price and position_side and bot_data.get('entry_price'):
                        entry_price = bot_data.get('entry_price')
                        max_loss_percent = config.get('max_loss_percent', 15.0)
                        
                        if position_side == 'Long':
                            stop_loss_price = entry_price * (1 - max_loss_percent / 100)
                            stop_loss_pnl = (stop_loss_price - entry_price) / entry_price * 100
                        else:  # Short
                            stop_loss_price = entry_price * (1 + max_loss_percent / 100)
                            stop_loss_pnl = (entry_price - stop_loss_price) / entry_price * 100
                    
                    active_bots.append({
                        'symbol': symbol,
                        'status': bot_data.get('status', 'unknown'),
                        'position_size': bot_data.get('position_size', 0),
                        'pnl': bot_data.get('pnl', 0),
                        'current_price': current_price,
                        'position_side': position_side,
                        'entry_price': bot_data.get('entry_price'),
                        'trailing_stop_active': bot_data.get('trailing_stop_active', False),
                        'stop_loss_price': bot_data.get('stop_loss_price'),
                        'stop_loss_pnl': stop_loss_pnl,
                        'position_start_time': bot_data.get('position_start_time'),
                        'max_position_hours': config.get('max_position_hours', 48),
                        'created_at': bot_data.get('created_at'),
                        'last_update': bot_data.get('last_update')
                    })
            
            return jsonify({
                'success': True,
                'bots': active_bots,
                'total': len(active_bots)
            })
            
    except Exception as e:
        logger.error(f"[API] ❌ Ошибка получения детальной информации о ботах: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bots_app.route('/api/bots/history', methods=['GET'])
def get_bot_history():
    """Получает историю действий ботов"""
    try:
        # Проверяем, что bot_history_manager доступен
        if bot_history_manager is None:
            logger.error("[API] bot_history_manager is None!")
            return jsonify({
                'success': False,
                'error': 'Bot history manager not initialized'
            }), 500
        
        symbol = request.args.get('symbol')
        action_type = request.args.get('action_type')
        limit = int(request.args.get('limit', 100))
        
        history = bot_history_manager.get_bot_history(symbol, action_type, limit)
        
        return jsonify({
            'success': True,
            'history': history,
            'count': len(history)
        })
        
    except Exception as e:
        logger.error(f"[API] Ошибка получения истории ботов: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/trades', methods=['GET'])
def get_bot_trades():
    """Получает историю торговых сделок ботов"""
    try:
        symbol = request.args.get('symbol')
        trade_type = request.args.get('trade_type')
        limit = int(request.args.get('limit', 100))
        
        trades = bot_history_manager.get_bot_trades(symbol, trade_type, limit)
        
        return jsonify({
            'success': True,
            'trades': trades,
            'count': len(trades)
        })
        
    except Exception as e:
        logger.error(f"[API] Ошибка получения сделок ботов: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/stops', methods=['GET'])
def get_stopped_trades():
    """Получает все сделки, закрытые по стопу (ПРЕМИУМ ФУНКЦИЯ!)"""
    try:
        # Проверяем лицензию
        try:
            from bot_engine.ai import check_premium_license
            is_premium = check_premium_license()
        except Exception as e:
            logger.warning(f"[API] Не удалось проверить лицензию: {e}")
            is_premium = False
        
        if not is_premium:
            return jsonify({
                'success': False,
                'error': 'Premium license required',
                'license_required': True,
                'message': 'Этот функционал доступен только с премиум лицензией'
            }), 403
        
        limit = int(request.args.get('limit', 100))
        
        stopped_trades = bot_history_manager.get_stopped_trades(limit)
        
        # Анализируем стопы через SmartRiskManager
        try:
            from bot_engine.ai.smart_risk_manager import SmartRiskManager
            smart_risk = SmartRiskManager()
            analysis = smart_risk.analyze_stopped_trades(limit)
        except ImportError:
            # SmartRiskManager недоступен (нет лицензии или ошибка)
            analysis = None
        
        return jsonify({
            'success': True,
            'trades': stopped_trades,
            'count': len(stopped_trades),
            'analysis': analysis,
            'premium': True,
            'message': 'Данные о стоп-сделках получены (Премиум)'
        })
        
    except Exception as e:
        logger.error(f"[API] Ошибка получения стопов: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/statistics', methods=['GET'])
def get_bot_statistics():
    """Получает статистику по ботам"""
    try:
        symbol = request.args.get('symbol')
        
        statistics = bot_history_manager.get_bot_statistics(symbol)
        
        return jsonify({
            'success': True,
            'statistics': statistics
        })
        
    except Exception as e:
        logger.error(f"[API] Ошибка получения статистики ботов: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/history/clear', methods=['POST'])
def clear_bot_history():
    """Очищает историю ботов"""
    try:
        data = request.get_json() or {}
        symbol = data.get('symbol')
        
        bot_history_manager.clear_history(symbol)
        
        message = f"История для {symbol} очищена" if symbol else "Вся история очищена"
        
        return jsonify({
            'success': True,
            'message': message
        })
        
    except Exception as e:
        logger.error(f"[API] Ошибка очистки истории ботов: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/history/demo', methods=['POST'])
def create_demo_history():
    """Создает демо-данные для истории ботов"""
    try:
        from bot_engine.bot_history import create_demo_data
        
        success = create_demo_data()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Демо-данные созданы успешно'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Ошибка создания демо-данных'
            }), 500
        
    except Exception as e:
        logger.error(f"[API] Ошибка создания демо-данных: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # КРИТИЧЕСКИ ВАЖНО: Проверяем и останавливаем старые процессы bots.py САМЫМ ПЕРВЫМ!
    print()  # Пустая строка для читаемости
    if not check_and_stop_existing_bots_processes():
        print("❌ Запуск отменен")
        sys.exit(0)
    
    # Загружаем конфигурацию Auto Bot после проверки процессов
    load_auto_bot_config()
    
    print("=" * 60)
    print("INFOBOT - Trading Bots Service")
    print("=" * 60)
    
    # Инициализация биржи будет выполнена в init_bot_service()
    print("*** ОСНОВНЫЕ ФУНКЦИИ:")
    print("  - Постоянный мониторинг RSI 6H для всех монет")
    print("  - Анализ тренда 6H (EMA50/EMA200)")
    print("  - Торговые боты с Auto Bot режимом")
    print("  - Автовход: RSI ≤29 = LONG, RSI ≥71 = SHORT")
    print()
    print(f"*** Порт: {SystemConfig.BOTS_SERVICE_PORT}")
    print("*** API Эндпоинты:")
    print("  GET  /health                    - Проверка статуса")
    print("  GET  /api/bots/coins-with-rsi   - Все монеты с RSI 6H")
    print("  GET  /api/bots/list             - Список ботов")
    print("  POST /api/bots/create           - Создать бота")
    print("  GET  /api/bots/auto-bot         - Конфигурация Auto Bot")
    print("  POST /api/bots/auto-bot         - Обновить Auto Bot")
    print("  GET  /api/bots/optimal-ema      - Оптимальные EMA периоды")
    print("  GET  /api/bots/optimal-ema-worker/status - Статус воркера EMA")
    print("  POST /api/bots/optimal-ema-worker/force-update - Принудительное обновление")
    print("=" * 60)
    print("*** Запуск...")
    
    run_bots_service()

    print("  - Постоянный мониторинг RSI 6H для всех монет")
    print("  - Анализ тренда 6H (EMA50/EMA200)")
    print("  - Торговые боты с Auto Bot режимом")
    print("  - Автовход: RSI ≤29 = LONG, RSI ≥71 = SHORT")
    print()
    print(f"*** Порт: {SystemConfig.BOTS_SERVICE_PORT}")
    print("*** API Эндпоинты:")
    print("  GET  /health                    - Проверка статуса")
    print("  GET  /api/bots/coins-with-rsi   - Все монеты с RSI 6H")
    print("  GET  /api/bots/list             - Список ботов")
    print("  POST /api/bots/create           - Создать бота")
    print("  GET  /api/bots/auto-bot         - Конфигурация Auto Bot")
    print("  POST /api/bots/auto-bot         - Обновить Auto Bot")
    print("  GET  /api/bots/optimal-ema      - Оптимальные EMA периоды")
    print("  GET  /api/bots/optimal-ema-worker/status - Статус воркера EMA")
    print("  POST /api/bots/optimal-ema-worker/force-update - Принудительное обновление")
    print("=" * 60)
    print("*** Запуск...")
    
    run_bots_service()
