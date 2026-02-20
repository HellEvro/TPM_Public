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
from copy import deepcopy
from datetime import datetime
from typing import Dict
from flask import Flask, request, jsonify

logger = logging.getLogger('BotsService')

# ⚡ ОПТИМИЗАЦИЯ ПАМЯТИ: Кэш для bots_state_file
_bots_state_cache = {
    'symbols': set(),
    'last_update': 0,
    'cache_ttl': 30  # Кэш на 30 секунд
}

def _get_cached_bot_symbols():
    """Получает кэшированный список символов ботов из файла"""
    import time
    current_time = time.time()
    
    # Проверяем, нужно ли обновить кэш
    if current_time - _bots_state_cache['last_update'] < _bots_state_cache['cache_ttl']:
        return _bots_state_cache['symbols'].copy()
    
    # Обновляем кэш
    saved_bot_symbols = set()
    try:
        bots_state_file = 'data/bots_state.json'
        if os.path.exists(bots_state_file):
            # Проверяем только время изменения файла, не читаем весь файл если не изменился
            file_mtime = os.path.getmtime(bots_state_file)
            if file_mtime <= _bots_state_cache.get('file_mtime', 0):
                # Файл не изменился - используем старый кэш
                return _bots_state_cache['symbols'].copy()
            
            # Файл изменился - читаем только ключи ботов
            with open(bots_state_file, 'r', encoding='utf-8') as f:
                file_content = f.read()
                if file_content.strip():
                    try:
                        saved_data = json.loads(file_content)
                        if 'bots' in saved_data and isinstance(saved_data['bots'], dict):
                            saved_bot_symbols = set(saved_data['bots'].keys())
                    except json.JSONDecodeError:
                        # Если ошибка парсинга - используем старый кэш
                        return _bots_state_cache['symbols'].copy()
        
        # Обновляем кэш
        _bots_state_cache['symbols'] = saved_bot_symbols
        _bots_state_cache['last_update'] = current_time
        if os.path.exists(bots_state_file):
            _bots_state_cache['file_mtime'] = os.path.getmtime(bots_state_file)
    except Exception as e:
        pass
        # В случае ошибки возвращаем старый кэш
        return _bots_state_cache['symbols'].copy()
    
    return saved_bot_symbols


def _load_json_file(file_path):
    """Безопасно считывает JSON и возвращает (data, iso_timestamp)."""
    if not os.path.exists(file_path):
        return None, None
    try:
        with open(file_path, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        updated_at = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
        return data, updated_at
    except Exception as exc:
        logger.warning(f"[AI_OPTIMIZER] Не удалось прочитать {file_path}: {exc}")
        return None, None

# Импорт SystemConfig
from bot_engine.config_loader import SystemConfig

# Импорт Flask приложения и глобальных переменных из imports_and_globals
from bots_modules.imports_and_globals import (
    bots_app, exchange, smart_rsi_manager, async_processor,
    bots_data_lock, bots_data, rsi_data_lock, coins_rsi_data,
    bots_cache_data, bots_cache_lock, process_state,
    system_initialized, shutdown_flag, mature_coins_storage,
    mature_coins_lock, coin_processing_locks,
    BOT_STATUS, ASYNC_AVAILABLE, RSI_CACHE_FILE, bot_history_manager,
    get_exchange, load_individual_coin_settings,
    get_individual_coin_settings, set_individual_coin_settings,
    remove_individual_coin_settings, copy_individual_coin_settings_to_all,
    remove_all_individual_coin_settings, RealTradingBot,
    get_config_snapshot, get_insufficient_funds, set_insufficient_funds
)
import bots_modules.imports_and_globals as globals_module

# Импорт RSI констант из bot_config
# Enhanced RSI константы теперь в SystemConfig

# Импорт функций из других модулей
try:
    from bots_modules.sync_and_cache import (
        update_bots_cache_data, save_system_config, load_system_config,
        get_system_config_snapshot,
        save_auto_bot_config, save_bots_state, save_rsi_cache,
        save_process_state, restore_default_config, load_default_config
    )
    from bots_modules.init_functions import (
        ensure_exchange_initialized,
        create_bot,
        init_bot_service,
        start_async_processor,
        stop_async_processor,
    )
    from bots_modules.maturity import (
        save_mature_coins_storage, load_mature_coins_storage,
        remove_mature_coin_from_storage, check_coin_maturity_with_storage
    )
    # ❌ ОТКЛЮЧЕНО: optimal_ema перемещен в backup (используются заглушки из imports_and_globals)
    # from bots_modules.optimal_ema import (
    #     load_optimal_ema_data, update_optimal_ema_data
    # )
    from bots_modules.filters import (
        get_effective_signal, check_auto_bot_filters,
        process_auto_bot_signals, test_exit_scam_filter, test_rsi_time_filter,
        process_trading_signals_for_all_bots, get_coin_rsi_data
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
    def get_system_config_snapshot():
        return {}
    def save_auto_bot_config():
        pass
    def save_bots_state():
        pass
    def save_rsi_cache():
        pass
    def save_process_state():
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

# ❌ ОТКЛЮЧЕНО: optimal_ema перемещен в backup (используется заглушка из imports_and_globals)
# try:
#     from bots_modules.optimal_ema import get_optimal_ema_periods
# except:
#     def get_optimal_ema_periods(symbol):
#         return {}

def health_check():
    """Проверка состояния сервиса"""
    try:
        logger.info(f"✅ Flask работает, запрос обработан")
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
        
        # Сбрасываем флаг «недостаточно средств», если доступный баланс достаточен (например > 5 USDT)
        if get_insufficient_funds() and account_info.get("success") and float(account_info.get("total_available_balance", 0)) > 5:
            set_insufficient_funds(False)
        
        # Добавляем информацию о ботах и флаг недостатка средств для UI
        # ⚡ БЕЗ БЛОКИРОВКИ: чтение словаря - атомарная операция
        bots_list = list(bots_data['bots'].values())
        account_info["bots_count"] = len(bots_list)
        account_info["active_bots"] = sum(1 for bot in bots_list 
                                        if bot.get('status') not in ['paused'])
        account_info["insufficient_funds"] = get_insufficient_funds()
        
        response = jsonify(account_info)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except Exception as e:
        logger.error(f" Ошибка получения информации о счете: {str(e)}")
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
                import shutil
                bots_state_file = 'data/bots_state.json'
                if os.path.exists(bots_state_file):
                    with open(bots_state_file, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                        # Проверяем, что файл не пустой
                        if file_content.strip():
                            try:
                                saved_data = json.loads(file_content)
                                if 'bots' in saved_data:
                                    saved_bot_symbols = set(saved_data['bots'].keys())
                            except json.JSONDecodeError as e:
                                logger.warning(f"⚠️ Ошибка парсинга JSON (строка {e.lineno}, колонка {e.colno}): {e.msg}")
                                pass
                                
                                # ✅ Пытаемся восстановить из резервной копии
                                backup_file = f"{bots_state_file}.backup"
                                corrupted_file = f"{bots_state_file}.corrupted"
                                
                                if os.path.exists(backup_file):
                                    try:
                                        logger.info(f"🔄 Пытаемся восстановить из резервной копии: {backup_file}")
                                        with open(backup_file, 'r', encoding='utf-8') as backup_f:
                                            saved_data = json.load(backup_f)
                                            if 'bots' in saved_data:
                                                saved_bot_symbols = set(saved_data['bots'].keys())
                                                logger.info(f"✅ Восстановлено из резервной копии: {len(saved_bot_symbols)} ботов")
                                                # Восстанавливаем основной файл из резервной копии
                                                shutil.copy2(backup_file, bots_state_file)
                                                logger.info(f"✅ Основной файл восстановлен из резервной копии")
                                    except Exception as backup_error:
                                        logger.error(f"❌ Ошибка восстановления из резервной копии: {backup_error}")
                                
                                # Сохраняем поврежденный файл для анализа
                                try:
                                    shutil.copy2(bots_state_file, corrupted_file)
                                    logger.info(f"📁 Поврежденный файл сохранен: {corrupted_file}")
                                except Exception as copy_error:
                                    pass
                        else:
                            logger.warning(" ⚠️ Файл состояния пустой! Пытаемся восстановить или инициализировать...")
                            # ✅ Пытаемся восстановить из резервной копии
                            backup_file = f"{bots_state_file}.backup"
                            if os.path.exists(backup_file):
                                try:
                                    logger.info(f"🔄 Пытаемся восстановить из резервной копии: {backup_file}")
                                    with open(backup_file, 'r', encoding='utf-8') as backup_f:
                                        backup_content = backup_f.read()
                                        if backup_content.strip():
                                            saved_data = json.loads(backup_content)
                                            if 'bots' in saved_data:
                                                saved_bot_symbols = set(saved_data['bots'].keys())
                                                logger.info(f"✅ Восстановлено из резервной копии: {len(saved_bot_symbols)} ботов")
                                                # Восстанавливаем основной файл из резервной копии
                                                shutil.copy2(backup_file, bots_state_file)
                                                logger.info(f"✅ Основной файл восстановлен из резервной копии")
                                except Exception as backup_error:
                                    logger.error(f"❌ Ошибка восстановления из резервной копии: {backup_error}")
                            
                            # ✅ Если резервной копии нет или она тоже пустая - инициализируем файл
                            if not saved_bot_symbols:
                                try:
                                    from bots_modules.sync_and_cache import load_bots_state, save_bots_state
                                    # Пытаемся загрузить состояние через стандартную функцию (она может синхронизировать с биржей)
                                    if not load_bots_state():
                                        # Если загрузка не удалась - создаем базовую структуру
                                        logger.info("📝 Инициализируем файл состояния с базовой структурой...")
                                        from datetime import datetime
                                        default_state = {
                                            'version': '1.0',
                                            'last_saved': datetime.now().isoformat(),
                                            'bots': {},
                                            'global_stats': {
                                                'total_trades': 0,
                                                'total_profit': 0.0,
                                                'win_rate': 0.0
                                            }
                                        }
                                        with open(bots_state_file, 'w', encoding='utf-8') as f:
                                            json.dump(default_state, f, ensure_ascii=False, indent=2)
                                        logger.info("✅ Файл состояния инициализирован с базовой структурой")
                                except Exception as init_error:
                                    logger.error(f"❌ Ошибка инициализации файла состояния: {init_error}")
            except Exception as e:
                logger.warning(f" ⚠️ Не удалось загрузить сохраненных ботов: {e}")
            
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
        logger.error(f" ❌ Ошибка обновления ручных позиций: {str(e)}")
        import traceback
        logger.error(f" ❌ Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/coins-with-rsi', methods=['GET'])
def get_coins_with_rsi():
    """Получить все монеты с RSI данными (по текущему таймфрейму).
    Единый источник RSI для отображения: списки монет, фильтры, «боты в работе», карточки ботов
    и миниграфики — везде используются coins_rsi_data['coins'] (обновляется continuous_data_loader
    и для позиций — sync_positions → _refresh_rsi_for_bots_in_position)."""
    # ⚡ ИНИЦИАЛИЗАЦИЯ: Инициализируем переменные до try блока для обработки ошибок
    cleaned_coins = {}
    manual_positions = []
    cache_age = None
    
    try:
        # Проверяем параметр refresh_symbol для обновления конкретной монеты
        refresh_symbol = request.args.get('refresh_symbol')
        if refresh_symbol:
            logger.info(f"🔄 Запрос на обновление RSI данных для {refresh_symbol}")
            try:
                if ensure_exchange_initialized():
                    coin_data = get_coin_rsi_data(refresh_symbol, get_exchange())
                    if coin_data:
                        # ⚡ БЕЗ БЛОКИРОВКИ: GIL делает запись атомарной
                        coins_rsi_data['coins'][refresh_symbol] = coin_data
                        logger.info(f"✅ RSI данные для {refresh_symbol} обновлены")
                    else:
                        logger.warning(f"⚠️ Не удалось обновить RSI данные для {refresh_symbol}")
            except Exception as e:
                logger.error(f"❌ Ошибка обновления RSI для {refresh_symbol}: {e}")
        
        # ⚡ ОПТИМИЗАЦИЯ ПАМЯТИ: Получаем размер данных перед обработкой
        coins_count = len(coins_rsi_data['coins'])
        if coins_count > 1000:
            logger.warning(f"⚠️ Большое количество монет ({coins_count}), возможны проблемы с памятью")
        
        # ⚡ БЕЗ БЛОКИРОВКИ: чтение словаря - атомарная операция
        # Проверяем возраст кэша
        if os.path.exists(RSI_CACHE_FILE):
            try:
                cache_stat = os.path.getmtime(RSI_CACHE_FILE)
                cache_age = (time.time() - cache_stat) / 60  # в минутах
            except:
                cache_age = None
        
        # ⚡ ОПТИМИЗАЦИЯ ПАМЯТИ: Минимизируем копирование данных
        # Очищаем данные от несериализуемых объектов
        coins_items = list(coins_rsi_data['coins'].items())  # Создаем список один раз
        for symbol, coin_data in coins_items:
            # ✅ ИСПРАВЛЕНИЕ: НЕ фильтруем монеты по зрелости для UI!
            # Фильтр зрелости применяется в get_coin_rsi_data() через изменение сигнала на WAIT
            # Здесь показываем ВСЕ монеты, независимо от зрелости
            
            try:
                # ⚡ ОПТИМИЗАЦИЯ ПАМЯТИ: Создаем только необходимые поля, избегаем глубокого копирования
                cleaned_coin = {}
                
                # Копируем только необходимые базовые поля
                # Получаем ключи для текущего таймфрейма
                from bot_engine.config_loader import get_current_timeframe, get_rsi_key, get_trend_key
                current_timeframe = get_current_timeframe()
                rsi_key = get_rsi_key(current_timeframe)
                trend_key = get_trend_key(current_timeframe)
                
                essential_fields = ['symbol', rsi_key, trend_key, 'rsi_zone', 'signal', 'price', 
                                  'change24h', 'last_update', 'blocked_by_scope', 'has_existing_position',
                                  'is_mature', 'maturity_reason', 'blocked_by_exit_scam', 'blocked_by_rsi_time', 'blocked_by_loss_reentry',
                                  'trading_status', 'is_delisting']
                # Также добавляем старые ключи для обратной совместимости
                essential_fields.extend(['rsi6h', 'trend6h', 'rsi', 'trend'])
                
                for field in essential_fields:
                    if field in coin_data:
                        cleaned_coin[field] = coin_data[field]
                
                # Копируем структурированные данные только если они есть
                if 'time_filter_info' in coin_data and coin_data['time_filter_info']:
                    cleaned_coin['time_filter_info'] = coin_data['time_filter_info']
                if 'exit_scam_info' in coin_data and coin_data['exit_scam_info']:
                    cleaned_coin['exit_scam_info'] = coin_data['exit_scam_info']
                if 'loss_reentry_info' in coin_data and coin_data['loss_reentry_info']:
                    cleaned_coin['loss_reentry_info'] = coin_data['loss_reentry_info']
                
                # ⚡ ОПТИМИЗАЦИЯ: Очищаем enhanced_rsi только если он есть
                if 'enhanced_rsi' in coin_data and coin_data['enhanced_rsi']:
                    enhanced_rsi = coin_data['enhanced_rsi']
                    cleaned_enhanced_rsi = {}
                    
                    # Копируем только необходимые поля из enhanced_rsi
                    if 'enabled' in enhanced_rsi:
                        cleaned_enhanced_rsi['enabled'] = enhanced_rsi['enabled']
                    
                    # Очищаем confirmations от numpy типов
                    if 'confirmations' in enhanced_rsi and enhanced_rsi['confirmations']:
                        confirmations = {}
                        for key, value in enhanced_rsi['confirmations'].items():
                            if hasattr(value, 'item'):  # numpy scalar
                                confirmations[key] = value.item()
                            elif value is not None:
                                confirmations[key] = value
                            else:
                                confirmations[key] = None
                        cleaned_enhanced_rsi['confirmations'] = confirmations
                        
                        # Копируем Stochastic RSI данные в основные поля для совместимости с UI
                        cleaned_coin['stoch_rsi_k'] = confirmations.get('stoch_rsi_k')
                        cleaned_coin['stoch_rsi_d'] = confirmations.get('stoch_rsi_d')
                    
                    # Конвертируем adaptive_levels если это tuple
                    if 'adaptive_levels' in enhanced_rsi and enhanced_rsi['adaptive_levels']:
                        adaptive_levels = enhanced_rsi['adaptive_levels']
                        cleaned_enhanced_rsi['adaptive_levels'] = list(adaptive_levels) if isinstance(adaptive_levels, tuple) else adaptive_levels
                    
                    cleaned_coin['enhanced_rsi'] = cleaned_enhanced_rsi
                else:
                    cleaned_coin['enhanced_rsi'] = {'enabled': False}
                
                # Копируем trend_analysis если есть
                if 'trend_analysis' in coin_data and coin_data['trend_analysis']:
                    cleaned_coin['trend_analysis'] = coin_data['trend_analysis']
                
                # Добавляем эффективный сигнал для единообразия с фронтендом
                effective_signal = get_effective_signal(cleaned_coin)
                cleaned_coin['effective_signal'] = effective_signal
                # В список LONG/SHORT слева попадают только монеты, прошедшие проверку AI (как в potential_coins)
                # КРИТИЧНО: AI блокировка только когда AIConfig.AI_ENABLED и (ai_enabled или full_ai_control)
                if effective_signal in ('ENTER_LONG', 'ENTER_SHORT'):
                    try:
                        auto_config = bots_data.get('auto_bot_config', {})
                        from bot_engine.config_live import get_ai_config_attr
                        ai_modules_on = get_ai_config_attr('AI_ENABLED', False)
                        ai_check_needed = bool(
                            ai_modules_on and (auto_config.get('ai_enabled') or auto_config.get('full_ai_control'))
                        )
                        if ai_check_needed:
                            from bot_engine.ai.ai_integration import should_open_position_with_ai
                            direction = 'LONG' if effective_signal == 'ENTER_LONG' else 'SHORT'
                            rsi_val = cleaned_coin.get('rsi') or cleaned_coin.get(rsi_key)
                            trend_val = cleaned_coin.get('trend') or cleaned_coin.get(trend_key) or 'NEUTRAL'
                            price_val = float(cleaned_coin.get('price') or 0)
                            config_snapshot = get_config_snapshot(symbol)
                            filter_config = (config_snapshot.get('merged') or {}) if config_snapshot else auto_config
                            if not filter_config:
                                filter_config = auto_config
                            ai_result = should_open_position_with_ai(
                                symbol=symbol,
                                direction=direction,
                                rsi=rsi_val or 50,
                                trend=trend_val,
                                price=price_val,
                                config=filter_config,
                                candles=None
                            )
                            if ai_result.get('ai_used') and not ai_result.get('should_open'):
                                effective_signal = 'WAIT'
                                cleaned_coin['effective_signal'] = 'WAIT'
                                cleaned_coin['signal_block_reason'] = ai_result.get('reason') or 'AI не рекомендует вход'
                    except Exception as ai_err:
                        pass
                
                # ✅ ИСПРАВЛЕНИЕ: Добавляем количество свечей из данных зрелых монет
                try:
                    from bots_modules.imports_and_globals import mature_coins_storage
                    if symbol in mature_coins_storage:
                        maturity_data = mature_coins_storage[symbol].get('maturity_data', {})
                        details = maturity_data.get('details', {})
                        candles_count = details.get('candles_count')
                        if candles_count is not None:
                            cleaned_coin['candles_count'] = candles_count
                except Exception as e:
                    pass
                
                cleaned_coins[symbol] = cleaned_coin
                
            except MemoryError:
                logger.error(f"❌ MemoryError при обработке монеты {symbol}, пропускаем")
                continue
            except Exception as e:
                logger.warning(f"⚠️ Ошибка обработки монеты {symbol}: {e}, пропускаем")
                continue
        
        # Получаем список монет с ручными позициями на бирже (позиции БЕЗ ботов)
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
                
                # ⚡ ОПТИМИЗАЦИЯ: Используем кэшированный список сохраненных ботов
                saved_bot_symbols = _get_cached_bot_symbols()
                
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
        except Exception as e:
            logger.error(f" Ошибка получения ручных позиций: {str(e)}")
        
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
            pass
        return jsonify(result)
        
    except MemoryError as e:
        logger.error(f"❌ MemoryError при получении монет с RSI: {e}")
        cleaned_count = len(cleaned_coins) if cleaned_coins else 0
        logger.error(f"❌ Количество обработанных монет перед ошибкой: {cleaned_count}")
        # Возвращаем частичные данные если они есть
        if cleaned_coins and cleaned_count > 0:
            result = {
                'success': True,
                'coins': cleaned_coins,
                'total': len(cleaned_coins),
                'warning': 'Данные могут быть неполными из-за нехватки памяти',
                'last_update': coins_rsi_data.get('last_update'),
                'update_in_progress': coins_rsi_data.get('update_in_progress', False),
                'data_version': coins_rsi_data.get('data_version', 0),
                'manual_positions': [],
                'stats': {
                    'total_coins': coins_rsi_data.get('total_coins', 0),
                    'successful_coins': len(cleaned_coins),
                    'failed_coins': coins_rsi_data.get('total_coins', 0) - len(cleaned_coins)
                }
            }
            return jsonify(result), 200
        else:
            return jsonify({'success': False, 'error': 'Нехватка памяти при обработке данных'}), 500
    except Exception as e:
        logger.error(f" Ошибка получения монет с RSI: {str(e)}")
        import traceback
        logger.error(f" Traceback: {traceback.format_exc()}")
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
        
        # Виртуальные позиции ПРИИ (FullAI Adaptive) — показываем в списке «Боты в работе» с пометкой «Виртуальная»
        try:
            ac = bots_data.get('auto_bot_config') or {}
            if ac.get('full_ai_control'):
                from bots_modules.fullai_adaptive import get_virtual_positions_for_api, is_adaptive_enabled
                if is_adaptive_enabled():
                    vp_list = get_virtual_positions_for_api()
                    # Подмешиваем текущую цену из RSI-кэша для отображения в карточке
                    with rsi_data_lock:
                        coins = coins_rsi_data.get('coins') or {}
                        for pos in vp_list:
                            sym = pos.get('symbol')
                            if sym and sym in coins:
                                pos['current_price'] = coins[sym].get('price')
                    response_data['virtual_positions'] = vp_list
                else:
                    response_data['virtual_positions'] = []
            else:
                response_data['virtual_positions'] = []
        except Exception:
            response_data['virtual_positions'] = []
        
        # ✅ Не логируем частые запросы списка ботов
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения списка ботов: {e}")
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
        client_config = data.get('config', {}) or {}
        skip_maturity_check = data.get('skip_maturity_check', False)
        force_manual_entry = data.get('force_manual_entry', False) or data.get('ignore_filters', False)
        if force_manual_entry:
            skip_maturity_check = True  # принудительно отключаем зрелость для ручного запуска
        
        logger.info(f" Запрос на создание бота для {symbol}")
        logger.info(f" Клиентские overrides: {client_config}")
        
        # 🔄 Получаем актуальные настройки из бэкенда (global + individual)
        merged_server_config = {}
        try:
            snapshot = get_config_snapshot(symbol)
            merged_server_config = deepcopy(snapshot.get('merged') or snapshot.get('global') or {})
        except Exception as snapshot_error:
            logger.warning(f" ⚠️ Не удалось получить config snapshot для {symbol}: {snapshot_error}")
        
        if not merged_server_config:
            with bots_data_lock:
                merged_server_config = deepcopy(bots_data.get('auto_bot_config', {}))
        
        allowed_manual_overrides = {
            'volume_mode', 'volume_value', 'leverage',
            'status', 'auto_managed', 'margin_usdt'
        }
        manual_overrides = {
            key: client_config[key]
            for key in allowed_manual_overrides
            if key in client_config
        }
        
        bot_runtime_config = merged_server_config.copy()
        bot_runtime_config.update(manual_overrides)
        
        logger.info(f" 🧠 Серверный конфиг для {symbol}: avoid_up_trend={bot_runtime_config.get('avoid_up_trend')} / avoid_down_trend={bot_runtime_config.get('avoid_down_trend')}")
        logger.info(f" 🔍 Размер merged конфига: {len(merged_server_config)} ключей, размер итогового конфига: {len(bot_runtime_config)} ключей")
        logger.info(f" 🔍 Individual settings для {symbol}: {snapshot.get('individual') is not None}")
        if snapshot.get('individual'):
            logger.info(f" 🔍 Individual settings содержат avoid_up_trend: {'avoid_up_trend' in snapshot.get('individual', {})}")
        
        # ✅ Проверяем, есть ли ручная позиция для этой монеты
        has_manual_position = False
        try:
            current_exchange = get_exchange()
            if current_exchange:
                positions_response = current_exchange.get_positions()
                if isinstance(positions_response, tuple):
                    positions_list = positions_response[0] if positions_response else []
                else:
                    positions_list = positions_response if positions_response else []
                
                # Проверяем, есть ли позиция для этой монеты без бота в системе
                for pos in positions_list:
                    pos_symbol = pos.get('symbol', '').replace('USDT', '')
                    if pos_symbol == symbol and abs(float(pos.get('size', 0))) > 0:
                        # Проверяем, нет ли уже бота для этой монеты
                        if symbol not in bots_data.get('bots', {}):
                            has_manual_position = True
                            logger.info(f" ✋ {symbol}: Обнаружена ручная позиция - пропускаем проверку зрелости")
                            break
        except Exception as e:
            pass
        
        # Проверяем зрелость монеты (если включена проверка для этой монеты И нет ручной позиции)
        enable_maturity_check_coin = bot_runtime_config.get('enable_maturity_check', True)
        if skip_maturity_check:
            logger.info(f" ✋ {symbol}: Принудительное создание бота - проверка зрелости отключена")
            enable_maturity_check_coin = False
            # Дополнительно отключаем защитные фильтры, если запрос ручной
            bot_runtime_config['avoid_down_trend'] = False
            bot_runtime_config['avoid_up_trend'] = False
            bot_runtime_config['rsi_time_filter_enabled'] = False
        
        if enable_maturity_check_coin and not has_manual_position:
            # Проверяем зрелость по текущему системному ТФ (хранилище или загрузка свечей при верификации)
            from bots_modules.filters import check_coin_maturity_stored_or_verify
            is_mature, maturity_reason = check_coin_maturity_stored_or_verify(symbol)
            if not is_mature:
                from bot_engine.config_loader import get_current_timeframe
                tf = get_current_timeframe()
                err_msg = maturity_reason or f'Проверка по таймфрейму {tf}'
                logger.warning(f" {symbol}: Монета не прошла проверку зрелости: {err_msg}")
                return jsonify({
                    'success': False,
                    'error': f'Монета {symbol} не прошла проверку зрелости: {err_msg}'
                }), 400
        elif has_manual_position:
            logger.info(f" ✋ {symbol}: Ручная позиция обнаружена - проверка зрелости пропущена")
        
        # Создаем бота
        bot_state = create_bot(symbol, bot_runtime_config, exchange_obj=get_exchange())
        
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
                        logger.info(f" 🔍 {symbol}: Обнаружена существующая позиция на бирже (размер: {pos.get('size')})")
                        break
        except Exception as e:
            pass
        
        # ✅ Возвращаем ответ БЫСТРО
        logger.info(f" ✅ Бот для {symbol} создан")
        
        manual_signal = data.get('signal')
        manual_direction = None
        if manual_signal:
            signal_upper = str(manual_signal).upper()
            if 'SHORT' in signal_upper:
                manual_direction = 'SHORT'
            elif 'LONG' in signal_upper:
                manual_direction = 'LONG'

        # ✅ Запускаем вход в позицию АСИНХРОННО (только если НЕТ существующей позиции!)
        # Бот в списке = проверки пройдены → обязан по рынку зайти в сделку, без ожидания сигнала.
        if not has_existing_position:
            def enter_position_async():
                try:
                    # ✅ Перечитываем конфиг с диска перед решением о входе — пороги RSI из UI учитываются
                    try:
                        from bots_modules.imports_and_globals import load_auto_bot_config
                        if hasattr(load_auto_bot_config, '_last_mtime'):
                            load_auto_bot_config._last_mtime = 0
                        load_auto_bot_config()
                    except Exception:
                        pass
                    direction = None
                    if force_manual_entry and manual_direction:
                        direction = manual_direction
                        logger.info(f" 🚀 Принудительный вход в {direction} для {symbol} (ручной запуск)")
                        # Предупреждение: ручной вход без проверки RSI — может быть «против» порогов
                        try:
                            from bot_engine.config_loader import get_rsi_from_coin_data, get_current_timeframe
                            with rsi_data_lock:
                                _cd = coins_rsi_data['coins'].get(symbol)
                            _rsi = get_rsi_from_coin_data(_cd, timeframe=get_current_timeframe()) if _cd else None
                            if _rsi is not None:
                                with bots_data_lock:
                                    _cfg = bots_data.get('auto_bot_config', {})
                                    _long_th = bot_state.get('rsi_long_threshold') or _cfg.get('rsi_long_threshold', 29)
                                    _short_th = bot_state.get('rsi_short_threshold') or _cfg.get('rsi_short_threshold', 71)
                                if direction == 'LONG' and _rsi > _long_th:
                                    logger.warning(f" ⚠️ Ручной LONG при RSI={_rsi:.1f} > порога {_long_th} — вход не по конфигу RSI")
                                elif direction == 'SHORT' and _rsi < _short_th:
                                    logger.warning(f" ⚠️ Ручной SHORT при RSI={_rsi:.1f} < порога {_short_th} — вход не по конфигу RSI")
                        except Exception:
                            pass
                    else:
                        # Автовход — RSI строго по системному ТФ (get_rsi_from_coin_data), без fallback на 'rsi'
                        from bot_engine.config_loader import get_rsi_from_coin_data, get_current_timeframe
                        with rsi_data_lock:
                            coin_data = coins_rsi_data['coins'].get(symbol)
                        tf = get_current_timeframe()
                        rsi_val = get_rsi_from_coin_data(coin_data, timeframe=tf) if coin_data else None
                        if rsi_val is not None:
                            rsi_val = float(rsi_val)
                            with bots_data_lock:
                                auto_config = bots_data.get('auto_bot_config', {})
                                rsi_long_threshold = bot_state.get('rsi_long_threshold') or auto_config.get('rsi_long_threshold', 29)
                                rsi_short_threshold = bot_state.get('rsi_short_threshold') or auto_config.get('rsi_short_threshold', 71)
                            # Направление только если RSI в пороге (сигнал из coin_data мог быть от другого ТФ до фикса в filters)
                            if rsi_val <= rsi_long_threshold:
                                direction = 'LONG'
                                logger.info(f" 🚀 Вход по рынку для {symbol}: RSI={rsi_val:.1f} <= {rsi_long_threshold} (ТФ={tf}) → LONG")
                            elif rsi_val >= rsi_short_threshold:
                                direction = 'SHORT'
                                logger.info(f" 🚀 Вход по рынку для {symbol}: RSI={rsi_val:.1f} >= {rsi_short_threshold} (ТФ={tf}) → SHORT")
                            elif coin_data and coin_data.get('signal') in ['ENTER_LONG', 'ENTER_SHORT']:
                                logger.warning(f" ⚠️ {symbol}: сигнал {coin_data.get('signal')}, но RSI={rsi_val:.1f} вне порогов (LONG<={rsi_long_threshold}, SHORT>={rsi_short_threshold}) — вход отменён")
                        else:
                            logger.info(f" ℹ️ {symbol}: нет RSI по ТФ {tf} — вход отменён")
                    
                    if direction:
                        trading_bot = RealTradingBot(symbol, get_exchange(), bot_state)
                        force_market = True
                        try:
                            with bots_data_lock:
                                cfg = bots_data.get('auto_bot_config', {})
                            if cfg.get('limit_orders_entry_enabled') or cfg.get('rsi_limit_entry_enabled'):
                                force_market = False
                        except Exception:
                            pass
                        import time
                        intended_price = 0.0
                        try:
                            with rsi_data_lock:
                                _cd = coins_rsi_data.get('coins', {}).get(symbol, {})
                            intended_price = float(_cd.get('price', 0) or 0)
                        except Exception:
                            pass
                        _t0 = time.time()
                        result = trading_bot._enter_position(direction, force_market_entry=force_market)
                        _delay = time.time() - _t0
                        if result and result.get('success'):
                            logger.info(f" ✅ Успешно вошли в {direction} позицию для {symbol}")
                            with bots_data_lock:
                                bots_data['bots'][symbol] = trading_bot.to_dict()
                            # FullAI аналитика: всегда запись real_open с полными данными
                            try:
                                from bot_engine.fullai_analytics import append_event, EVENT_REAL_OPEN
                                from bots_modules.fullai_adaptive import build_real_open_extra
                                actual_price = float(result.get('entry_price') or intended_price or 0)
                                order_type = 'Limit' if not force_market else 'Market'
                                extra = build_real_open_extra(
                                    symbol=symbol, direction=direction,
                                    intended_price=intended_price or actual_price,
                                    actual_price=actual_price,
                                    order_type=order_type, delay_sec=_delay,
                                )
                                append_event(
                                    symbol=symbol,
                                    event_type=EVENT_REAL_OPEN,
                                    direction=direction,
                                    is_virtual=False,
                                    reason=extra.get('attempt_label', 'Реальная сделка'),
                                    extra=extra,
                                )
                            except Exception as _fa_err:
                                logger.debug("FullAI analytics real_open (API): %s", _fa_err)
                        else:
                            error_msg = (result or {}).get('error', 'unknown')
                            if 'MIN_NOTIONAL' in error_msg or '110007' in error_msg or 'меньше минимального ордера' in error_msg or 'Недостаточно доступного остатка' in error_msg:
                                logger.warning(f" ⚠️ Не удалось войти в {direction} позицию для {symbol}: {error_msg}")
                            else:
                                logger.error(f" ❌ НЕ УДАЛОСЬ войти в {direction} позицию для {symbol}: {error_msg}")
                    else:
                        logger.info(f" ℹ️ {symbol}: RSI не в зоне порогов конфига — бот будет ждать условия в следующем цикле")
                except Exception as e:
                    err_str = str(e)
                    if 'MIN_NOTIONAL' in err_str or '110007' in err_str or 'меньше минимального ордера' in err_str or 'Недостаточно доступного остатка' in err_str:
                        logger.warning(f" ⚠️ Ошибка входа в позицию: {e}")
                    else:
                        logger.error(f" ❌ Ошибка входа в позицию: {e}")
            
            # Запускаем асинхронно
            thread = threading.Thread(target=enter_position_async)
            thread.daemon = True
            thread.start()
        else:
            # ✅ Для существующей позиции - просто запускаем синхронизацию
            logger.info(f" 🔄 {symbol}: Запускаем синхронизацию существующей позиции...")
            
            def sync_existing_position():
                try:
                    from bots_modules.sync_and_cache import sync_bots_with_exchange
                    sync_bots_with_exchange()
                    logger.info(f" ✅ Синхронизация позиции {symbol} завершена")
                except Exception as e:
                    logger.error(f" ❌ Ошибка синхронизации: {e}")
            
            thread = threading.Thread(target=sync_existing_position)
            thread.daemon = True
            thread.start()
        
        return jsonify({
            'success': True,
            'message': f'Бот для {symbol} создан успешно',
            'bot': bot_state,
            'existing_position': has_existing_position
        })
        
    except Exception as e:
        logger.error(f" Ошибка создания бота: {str(e)}")
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
                logger.info(f" {symbol}: Бот запущен (снята пауза)")
            else:
                logger.info(f" {symbol}: Бот уже активен")
        
        return jsonify({
            'success': True,
            'message': f'Бот для {symbol} запущен'
        })
            
    except Exception as e:
        logger.error(f" Ошибка запуска бота: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/stop', methods=['POST'])
def stop_bot_endpoint():
    """Остановить бота"""
    try:
        logger.info(f"📥 Получен запрос остановки бота: {request.get_data()}")
        
        # Пробуем разные способы получения данных
        try:
            data = request.get_json()
        except Exception as json_error:
            logger.error(f"❌ Ошибка парсинга JSON: {json_error}")
            # Пробуем получить данные как form data
            data = request.form.to_dict()
            if not data:
                # Пробуем получить данные из args
                data = request.args.to_dict()
        
        logger.info(f"📊 Распарсенные данные: {data}")
        
        if not data or not data.get('symbol'):
            logger.error(f"❌ Отсутствует symbol в данных: {data}")
            return jsonify({'success': False, 'error': 'Symbol required'}), 400
        
        symbol = data['symbol']
        reason = data.get('reason', 'Остановлен пользователем')
        
        # Проверяем, есть ли открытая позиция у бота
        # ⚡ БЕЗ БЛОКИРОВКИ: чтение и простое присваивание - атомарные операции
        position_to_close = None
        
        logger.info(f"🔍 Проверяем наличие бота {symbol}...")
        if symbol not in bots_data['bots']:
            logger.error(f"❌ Бот {symbol} не найден!")
            return jsonify({'success': False, 'error': 'Bot not found'}), 404
        
        logger.info(f"✅ Бот {symbol} найден, останавливаем...")
        
        # ⚡ КРИТИЧНО: Используем блокировку для атомарной операции!
        with bots_data_lock:
            bot_data = bots_data['bots'][symbol]
            old_status = bot_data['status']
            logger.info(f"📊 Старый статус: {old_status}")
            
            # Проверяем, есть ли открытая позиция
            if bot_data.get('position_side') in ['LONG', 'SHORT']:
                position_to_close = bot_data['position_side']
                logger.info(f" {symbol}: Найдена открытая позиция {position_to_close}, будет закрыта при остановке")
            
            bot_data['status'] = BOT_STATUS['PAUSED']
            # НЕ сбрасываем entry_price для возможности возобновления
            # НЕ сбрасываем position_side - оставляем для отображения в UI
            # bot_data['position_side'] = None
            # bot_data['unrealized_pnl'] = 0.0
            logger.warning(f" {symbol}: Бот остановлен, новый статус: {bot_data['status']}")
            
            # Обновляем глобальную статистику
            bots_data['global_stats']['active_bots'] = len([bot for bot in bots_data['bots'].values() if bot.get('status') in ['running', 'idle']])
            bots_data['global_stats']['bots_in_position'] = len([bot for bot in bots_data['bots'].values() if bot.get('position_side')])
        
        # ⚠️ НЕ ЗАКРЫВАЕМ ПОЗИЦИЮ АВТОМАТИЧЕСКИ - это вызывает зависание!
        # Позиция останется на бирже и закроется при следующей проверке
        if position_to_close:
            logger.info(f" {symbol}: ⚠️ Позиция {position_to_close} осталась открытой на бирже (закроется автоматически)")
        
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
                logger.error(f" {symbol}: ❌ КРИТИЧНАЯ ОШИБКА! Статус НЕ изменен: {final_status}")
                bots_data['bots'][symbol]['status'] = BOT_STATUS['PAUSED']
                save_bots_state()
                logger.error(f" {symbol}: ✅ Статус принудительно исправлен на PAUSED")
            else:
                logger.info(f" {symbol}: ✅ Статус корректно установлен: {final_status}")
        
        logger.info(f" {symbol}: ✅ Кэш НЕ обновлен (статус PAUSED сохранен)")
        
        return jsonify({
            'success': True, 
            'message': f'Бот для {symbol} остановлен'
        })
        
    except Exception as e:
        logger.error(f" Ошибка остановки бота: {str(e)}")
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
                logger.info(f" {symbol}: Найдена открытая позиция {position_to_close}, будет закрыта при приостановке")
            
            bot_data['status'] = BOT_STATUS['PAUSED']
            logger.info(f" {symbol}: Бот приостановлен (был: {old_status})")
        
        # ⚠️ НЕ ЗАКРЫВАЕМ ПОЗИЦИЮ АВТОМАТИЧЕСКИ - это вызывает зависание!
        if position_to_close:
            logger.info(f" {symbol}: ⚠️ Позиция {position_to_close} осталась открытой на бирже (закроется автоматически)")
        
        return jsonify({
            'success': True,
            'message': f'Бот для {symbol} приостановлен'
        })
        
    except Exception as e:
        logger.error(f" Ошибка приостановки бота: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/delete', methods=['POST'])
def delete_bot_endpoint():
    """Удалить бота"""
    try:
        logger.info(f"📥 Получен запрос удаления бота: {request.get_data()}")
        
        # Пробуем разные способы получения данных
        try:
            data = request.get_json()
        except Exception as json_error:
            logger.error(f"❌ Ошибка парсинга JSON: {json_error}")
            # Пробуем получить данные как form data
            data = request.form.to_dict()
            if not data:
                # Пробуем получить данные из args
                data = request.args.to_dict()
        
        logger.info(f"📊 Распарсенные данные: {data}")
        
        if not data or not data.get('symbol'):
            logger.error(f"❌ Отсутствует symbol в данных: {data}")
            return jsonify({'success': False, 'error': 'Symbol required'}), 400
        
        symbol = data['symbol']
        reason = data.get('reason', 'Удален пользователем')
        
        # ⚡ БЕЗ БЛОКИРОВКИ: операции с dict атомарны в Python
        logger.info(f"🔍 Ищем бота {symbol} в bots_data. Доступные боты: {list(bots_data['bots'].keys())}")
        if symbol not in bots_data['bots']:
            logger.error(f"❌ Бот {symbol} не найден в bots_data")
            return jsonify({'success': False, 'error': 'Bot not found'}), 404
        
        # ✅ ТУПО УДАЛЯЕМ БОТА ИЗ ФАЙЛА!
        del bots_data['bots'][symbol]
        logger.info(f" {symbol}: Бот удален из файла")
        
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
        logger.error(f" Ошибка удаления бота: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/close-position', methods=['POST'])
def close_position_endpoint():
    """Принудительно закрыть позицию бота"""
    try:
        logger.info(f"📥 Получен запрос закрытия позиции: {request.get_data()}")
        
        # Пробуем разные способы получения данных
        try:
            data = request.get_json()
        except Exception as json_error:
            logger.error(f"❌ Ошибка парсинга JSON: {json_error}")
            # Пробуем получить данные как form data
            data = request.form.to_dict()
            if not data:
                # Пробуем получить данные из args
                data = request.args.to_dict()
        
        logger.info(f"📊 Распарсенные данные: {data}")
        
        if not data or not data.get('symbol'):
            logger.error(f"❌ Отсутствует symbol в данных: {data}")
            return jsonify({'success': False, 'error': 'Symbol required'}), 400
        
        symbol = data['symbol']
        force_close = data.get('force', False)  # Принудительное закрытие даже если бот не в позиции
        
        current_exchange = get_exchange()
        if not current_exchange:
            logger.error(f" ❌ Биржа не инициализирована")
            return jsonify({'success': False, 'error': 'Exchange not initialized'}), 500
        
        # ⚡ ИСПРАВЛЕНИЕ: Получаем актуальные позиции с биржи вместо кэша
        try:
            positions_response = current_exchange.get_positions()
            if isinstance(positions_response, tuple):
                positions = positions_response[0] if positions_response else []
            else:
                positions = positions_response if positions_response else []
        except Exception as e:
            logger.error(f" ❌ Ошибка получения позиций с биржи: {e}")
            positions = []
        
        # Ищем позиции для данного символа
        symbol_positions = []
        for pos in positions:
            if pos['symbol'] == f"{symbol}USDT" and float(pos.get('size', 0)) > 0:
                symbol_positions.append(pos)
        
        if not symbol_positions:
            logger.warning(f" ⚠️ Позиции для {symbol} не найдены на бирже")
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
                
                logger.info(f" 🔄 Закрываем позицию {position_side} размером {position_size} для {symbol}")
                
                close_result = current_exchange.close_position(
                    symbol=symbol,
                    size=position_size,
                    side=position_side,
                    order_type="Market"
                )
                
                if close_result and close_result.get('success'):
                    entry_price = float(pos.get('avg_price') or pos.get('entry_price') or 0)
                    exit_price = float(close_result.get('avg_price') or close_result.get('fill_price') or close_result.get('price') or 0)
                    if not exit_price and current_exchange and hasattr(current_exchange, 'get_ticker'):
                        try:
                            t = current_exchange.get_ticker(symbol)
                            if t and t.get('last'):
                                exit_price = float(t['last'])
                        except Exception:
                            pass
                    if not exit_price:
                        exit_price = float(pos.get('mark_price') or pos.get('current_price') or entry_price or 0)
                    closed_positions.append({
                        'side': position_side,
                        'size': position_size,
                        'order_id': close_result.get('order_id'),
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                    })
                    logger.info(f" ✅ Позиция {position_side} для {symbol} успешно закрыта")
                else:
                    error_msg = close_result.get('message', 'Unknown error') if close_result else 'No response'
                    errors.append(f"Позиция {position_side}: {error_msg}")
                    logger.error(f" ❌ Ошибка закрытия позиции {position_side} для {symbol}: {error_msg}")
                    
            except Exception as e:
                error_msg = f"Позиция {pos['side']}: {str(e)}"
                errors.append(error_msg)
                logger.error(f" ❌ Исключение при закрытии позиции {pos['side']} для {symbol}: {str(e)}")
        
        # КРИТИЧНО: Сохраняем сделку в bot_trades_history (иначе закрытия через UI не попадают в аналитику!)
        if closed_positions:
            try:
                from bot_engine.bots_database import get_bots_database
                bots_db = get_bots_database()
                with bots_data_lock:
                    bot_data = bots_data.get('bots', {}).get(symbol, {})
                for cp in closed_positions:
                    entry_price = float(cp.get('entry_price') or 0)
                    exit_price = float(cp.get('exit_price') or 0)
                    pos_size = float(cp.get('size') or 0)
                    direction = (cp.get('side') or 'LONG').upper()
                    if direction == 'SHORT':
                        pnl_usdt = (entry_price - exit_price) * pos_size
                    else:
                        pnl_usdt = (exit_price - entry_price) * pos_size
                    margin = bot_data.get('volume_value') or bot_data.get('margin_usdt')
                    roi_pct = (pnl_usdt / float(margin) * 100) if margin and float(margin) != 0 else 0
                    entry_ts = (datetime.now().timestamp() - 3600) * 1000
                    if bot_data.get('position_start_time'):
                        try:
                            dt = datetime.fromisoformat(str(bot_data['position_start_time']).replace('Z', '+00:00'))
                            ts = dt.timestamp()
                            entry_ts = ts * 1000 if ts < 1e10 else ts
                        except Exception:
                            pass
                    trade_data = {
                        'bot_id': bot_data.get('id') or symbol,
                        'symbol': symbol,
                        'direction': direction,
                        'entry_price': entry_price or 0,
                        'exit_price': exit_price or entry_price or 0,
                        'entry_time': bot_data.get('position_start_time') or datetime.now().isoformat(),
                        'exit_time': datetime.now().isoformat(),
                        'entry_timestamp': entry_ts,
                        'exit_timestamp': datetime.now().timestamp() * 1000,
                        'position_size_usdt': float(margin) if margin else None,
                        'position_size_coins': pos_size,
                        'pnl': pnl_usdt,
                        'roi': roi_pct,
                        'status': 'CLOSED',
                        'close_reason': 'MANUAL_CLOSE_UI',
                        'decision_source': bot_data.get('decision_source', 'SCRIPT'),
                        'ai_decision_id': bot_data.get('ai_decision_id'),
                        'ai_confidence': bot_data.get('ai_confidence'),
                        'entry_rsi': bot_data.get('entry_rsi'),
                        'exit_rsi': None,
                        'entry_trend': bot_data.get('entry_trend'),
                        'exit_trend': None,
                        'entry_volatility': None,
                        'entry_volume_ratio': None,
                        'is_successful': pnl_usdt > 0,
                        'is_simulated': False,
                        'source': 'bot',
                        'order_id': cp.get('order_id'),
                        'extra_data': {},
                    }
                    tid = bots_db.save_bot_trade_history(trade_data)
                    if tid:
                        logger.info(f" ✅ Закрытие через UI: сделка {symbol} сохранена в bots_data.db (ID: {tid})")
                    # FullAI аналитика: всегда записываем real_close при закрытии (для полноты журнала событий)
                    try:
                        from bots_modules.fullai_adaptive import record_real_close
                        extra = {
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl_usdt': pnl_usdt,
                            'direction': direction,
                        }
                        record_real_close(symbol, roi_pct, reason='MANUAL_CLOSE_UI', extra=extra)
                    except Exception as fa_err:
                        logger.debug("FullAI analytics real_close UI: %s", fa_err)
            except Exception as save_err:
                logger.warning(f" ⚠️ Ошибка сохранения сделки в bot_trades_history: {save_err}")

        # Обновляем данные бота, если он существует
        with bots_data_lock:
            if symbol in bots_data['bots']:
                bot_data = bots_data['bots'][symbol]
                if closed_positions:
                    bot_data['position_side'] = None
                    bot_data['unrealized_pnl'] = 0.0
                    bot_data['status'] = BOT_STATUS['IDLE']
                    logger.info(f" 🔄 Обновлены данные бота {symbol} после закрытия позиций")
                
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
        logger.error(f" Ошибка закрытия позиций: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Словарь человекочитаемых названий параметров конфигурации
CONFIG_NAMES = {
    # Auto Bot Configuration
    'enabled': 'Auto Bot включен',
    'max_concurrent': 'Максимум одновременных ботов',
    'risk_cap_percent': 'Риск (% от депозита)',
    'scope': 'Область действия',
    'default_position_size': 'Размер позиции по умолчанию',
    'default_position_mode': 'Режим размера позиции по умолчанию',
    'rsi_long_threshold': 'RSI порог для LONG',
    'rsi_short_threshold': 'RSI порог для SHORT',
    'rsi_exit_long_with_trend': 'RSI выход LONG (по тренду)',
    'rsi_exit_long_against_trend': 'RSI выход LONG (против тренда)',
    'rsi_exit_short_with_trend': 'RSI выход SHORT (по тренду)',
    'rsi_exit_short_against_trend': 'RSI выход SHORT (против тренда)',
    'rsi_exit_min_candles': 'Мин. свечей до выхода по RSI',
    'rsi_exit_min_minutes': 'Мин. минут до выхода по RSI (адаптивно по ТФ)',
    'rsi_exit_min_move_percent': 'Мин. % движения для выхода по RSI (блокирует до достижения)',
    'rsi_limit_entry_enabled': 'Вход лимитом по цене RSI (расчёт цены по порогу)',
    'rsi_limit_exit_enabled': 'Выход лимитом по цене RSI',
    'rsi_limit_offset_percent': 'Смещение лимита входа по RSI (%)',
    'rsi_limit_exit_offset_percent': 'Смещение лимита выхода по RSI (%)',
    'rsi_time_filter_enabled': 'RSI Time Filter (фильтр по времени)',
    'rsi_time_filter_candles': 'RSI Time Filter - количество свечей',
    'rsi_time_filter_lower': 'RSI Time Filter - нижний порог',
    'rsi_time_filter_upper': 'RSI Time Filter - верхний порог',
    'avoid_down_trend': 'Фильтр DOWN тренда для LONG',
    'avoid_up_trend': 'Фильтр UP тренда для SHORT',
    'trend_detection_enabled': 'Определение тренда включено',
    'trend_analysis_period': 'Период анализа тренда',
    'trend_candles_threshold': 'Порог свечей для тренда',
    'trend_price_change_threshold': 'Порог изменения цены для тренда',
    'min_candles_for_maturity': 'Минимум свечей для зрелости',
    'min_rsi_low': 'Минимальный RSI Low для зрелости',
    'max_rsi_high': 'Максимальный RSI High для зрелости',
    'enable_maturity_check': 'Проверка зрелости включена',
    'take_profit_percent': 'Take Profit (%)',
    'max_loss_percent': 'Stop Loss (%)',
    'leverage': 'Плечо',
    'check_interval': 'Интервал проверки (сек)',
    'monitoring_interval': 'Интервал мониторинга (сек)',
    'trading_enabled': 'Торговля включена',
    'use_test_server': 'Тестовый сервер',
    'trailing_stop_activation': 'Активация трейлинг-стопа (%)',
    'trailing_stop_distance': 'Дистанция трейлинг-стопа (%)',
    'trailing_take_distance': 'Дистанция трейлинг-тейка (%)',
    'trailing_update_interval': 'Интервал обновления трейлинга (сек)',
    'max_position_hours': 'Максимальное время позиции (часы)',
    'break_even_protection': 'Защита безубыточности',
    'break_even_trigger': 'Триггер безубыточности (%)',
    'break_even_trigger_percent': 'Триггер безубыточности (%)',
    'exit_scam_enabled': 'Exit Scam защита',
    'exit_scam_auto_learn_enabled': 'Автоподбор ExitScam по истории',
    'exit_scam_candles': 'Exit Scam - количество свечей',
    'exit_scam_single_candle_percent': 'Exit Scam - % одной свечи',
    'exit_scam_multi_candle_count': 'Exit Scam - количество свечей (множественный)',
    'exit_scam_multi_candle_percent': 'Exit Scam - % множественных свечей',
    'limit_orders_entry_enabled': 'Лимитные ордера включены',
    'limit_orders_percent_steps': 'Шаги лимитных ордеров (%)',
    'limit_orders_margin_amounts': 'Суммы маржи лимитных ордеров',
    'ai_enabled': 'AI подтверждение включено',
    'ai_min_confidence': 'Минимальная уверенность AI',
    'ai_override_original': 'AI блокирует скрипт',
    'ai_optimal_entry_enabled': 'AI оптимальный вход',
    'min_volatility_threshold': 'Минимальный порог волатильности',
    
    # System Configuration
    'rsi_update_interval': 'Интервал обновления RSI (сек)',
    'auto_save_interval': 'Интервал автосохранения (сек)',
    'debug_mode': 'Режим отладки',
    'refresh_interval': 'Интервал обновления UI (сек)',
    'position_sync_interval': 'Интервал синхронизации позиций (сек)',  # также задаёт интервал обновления миниграфиков
    'inactive_bot_cleanup_interval': 'Интервал очистки неактивных ботов (сек)',
    'inactive_bot_timeout': 'Таймаут неактивного бота (сек)',
    'stop_loss_setup_interval': 'Интервал установки Stop Loss (сек)',
    'enhanced_rsi_enabled': 'Улучшенная система RSI',
    'enhanced_rsi_require_volume_confirmation': 'Подтверждение объемом',
    'enhanced_rsi_require_divergence_confirmation': 'Строгий режим (дивергенции)',
    'enhanced_rsi_use_stoch_rsi': 'Использовать Stochastic RSI',
    'rsi_extreme_zone_timeout': 'Таймаут экстремальной зоны RSI',
    'rsi_extreme_oversold': 'Экстремальная перепроданность RSI',
    'rsi_extreme_overbought': 'Экстремальная перекупленность RSI',
    'rsi_volume_confirmation_multiplier': 'Множитель подтверждения объема',
    'rsi_divergence_lookback': 'Период поиска дивергенций',
    'anomaly_detection_enabled': 'Обнаружение аномалий',
    'anomaly_block_threshold': 'Порог блокировки аномалий',
    'anomaly_log_enabled': 'Логирование аномалий',
    'risk_management_enabled': 'Управление рисками',
    'risk_update_interval': 'Интервал обновления рисков (сек)',
    'lstm_enabled': 'LSTM модель',
    'lstm_min_confidence': 'Минимальная уверенность LSTM',
    'lstm_weight': 'Вес LSTM',
    'pattern_enabled': 'Обнаружение паттернов',
    'pattern_min_confidence': 'Минимальная уверенность паттернов',
    'pattern_weight': 'Вес паттернов',
    'auto_train_enabled': 'Автообучение',
    'auto_update_data': 'Автообновление данных',
    'data_update_interval': 'Интервал обновления данных (сек)',
    'auto_retrain': 'Автопереобучение',
    'retrain_interval': 'Интервал переобучения (сек)',
    'retrain_hour': 'Час переобучения',
    'log_predictions': 'Логирование предсказаний',
    'log_anomalies': 'Логирование аномалий',
    'log_patterns': 'Логирование паттернов',
}

def log_config_change(key, old_value, new_value, description=""):
    """Логирует изменение конфигурации только если значение изменилось"""
    if old_value != new_value:
        arrow = '→'
        # Используем понятное название из словаря или техническое название
        display_name = description or CONFIG_NAMES.get(key, key)
        # Используем logger для записи в файл и консоль
        logger.info(f"✓ {display_name}: {old_value} {arrow} {new_value}")
        return True
    return False

@bots_app.route('/api/bots/timeframe', methods=['GET', 'POST'])
def timeframe_config():
    """Получить или установить текущий таймфрейм системы"""
    try:
        from bot_engine.config_loader import get_current_timeframe, set_current_timeframe, reset_timeframe_to_config
        
        if request.method == 'GET':
            current_tf = get_current_timeframe()
            return jsonify({
                'success': True,
                'timeframe': current_tf,
                'supported_timeframes': ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
            })
        
        elif request.method == 'POST':
            data = request.get_json()
            if not data or 'timeframe' not in data:
                return jsonify({'success': False, 'error': 'timeframe parameter is required'}), 400
            
            new_timeframe = data['timeframe']
            old_timeframe = get_current_timeframe()
            
            # Устанавливаем новый таймфрейм в память
            success = set_current_timeframe(new_timeframe)
            if not success:
                return jsonify({
                    'success': False,
                    'error': f'Unsupported timeframe: {new_timeframe}'
                }), 400

            # Единый конфиг: сохраняем таймфрейм в configs/bot_config.py (AutoBotConfig + SystemConfig)
            try:
                from bots_modules.config_writer import save_auto_bot_config_current_to_py, save_system_config_to_py
                save_auto_bot_config_current_to_py({'system_timeframe': new_timeframe})
                save_system_config_to_py({'SYSTEM_TIMEFRAME': new_timeframe})
                logger.info(f"✅ Таймфрейм сохранён в конфиг: {new_timeframe}")
            except Exception as save_config_err:
                logger.warning(f"⚠️ Не удалось сохранить таймфрейм в конфиг: {save_config_err}")
            
            logger.info(f"🔄 Таймфрейм изменен: {old_timeframe} → {new_timeframe}")
            
            # Сохраняем текущие данные перед переключением
            try:
                from bots_modules.sync_and_cache import save_rsi_cache
                from bots_modules.imports_and_globals import coins_rsi_data, rsi_data_lock
                with rsi_data_lock:
                    if coins_rsi_data.get('coins'):
                        # Сохраняем текущий кэш
                        save_rsi_cache()
            except Exception as save_err:
                logger.warning(f"⚠️ Не удалось сохранить RSI кэш при переключении таймфрейма: {save_err}")
            
            # Очищаем кэш свечей для перезагрузки с новым таймфреймом
            try:
                from bots_modules.imports_and_globals import coins_rsi_data, rsi_data_lock
                with rsi_data_lock:
                    coins_rsi_data['candles_cache'] = {}
                    coins_rsi_data['last_candles_update'] = None
                    coins_rsi_data['last_update'] = None
                    coins_rsi_data['coins'] = {}
                    coins_rsi_data['update_in_progress'] = False  # Сбрасываем, чтобы continuous loader смог рассчитать RSI
                    logger.info("🗑️ Кэш свечей и RSI данных очищен для перезагрузки с новым таймфреймом")
            except Exception as clear_err:
                logger.warning(f"⚠️ Не удалось очистить кэш свечей: {clear_err}")
            
            # Не запускаем отдельный поток load_all_coins_rsi: кэш пуст, поток бы делал 500+ API вызовов
            # и блокировал continuous_data_loader. Он сам загрузит свечи и RSI на следующем раунде.
            
            return jsonify({
                'success': True,
                'message': f'Таймфрейм изменен с {old_timeframe} на {new_timeframe}. Continuous loader загрузит свечи и RSI на следующем раунде.',
                'old_timeframe': old_timeframe,
                'new_timeframe': new_timeframe
            })
    
    except Exception as e:
        logger.error(f"❌ Ошибка работы с таймфреймом: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def _build_full_export_config():
    """Собирает полный конфиг для экспорта из текущего состояния в памяти (без перезагрузки с диска)."""
    # Не вызываем load_auto_bot_config/load_system_config — иначе при экспорте память перезапишется с диска и конфиг «станет дефолтным»
    with bots_data_lock:
        auto_bot = deepcopy(bots_data.get('auto_bot_config', {}))
    system_cfg = get_system_config_snapshot()
    try:
        from bot_engine.config_loader import get_current_timeframe
        tf = get_current_timeframe()
    except Exception:
        raise RuntimeError("КРИТИЧЕСКАЯ ОШИБКА: таймфрейм не выбран или недоступен. Выберите таймфрейм в настройках.")
    if not tf or not str(tf).strip():
        raise RuntimeError("КРИТИЧЕСКАЯ ОШИБКА: таймфрейм не выбран или недоступен. Выберите таймфрейм в настройках.")
    system_cfg = dict(system_cfg) if system_cfg else {}
    system_cfg['timeframe'] = tf
    ai_cfg = {}
    try:
        from bot_engine.config_loader import AIConfig, RiskConfig
        ai_cfg = {
            'ai_enabled': getattr(AIConfig, 'AI_ENABLED', False),
            'ai_confidence_threshold': getattr(AIConfig, 'AI_CONFIDENCE_THRESHOLD', 0.65),
            'anomaly_detection_enabled': getattr(AIConfig, 'AI_ANOMALY_DETECTION_ENABLED', True),
            'anomaly_block_threshold': getattr(AIConfig, 'AI_ANOMALY_BLOCK_THRESHOLD', 0.7),
            'anomaly_log_enabled': getattr(AIConfig, 'AI_LOG_ANOMALIES', True),
            'risk_management_enabled': getattr(AIConfig, 'AI_RISK_MANAGEMENT_ENABLED', True),
            'risk_update_interval': getattr(AIConfig, 'AI_RISK_UPDATE_INTERVAL', 300),
            'optimal_entry_enabled': getattr(RiskConfig, 'AI_OPTIMAL_ENTRY_ENABLED', True),
            'lstm_enabled': getattr(AIConfig, 'AI_LSTM_ENABLED', True),
            'lstm_min_confidence': getattr(AIConfig, 'AI_LSTM_MIN_CONFIDENCE', 0.6),
            'lstm_weight': getattr(AIConfig, 'AI_LSTM_WEIGHT', 1.5),
            'pattern_enabled': getattr(AIConfig, 'AI_PATTERN_ENABLED', True),
            'pattern_min_confidence': getattr(AIConfig, 'AI_PATTERN_MIN_CONFIDENCE', 0.6),
            'pattern_weight': getattr(AIConfig, 'AI_PATTERN_WEIGHT', 1.2),
            'auto_train_enabled': getattr(AIConfig, 'AI_AUTO_TRAIN_ENABLED', True),
            'auto_update_data': getattr(AIConfig, 'AI_AUTO_UPDATE_DATA', True),
            'auto_retrain': getattr(AIConfig, 'AI_AUTO_RETRAIN', True),
            'data_update_interval': getattr(AIConfig, 'AI_DATA_UPDATE_INTERVAL', 86400),
            'retrain_interval': getattr(AIConfig, 'AI_RETRAIN_INTERVAL', 604800),
            'retrain_hour': getattr(AIConfig, 'AI_RETRAIN_HOUR', 3),
            'update_coins_count': getattr(AIConfig, 'AI_UPDATE_COINS_COUNT', 50),
            'log_predictions': getattr(AIConfig, 'AI_LOG_PREDICTIONS', True),
            'log_anomalies': getattr(AIConfig, 'AI_LOG_ANOMALIES', True),
            'log_patterns': getattr(AIConfig, 'AI_LOG_PATTERNS', True),
            'self_learning_enabled': getattr(AIConfig, 'AI_SELF_LEARNING_ENABLED', True),
            'smc_enabled': getattr(AIConfig, 'AI_SMC_ENABLED', True),
        }
    except Exception as e:
        logger.warning(f" export-config AI: {e}")
    return {'autoBot': auto_bot, 'system': system_cfg, 'ai': ai_cfg}, tf


@bots_app.route('/api/bots/export-config', methods=['GET'])
def export_full_config():
    """Полный конфиг для экспорта в JSON: autoBot + system + ai. Имя файла: config_<timeframe>.json."""
    try:
        full_config, timeframe = _build_full_export_config()
        return jsonify({
            'success': True,
            'config': full_config,
            'timeframe': timeframe
        })
    except Exception as e:
        logger.exception("export-config")
        return jsonify({'success': False, 'error': str(e)}), 500


@bots_app.route('/api/bots/import-config', methods=['POST'])
def import_full_config():
    """
    Импорт конфига из JSON (формат экспорта: { autoBot, system?, ai? } или { config: { autoBot, ... } }).
    Применяет все переданные блоки одним запросом и сохраняет в файл и БД.
    """
    try:
        data = request.get_json(silent=True) or {}
        if not data:
            return jsonify({'success': False, 'error': 'Нет данных в теле запроса'}), 400
        config = data.get('config') if isinstance(data.get('config'), dict) else data
        auto_bot = config.get('autoBot') if isinstance(config.get('autoBot'), dict) else None
        system_cfg = config.get('system') if isinstance(config.get('system'), dict) else None
        ai_cfg = config.get('ai') if isinstance(config.get('ai'), dict) else None
        if not auto_bot and not system_cfg and not ai_cfg:
            return jsonify({'success': False, 'error': 'В теле должны быть autoBot, system и/или ai'}), 400

        applied = {'autoBot': False, 'system': False, 'ai': False}
        errors = []

        if auto_bot:
            try:
                from bots_modules.config_writer import CONFIG_KEY_ALIASES
                with bots_data_lock:
                    # Мержим с текущим конфигом: только переданные ключи обновляются, остальные не трогаем.
                    # Старые имена ключей из импорта маппятся на текущие (обратная совместимость).
                    for key, value in auto_bot.items():
                        canonical = CONFIG_KEY_ALIASES.get(key, key)
                        bots_data['auto_bot_config'][canonical] = value
                # Фильтры в БД
                from bot_engine.bots_database import get_bots_database
                db = get_bots_database()
                w = auto_bot.get('whitelist')
                b = auto_bot.get('blacklist')
                s = auto_bot.get('scope')
                if w is not None or b is not None or s is not None:
                    db.save_coin_filters(whitelist=w, blacklist=b, scope=s)
                save_auto_bot_config()
                _patch_ai_config_after_auto_bot_save(auto_bot)
                applied['autoBot'] = True
                logger.info(f"[API] ✅ Импорт: Auto Bot применён ({len(auto_bot)} параметров), сохранён в файл и БД")
            except Exception as e:
                logger.exception("import-config autoBot")
                errors.append(f"Auto Bot: {e}")

        if system_cfg and not errors:
            try:
                from bots_modules.sync_and_cache import save_system_config, load_system_config
                save_system_config(system_cfg)
                load_system_config()
                applied['system'] = True
                logger.info(f"[API] ✅ Импорт: System применён ({len(system_cfg)} параметров)")
            except Exception as e:
                logger.warning(f"[API] Импорт system: {e}")
                errors.append(f"System: {e}")

        if ai_cfg and not errors:
            try:
                from bot_engine.config_loader import reload_config
                from bot_engine.config_loader import AIConfig, RiskConfig
                for key, value in ai_cfg.items():
                    try:
                        attr = key.upper() if isinstance(key, str) else key
                        if hasattr(AIConfig, attr):
                            setattr(AIConfig, attr, value)
                        if hasattr(RiskConfig, attr):
                            setattr(RiskConfig, attr, value)
                    except Exception:
                        pass
                reload_config()
                applied['ai'] = True
                logger.info(f"[API] ✅ Импорт: AI применён ({len(ai_cfg)} параметров)")
            except Exception as e:
                logger.warning(f"[API] Импорт AI: {e}")
                errors.append(f"AI: {e}")

        if errors:
            return jsonify({
                'success': False,
                'error': '; '.join(errors),
                'applied': applied
            }), 500
        return jsonify({
            'success': True,
            'applied': applied,
            'message': 'Конфигурация импортирована и сохранена в файл'
        })
    except Exception as e:
        logger.exception("import-config")
        return jsonify({'success': False, 'error': str(e)}), 500


@bots_app.route('/api/bots/system-config', methods=['GET', 'POST'])
def system_config():
    """Получить или обновить системные настройки"""
    # Константы теперь в SystemConfig
    try:
        if request.method == 'GET':
            try:
                load_system_config()
            except Exception as load_err:
                logger.warning(f" ⚠️ Не удалось перезагрузить системную конфигурацию перед GET: {load_err}")
            
            # Добавляем текущий таймфрейм в системные настройки
            config = get_system_config_snapshot()
            from bot_engine.config_loader import get_current_timeframe
            config['timeframe'] = get_current_timeframe()
            
            return jsonify({
                'success': True,
                'config': config
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
                    # ❌ ОТКЛЮЧЕНО: Smart RSI Manager больше не используется
                    # Continuous Data Loader обновляет данные постоянно
            
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
            
            # auto_refresh_ui всегда True (настройка убрана из UI)
            if 'refresh_interval' in data:
                old_value = SystemConfig.UI_REFRESH_INTERVAL
                new_value = int(data['refresh_interval'])
                if log_config_change('refresh_interval', old_value, new_value):
                    SystemConfig.UI_REFRESH_INTERVAL = new_value
                    changes_count += 1
            
            # mini_chart_update_interval привязан к position_sync_interval (настройка убрана из UI)
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
                # Единый период для всех RSI-зависимых данных в UI
                if hasattr(SystemConfig, 'UI_REFRESH_INTERVAL'):
                    SystemConfig.UI_REFRESH_INTERVAL = new_value
            
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

        if 'bybit_margin_mode' in data:
            raw = (data.get('bybit_margin_mode') or 'auto').strip().lower()
            new_value = raw if raw in ('auto', 'cross', 'isolated') else 'auto'
            old_value = getattr(SystemConfig, 'BYBIT_MARGIN_MODE', 'auto')
            if log_config_change('bybit_margin_mode', old_value, new_value):
                SystemConfig.BYBIT_MARGIN_MODE = new_value
                system_changes_count += 1

        system_config_data = get_system_config_snapshot()
        saved_to_file = save_system_config(system_config_data)
        
        if changes_count > 0:
            logger.info(f"✅ Изменено параметров: {changes_count}, конфигурация сохранена")
        else:
            logger.info("ℹ️  Изменений не обнаружено")
        
        if system_changes_count > 0:
            logger.info(f"✅ System config: изменено параметров: {system_changes_count}, конфигурация сохранена")
        else:
            logger.info("ℹ️  System config: изменений не обнаружено")
        
        # Таймфрейм хранится только в конфиге; при перезагрузке берётся из файла
        if saved_to_file and (changes_count > 0 or system_changes_count > 0):
            load_system_config()

        return jsonify({
            'success': True,
            'message': 'Системные настройки обновлены и сохранены',
            'config': system_config_data,
            'saved_to_file': saved_to_file
        })
        
    except Exception as e:
        logger.error(f" Ошибка настройки системы: {str(e)}")
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
        logger.error(f" ❌ Ошибка принудительной синхронизации: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bots_app.route('/api/bots/cleanup-inactive', methods=['POST'])
def cleanup_inactive_manual():
    """Принудительная очистка неактивных ботов"""
    try:
        logger.info(" 🧹 Запуск принудительной очистки неактивных ботов")
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
        logger.error(f" ❌ Ошибка принудительной очистки: {e}")
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
        logger.error(f" ❌ Ошибка получения списка зрелых монет: {e}")
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
        logger.error(f" ❌ Ошибка API удаления монет: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bots_app.route('/api/bots/smart-rsi-status', methods=['GET'])
def get_smart_rsi_status():
    """Получить статус Smart RSI Manager (LEGACY - теперь используется Continuous Data Loader)"""
    try:
        # ❌ Smart RSI Manager отключен, вместо него работает Continuous Data Loader
        from bots_modules.continuous_data_loader import get_continuous_loader
        loader = get_continuous_loader()
        
        if loader:
            status = loader.get_status()
            return jsonify({
                'success': True,
                'status': {
                    'active': True,
                    'service': 'Continuous Data Loader',
                    'is_running': status['is_running'],
                    'update_count': status['update_count'],
                    'last_update': status['last_update'],
                    'note': 'Smart RSI Manager заменен на Continuous Data Loader'
                }
            })
        
        return jsonify({
            'success': False,
            'error': 'Continuous Data Loader не инициализирован'
        }), 500
        
    except Exception as e:
        logger.error(f" ❌ Ошибка получения статуса Smart RSI Manager: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bots_app.route('/api/bots/force-rsi-update', methods=['POST'])
def force_rsi_update():
    """Принудительно обновить RSI данные"""
    try:
        logger.info(" 🔄 Принудительное обновление RSI данных...")
        
        # Запускаем обновление RSI данных в отдельном потоке
        import threading
        def update_rsi():
            try:
                load_all_coins_rsi()
                logger.info(" ✅ RSI данные обновлены принудительно")
            except Exception as e:
                logger.error(f" ❌ Ошибка принудительного обновления RSI: {e}")
        
        thread = threading.Thread(target=update_rsi)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Обновление RSI данных запущено'
        })
        
    except Exception as e:
        logger.error(f" Ошибка принудительного обновления RSI: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/test-exit-scam/<symbol>', methods=['GET'])
def test_exit_scam_endpoint(symbol):
    """Тестирует ExitScam фильтр для конкретной монеты"""
    try:
        test_exit_scam_filter(symbol)
        return jsonify({'success': True, 'message': f'Тест ExitScam фильтра для {symbol} выполнен'})
    except Exception as e:
        logger.error(f" Ошибка тестирования ExitScam фильтра для {symbol}: {e}")
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
        logger.error(f" Ошибка тестирования RSI временного фильтра для {symbol}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/reload-modules', methods=['POST'])
def reload_modules_endpoint():
    """Перезагружает модули без перезапуска сервера"""
    try:
        import importlib
        import sys
        
        # Объявляем глобальные переменные в начале функции
        global exchange, system_initialized
        
        logger.info(" 🔄 Начинаем перезагрузку модулей...")
        
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
                    logger.info(f" Перезагружен модуль: {module_name}")
            except Exception as e:
                logger.warning(f" Не удалось перезагрузить {module_name}: {e}")
        
        # Восстанавливаем важные переменные
        if saved_exchange:
            exchange = saved_exchange
            logger.info(" ✅ Восстановлена переменная exchange")
        
        if saved_system_initialized:
            system_initialized = saved_system_initialized
            logger.info(" ✅ Восстановлен флаг system_initialized")
        
        logger.info(f" ✅ Перезагружено {reloaded_count} модулей")
        
        return jsonify({
            'success': True, 
            'message': f'Перезагружено {reloaded_count} модулей',
            'reloaded_modules': reloaded_count
        })
        
    except Exception as e:
        logger.error(f" Ошибка перезагрузки модулей: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/refresh-rsi/<symbol>', methods=['POST'])
def refresh_rsi_for_coin(symbol):
    """Обновляет RSI данные для конкретной монеты (применяет новую логику)"""
    try:
        global coins_rsi_data
        
        # Символ "all" не является торговой парой — не вызываем get_coin_rsi_data (биржа вернёт Symbol Is Invalid)
        if not symbol or str(symbol).strip().lower() == 'all':
            logger.info(" 🔄 Обновление RSI для 'all': перенаправление на полное обновление или отказ")
            return jsonify({
                'success': False,
                'error': 'Для обновления всех монет используйте полное обновление RSI (refresh-rsi-all). Символ "all" не поддерживается API биржи.'
            }), 400

        logger.info(f" 🔄 Обновление RSI данных для {symbol}...")
        
        # Проверяем биржу
        if not ensure_exchange_initialized():
            return jsonify({'success': False, 'error': 'Биржа не инициализирована'}), 500
        
        # Получаем новые данные монеты
        coin_data = get_coin_rsi_data(symbol, get_exchange())
        
        if coin_data:
            # Обновляем в глобальном кэше
            with rsi_data_lock:
                coins_rsi_data['coins'][symbol] = coin_data
            
            logger.info(f" ✅ RSI данные для {symbol} обновлены")
            
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
        logger.error(f" Ошибка обновления RSI для {symbol}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/rsi-history/<symbol>', methods=['GET'])
def get_rsi_history_for_chart(symbol):
    """Получить историю RSI для графика из кэша (без запроса к бирже).
    current_rsi для миниграфиков берётся из coins_rsi_data['coins'][symbol] — тот же источник, что и для
    ботов; для символов в позиции эти данные обновляются в sync_positions (_refresh_rsi_for_bots_in_position)."""
    try:
        from bots_modules.calculations import calculate_rsi_history
        from bot_engine.config_loader import get_rsi_from_coin_data

        # ✅ СНАЧАЛА ПРОВЕРЯЕМ КЭШ В ПАМЯТИ, ПОТОМ БД (тот же источник, что и для ботов)
        # ✅ ОПТИМИЗАЦИЯ: Поддержка новой структуры кэша (несколько таймфреймов)
        candles = None
        candles_cache = coins_rsi_data.get('candles_cache', {})
        from bot_engine.config_loader import get_current_timeframe
        current_timeframe = get_current_timeframe()
        
        if symbol in candles_cache:
            symbol_cache = candles_cache[symbol]
            # Новая структура: {timeframe: {candles: [...], ...}}
            if isinstance(symbol_cache, dict) and current_timeframe in symbol_cache:
                cached_data = symbol_cache[current_timeframe]
                candles = cached_data.get('candles')
            # Старая структура (обратная совместимость)
            elif isinstance(symbol_cache, dict) and 'candles' in symbol_cache:
                cached_timeframe = symbol_cache.get('timeframe')
                if cached_timeframe == current_timeframe:
                    candles = symbol_cache.get('candles')
        
        # Если нет в памяти, читаем из БД
        if not candles:
            try:
                from bot_engine.storage import get_candles_for_symbol
                db_cached_data = get_candles_for_symbol(symbol)
                if db_cached_data:
                    candles = db_cached_data.get('candles', [])
            except Exception as e:
                pass
        
        if not candles or len(candles) < 15:
            return jsonify({
                'success': False,
                'error': f'Недостаточно свечей для расчета RSI (требуется минимум 15)'
            }), 400
        
        # Берем последние 56 свечей для графика
        candles = candles[-56:] if len(candles) >= 56 else candles
        
        # Извлекаем цены закрытия
        closes = [float(candle['close']) for candle in candles]
        
        # Рассчитываем историю RSI
        rsi_history = calculate_rsi_history(closes, period=14)
        
        if not rsi_history:
            return jsonify({
                'success': False,
                'error': 'Не удалось рассчитать историю RSI'
            }), 500
        
        # Берем последние 56 значений RSI
        rsi_values = rsi_history[-56:] if len(rsi_history) > 56 else rsi_history
        
        # Текущее RSI для миниграфиков: из coins_rsi_data['coins'][symbol] (тот же источник, что и для ботов;
        # для символов в позиции обновляется в sync_positions → _refresh_rsi_for_bots_in_position)
        current_rsi = None
        coin_data = coins_rsi_data.get('coins', {}).get(symbol)
        if coin_data:
            current_rsi = get_rsi_from_coin_data(coin_data, timeframe=current_timeframe)
        if current_rsi is None and rsi_values:
            current_rsi = rsi_values[-1]

        return jsonify({
            'success': True,
            'rsi_history': rsi_values,
            'current_rsi': round(current_rsi, 2) if current_rsi is not None else None,
            'candles_count': len(candles),
            'source': 'cache'  # candles_cache + coins (current_rsi из coins = тот же источник, что для ботов и миниграфиков)
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения истории RSI для {symbol}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/candles/<symbol>', methods=['GET'])
def get_candles_from_cache(symbol):
    """Получить свечи из кэша или файла (без запроса к бирже)"""
    try:
        # Получаем параметры запроса
        # Получаем текущий таймфрейм из конфига
        from bot_engine.config_loader import get_current_timeframe
        timeframe = request.args.get('timeframe', get_current_timeframe())  # По умолчанию текущий таймфрейм
        period_days = request.args.get('period', None)  # Опционально, для совместимости
        
        # ✅ СНАЧАЛА ПРОВЕРЯЕМ КЭШ В ПАМЯТИ (по таймфрейму: symbol[timeframe]['candles']), ПОТОМ БД
        candles_6h = None
        candles_cache = coins_rsi_data.get('candles_cache', {})
        if symbol in candles_cache:
            cached_data = candles_cache[symbol]
            if isinstance(cached_data, dict):
                # Структура: symbol -> timeframe -> {'candles': [...]}
                candles_6h = cached_data.get(timeframe, {}).get('candles') or cached_data.get('candles')
            else:
                candles_6h = None
        
        # Если нет в памяти, читаем из БД
        if not candles_6h:
            try:
                from bot_engine.storage import get_candles_for_symbol
                db_cached_data = get_candles_for_symbol(symbol)
                if db_cached_data:
                    candles_6h = db_cached_data.get('candles', [])
            except Exception as e:
                pass
        
        if not candles_6h:
            return jsonify({
                'success': False,
                'error': f'Свечи для {symbol} не найдены в кэше или БД'
            }), 404
        
        # Конвертируем свечи в нужный таймфрейм
        if timeframe == '1d':
            # Конвертируем 6h свечи в дневные
            daily_candles = []
            current_day = None
            current_candle = None
            
            for candle in candles_6h:
                candle_time = datetime.fromtimestamp(int(candle['timestamp']) / 1000)
                day_key = candle_time.date()
                
                if current_day != day_key:
                    # Сохраняем предыдущую дневную свечу
                    if current_candle:
                        daily_candles.append(current_candle)
                    
                    # Начинаем новую дневную свечу
                    current_day = day_key
                    current_candle = {
                        'timestamp': candle['timestamp'],
                        'open': float(candle['open']),
                        'high': float(candle['high']),
                        'low': float(candle['low']),
                        'close': float(candle['close']),
                        'volume': float(candle.get('volume', 0))
                    }
                else:
                    # Обновляем текущую дневную свечу
                    if current_candle:
                        current_candle['high'] = max(current_candle['high'], float(candle['high']))
                        current_candle['low'] = min(current_candle['low'], float(candle['low']))
                        current_candle['close'] = float(candle['close'])
                        current_candle['volume'] += float(candle.get('volume', 0))
            
            # Добавляем последнюю свечу
            if current_candle:
                daily_candles.append(current_candle)
            
            candles = daily_candles
        # Получаем текущий таймфрейм из конфига
        from bot_engine.config_loader import get_current_timeframe
        current_timeframe = get_current_timeframe()
        
        if timeframe == current_timeframe:
            # Используем свечи текущего таймфрейма как есть
            candles = candles_6h
        elif timeframe == '1d':
            # Конвертируем свечи текущего таймфрейма в дневные (если нужно)
            # Пока возвращаем свечи текущего таймфрейма
            candles = candles_6h
        else:
            # Для других таймфреймов возвращаем свечи текущего таймфрейма
            candles = candles_6h
        
        # Ограничиваем количество свечей по периоду (если указан)
        if period_days:
            try:
                period_days = int(period_days)
                # Для дневных свечей: period_days свечей
                # Для 6h свечей: period_days * 4 свечей (4 свечи в день)
                if timeframe == '1d':
                    candles = candles[-period_days:] if len(candles) > period_days else candles
                else:
                    candles = candles[-period_days * 4:] if len(candles) > period_days * 4 else candles
            except:
                pass
        
        # Форматируем ответ (поддержка и 'time', и 'timestamp' из кэша)
        formatted_candles = []
        for candle in candles:
            ts = int(candle.get('time', candle.get('timestamp', 0)))
            formatted_candles.append({
                'timestamp': ts,
                'time': ts,
                'open': str(candle.get('open', 0)),
                'high': str(candle.get('high', 0)),
                'low': str(candle.get('low', 0)),
                'close': str(candle.get('close', 0)),
                'volume': str(candle.get('volume', 0))
            })
        
        return jsonify({
            'success': True,
            'data': {
                'candles': formatted_candles,
                'timeframe': timeframe,
                'count': len(formatted_candles)
            },
            'source': 'cache'  # Указываем, что данные из кэша
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения свечей из кэша для {symbol}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/refresh-rsi-all', methods=['POST'])
def refresh_rsi_for_all_coins():
    """Обновляет RSI данные для всех монет (применяет новую логику)"""
    try:
        global coins_rsi_data
        
        logger.info(" 🔄 Обновление RSI данных для всех монет...")
        
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
                logger.warning(f" Ошибка обновления {symbol}: {e}")
                failed_count += 1
        
        logger.info(f" ✅ Обновлено {updated_count} монет, ошибок: {failed_count}")
        
        return jsonify({
            'success': True,
            'message': f'RSI данные обновлены для {updated_count} монет',
            'updated_count': updated_count,
            'failed_count': failed_count
        })
        
    except Exception as e:
        logger.error(f" Ошибка обновления всех RSI данных: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/restart-service', methods=['POST'])
def restart_service_endpoint():
    """Перезапускает сервис ботов (только основные компоненты)"""
    try:
        logger.info(" 🔄 Перезапуск сервиса ботов...")
        
        # Перезагружаем глобальные переменные
        global exchange, coins_rsi_data, bots_data
        global system_initialized
        
        # Сбрасываем флаг инициализации
        system_initialized = False
        logger.info(" 🔄 Сброшен флаг инициализации")
        
        # Перезагружаем конфигурацию (БЕЗ принудительного выключения автобота)
        load_auto_bot_config(force_disable=False)
        load_system_config()
        logger.info(" 🔄 Перезагружена конфигурация")
        
        # Перезагружаем состояние ботов
        load_bots_state()
        logger.info(" 🔄 Перезагружено состояние ботов")
        
        # НЕ сбрасываем RSI данные! Используем существующий кэш
        # RSI данные обновятся автоматически по расписанию
        logger.info(" ⏭️  RSI данные сохранены (используется кэш)")
        
        # Восстанавливаем флаг инициализации
        system_initialized = True
        logger.info(" ✅ Флаг инициализации восстановлен")
        
        logger.info(" ✅ Сервис ботов перезапущен (RSI кэш сохранен)")
        
        return jsonify({
            'success': True, 
            'message': 'Сервис ботов перезапущен успешно'
        })
        
    except Exception as e:
        logger.error(f" Ошибка перезапуска сервиса: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/process-trading-signals', methods=['POST'])
def process_trading_signals_endpoint():
    """Принудительно обработать торговые сигналы для всех ботов"""
    try:
        logger.info(" 🔄 Принудительная обработка торговых сигналов...")
        
        # Вызываем process_trading_signals_for_all_bots в основном процессе
        process_trading_signals_for_all_bots(exchange_obj=get_exchange())
        
        # Получаем количество активных ботов для отчета
        with bots_data_lock:
            active_bots = {symbol: bot for symbol, bot in bots_data['bots'].items() 
                          if bot['status'] not in [BOT_STATUS['IDLE'], BOT_STATUS['PAUSED']]}
        
        logger.info(f" ✅ Обработка торговых сигналов завершена для {len(active_bots)} ботов")
        
        return jsonify({
            'success': True,
            'message': f'Обработка торговых сигналов завершена для {len(active_bots)} ботов',
            'active_bots_count': len(active_bots)
        })
        
    except Exception as e:
        logger.error(f" ❌ Ошибка обработки торговых сигналов: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/reset-update-flag', methods=['POST'])
def reset_update_flag():
    """Принудительно сбросить флаг update_in_progress"""
    try:
        with rsi_data_lock:
            was_in_progress = coins_rsi_data['update_in_progress']
            coins_rsi_data['update_in_progress'] = False
            
        logger.info(f" 🔄 Флаг update_in_progress сброшен (был: {was_in_progress})")
        return jsonify({
            'success': True,
            'message': 'Флаг update_in_progress сброшен',
            'was_in_progress': was_in_progress
        })
        
    except Exception as e:
        logger.error(f" ❌ Ошибка сброса флага update_in_progress: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bots_app.route('/api/bots/test-stop', methods=['POST'])
def test_stop_bot():
    """Тестовый endpoint для остановки бота"""
    try:
        logger.info(f" 🧪 Тестовый запрос остановки бота")
        logger.info(f" 📥 Raw data: {request.get_data()}")
        logger.info(f" 📥 Headers: {dict(request.headers)}")
        
        # Пробуем получить данные разными способами
        json_data = None
        form_data = None
        args_data = None
        
        try:
            json_data = request.get_json()
            logger.info(f" 📊 JSON data: {json_data}")
        except Exception as e:
            logger.error(f" ❌ JSON error: {e}")
        
        try:
            form_data = request.form.to_dict()
            logger.info(f" 📊 Form data: {form_data}")
        except Exception as e:
            logger.error(f" ❌ Form error: {e}")
        
        try:
            args_data = request.args.to_dict()
            logger.info(f" 📊 Args data: {args_data}")
        except Exception as e:
            logger.error(f" ❌ Args error: {e}")
        
        return jsonify({
            'success': True,
            'message': 'Тестовый запрос получен',
            'json_data': json_data,
            'form_data': form_data,
            'args_data': args_data
        })
        
    except Exception as e:
        logger.error(f" ❌ Ошибка тестового запроса: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bots_app.route('/api/bots/activate-trading-rules', methods=['POST'])
def activate_trading_rules_manual():
    """Активация правил торговли для зрелых монет"""
    try:
        logger.info(" 🎯 Запуск активации правил торговли")
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
        logger.error(f" ❌ Ошибка активации правил торговли: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bots_app.route('/api/bots/individual-settings/<symbol>', methods=['GET', 'POST', 'DELETE'])
def individual_coin_settings(symbol):
    """CRUD операции для индивидуальных настроек монет"""
    try:
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol is required'}), 400

        normalized_symbol = symbol.upper()

        # Загружаем настройки из файла если память пуста
        if not bots_data.get('individual_coin_settings'):
            load_individual_coin_settings()

        if request.method == 'GET':
            settings = get_individual_coin_settings(normalized_symbol)
            if not settings:
                return jsonify({'success': False, 'error': 'Individual settings not found'}), 404
            return jsonify({
                'success': True,
                'symbol': normalized_symbol,
                'settings': settings
            })

        if request.method == 'POST':
            payload = request.get_json(silent=True)
            if not payload or not isinstance(payload, dict):
                return jsonify({'success': False, 'error': 'Invalid settings payload'}), 400

            # Удаляем None значения чтобы не затирать настройки пустыми значениями
            filtered_payload = {k: v for k, v in payload.items() if v is not None}
            filtered_payload['updated_at'] = datetime.now().isoformat()

            stored = set_individual_coin_settings(normalized_symbol, filtered_payload, persist=True)
            logger.info(f" 💾 Настройки для {normalized_symbol} сохранены")

            return jsonify({
                'success': True,
                'symbol': normalized_symbol,
                'settings': stored
            })

        if request.method == 'DELETE':
            removed = remove_individual_coin_settings(normalized_symbol, persist=True)
            if not removed:
                return jsonify({'success': False, 'error': 'Individual settings not found'}), 404
            logger.info(f" 🗑️ Настройки для {normalized_symbol} удалены")
            return jsonify({
                'success': True,
                'symbol': normalized_symbol,
                'removed': True
            })

        return jsonify({'success': False, 'error': 'Unsupported method'}), 405

    except (ValueError, KeyError) as validation_error:
        logger.error(f" ❌ Ошибка валидации настроек {symbol}: {validation_error}")
        return jsonify({'success': False, 'error': str(validation_error)}), 400
    except Exception as e:
        logger.error(f" ❌ Ошибка обработки настроек {symbol}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== FullAI: конфиг (включая адаптивный блок) ====================

@bots_app.route('/api/bots/fullai-config', methods=['GET', 'POST'])
def fullai_config_get_post():
    """GET: конфиг FullAI (для UI, в т.ч. fullai_adaptive_*). POST: обновление (тело мержится в конфиг)."""
    try:
        from bots_modules.imports_and_globals import load_full_ai_config_from_db, save_full_ai_config_to_db, load_auto_bot_config, bots_data, bots_data_lock
        if request.method == 'GET':
            # Сначала подтягиваем рабочий конфиг из bot_config.py (иначе auto_bot_config может быть дефолтом и UI покажет 100 вместо 10)
            try:
                if hasattr(load_auto_bot_config, '_last_mtime'):
                    load_auto_bot_config._last_mtime = 0
                load_auto_bot_config()
            except Exception:
                pass
            cfg = load_full_ai_config_from_db() or {}
            # Подмешиваем значения из рабочего конфига (bot_config.py AutoBotConfig), чтобы UI показывал реальные настройки
            with bots_data_lock:
                auto = (bots_data.get('auto_bot_config') or {}).copy()
            for key in (
                'full_ai_control', 'fullai_adaptive_enabled', 'fullai_adaptive_dead_candles',
                'fullai_scoring_enabled',
            ):
                if key in auto and auto[key] is not None:
                    cfg[key] = auto[key]
            # Маппинг имён из файла (FULLAI_ADAPTIVE_VIRTUAL_SUCCESS) в ключи API (fullai_adaptive_virtual_success_count)
            if auto.get('fullai_adaptive_virtual_success_count') is not None:
                cfg['fullai_adaptive_virtual_success_count'] = auto['fullai_adaptive_virtual_success_count']
            elif auto.get('fullai_adaptive_virtual_success') is not None:
                cfg['fullai_adaptive_virtual_success_count'] = auto['fullai_adaptive_virtual_success']
            if auto.get('fullai_adaptive_real_loss_to_retry') is not None:
                cfg['fullai_adaptive_real_loss_to_retry'] = auto['fullai_adaptive_real_loss_to_retry']
            elif auto.get('fullai_adaptive_real_loss') is not None:
                cfg['fullai_adaptive_real_loss_to_retry'] = auto['fullai_adaptive_real_loss']
            if auto.get('fullai_adaptive_virtual_round_size') is not None:
                cfg['fullai_adaptive_virtual_round_size'] = auto['fullai_adaptive_virtual_round_size']
            elif auto.get('fullai_adaptive_round_size') is not None:
                cfg['fullai_adaptive_virtual_round_size'] = auto['fullai_adaptive_round_size']
            if auto.get('fullai_adaptive_virtual_max_failures') is not None:
                cfg['fullai_adaptive_virtual_max_failures'] = auto['fullai_adaptive_virtual_max_failures']
            elif auto.get('fullai_adaptive_max_failures') is not None:
                cfg['fullai_adaptive_virtual_max_failures'] = auto['fullai_adaptive_max_failures']
            # Источник истины для UI: класс AutoBotConfig из bot_config.py (гарантирует совпадение с файлом)
            try:
                from bot_engine.config_loader import AutoBotConfig
                if getattr(AutoBotConfig, 'FULL_AI_CONTROL', None) is not None:
                    cfg['full_ai_control'] = bool(AutoBotConfig.FULL_AI_CONTROL)
                if getattr(AutoBotConfig, 'FULLAI_ADAPTIVE_ENABLED', None) is not None:
                    cfg['fullai_adaptive_enabled'] = bool(AutoBotConfig.FULLAI_ADAPTIVE_ENABLED)
                if getattr(AutoBotConfig, 'FULLAI_ADAPTIVE_DEAD_CANDLES', None) is not None:
                    cfg['fullai_adaptive_dead_candles'] = int(AutoBotConfig.FULLAI_ADAPTIVE_DEAD_CANDLES)
                if getattr(AutoBotConfig, 'FULLAI_ADAPTIVE_VIRTUAL_SUCCESS', None) is not None:
                    cfg['fullai_adaptive_virtual_success_count'] = int(AutoBotConfig.FULLAI_ADAPTIVE_VIRTUAL_SUCCESS)
                if getattr(AutoBotConfig, 'FULLAI_ADAPTIVE_REAL_LOSS', None) is not None:
                    cfg['fullai_adaptive_real_loss_to_retry'] = int(AutoBotConfig.FULLAI_ADAPTIVE_REAL_LOSS)
                if getattr(AutoBotConfig, 'FULLAI_ADAPTIVE_ROUND_SIZE', None) is not None:
                    cfg['fullai_adaptive_virtual_round_size'] = int(AutoBotConfig.FULLAI_ADAPTIVE_ROUND_SIZE)
                if getattr(AutoBotConfig, 'FULLAI_ADAPTIVE_MAX_FAILURES', None) is not None:
                    cfg['fullai_adaptive_virtual_max_failures'] = int(AutoBotConfig.FULLAI_ADAPTIVE_MAX_FAILURES)
            except Exception as _fullai_err:
                logger.debug("FullAI из AutoBotConfig в fullai-config: %s", _fullai_err)
            return jsonify({'success': True, 'config': cfg})
        if request.method == 'POST':
            payload = request.get_json(silent=True)
            if not payload or not isinstance(payload, dict):
                return jsonify({'success': False, 'error': 'Invalid payload'}), 400
            cfg = load_full_ai_config_from_db() or {}
            for key in (
                'fullai_scoring_enabled',
                'fullai_adaptive_enabled', 'fullai_adaptive_dead_candles',
                'fullai_adaptive_virtual_success_count', 'fullai_adaptive_real_loss_to_retry',
                'fullai_adaptive_virtual_round_size', 'fullai_adaptive_virtual_max_failures',
            ):
                if key in payload:
                    cfg[key] = payload[key]
            if save_full_ai_config_to_db(cfg):
                # Синхронизируем в auto_bot_config и bot_config.py (только если рабочий конфиг уже загружен)
                try:
                    from bots_modules.imports_and_globals import bots_data, bots_data_lock
                    from bots_modules.sync_and_cache import save_auto_bot_config
                    with bots_data_lock:
                        ac = bots_data.get('auto_bot_config')
                    if ac is not None and isinstance(ac, dict):
                        with bots_data_lock:
                            for key in (
                                'full_ai_control', 'fullai_adaptive_enabled', 'fullai_adaptive_dead_candles',
                                'fullai_adaptive_virtual_success_count', 'fullai_adaptive_real_loss_to_retry',
                                'fullai_adaptive_virtual_round_size', 'fullai_adaptive_virtual_max_failures',
                                'fullai_scoring_enabled',
                            ):
                                if key in cfg and cfg[key] is not None:
                                    ac[key] = cfg[key]
                        save_auto_bot_config()
                except Exception as sync_err:
                    logger.debug("FullAI: синхронизация в bot_config.py: %s", sync_err)
                return jsonify({'success': True, 'config': cfg})
            return jsonify({'success': False, 'error': 'Save failed'}), 500
    except Exception as e:
        logger.exception("FullAI config: %s", e)
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== FullAI: параметры по монетам (только при full_ai_control) ====================

@bots_app.route('/api/bots/fullai-coin-params', methods=['GET', 'POST'])
def fullai_coin_params_list():
    """GET: список всех параметров FullAI по монетам. POST: массовое сохранение { "SYMBOL": {...}, ... }."""
    try:
        from bot_engine.bots_database import get_bots_database
        db = get_bots_database()
        if request.method == 'GET':
            all_params = db.load_all_full_ai_coin_params()
            return jsonify({'success': True, 'params': all_params})
        if request.method == 'POST':
            payload = request.get_json(silent=True)
            if not payload or not isinstance(payload, dict):
                return jsonify({'success': False, 'error': 'Invalid payload'}), 400
            saved = {}
            for sym, params in payload.items():
                if not sym or not isinstance(params, dict):
                    continue
                norm = str(sym).upper()
                if db.save_full_ai_coin_params(norm, params):
                    saved[norm] = params
            return jsonify({'success': True, 'saved': saved})
    except Exception as e:
        logger.exception("FullAI coin params: %s", e)
        return jsonify({'success': False, 'error': str(e)}), 500


@bots_app.route('/api/bots/fullai-trades-analysis', methods=['POST'])
def fullai_trades_analysis_run():
    """Запуск анализа сделок FullAI (блок 7.4). Можно вызывать по расписанию (cron)."""
    try:
        from bots_modules.fullai_trades_learner import run_fullai_trades_analysis
        payload = request.get_json(silent=True) or {}
        days_back = int(payload.get('days_back', 7))
        min_trades = int(payload.get('min_trades_per_symbol', 2))
        result = run_fullai_trades_analysis(days_back=days_back, min_trades_per_symbol=min_trades, adjust_params=True)
        return jsonify(result)
    except Exception as e:
        logger.exception("FullAI trades analysis: %s", e)
        return jsonify({'success': False, 'error': str(e)}), 500


@bots_app.route('/api/bots/fullai-coin-params/<symbol>', methods=['GET', 'POST', 'DELETE'])
def fullai_coin_params_one(symbol):
    """CRUD параметров FullAI для одной монеты."""
    try:
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol is required'}), 400
        from bot_engine.bots_database import get_bots_database
        db = get_bots_database()
        norm = str(symbol).upper()
        if request.method == 'GET':
            params = db.load_full_ai_coin_params(norm)
            if params is None:
                return jsonify({'success': False, 'error': 'Not found'}), 404
            return jsonify({'success': True, 'symbol': norm, 'params': params})
        if request.method == 'POST':
            payload = request.get_json(silent=True)
            if not payload or not isinstance(payload, dict):
                return jsonify({'success': False, 'error': 'Invalid payload'}), 400
            if db.save_full_ai_coin_params(norm, payload):
                return jsonify({'success': True, 'symbol': norm, 'params': payload})
            return jsonify({'success': False, 'error': 'Save failed'}), 500
        if request.method == 'DELETE':
            params = db.load_full_ai_coin_params(norm)
            if not params:
                return jsonify({'success': False, 'error': 'Not found'}), 404
            if db.save_full_ai_coin_params(norm, {}):
                return jsonify({'success': True, 'symbol': norm, 'removed': True})
            return jsonify({'success': False, 'error': 'Delete failed'}), 500
    except Exception as e:
        logger.exception("FullAI coin params %s: %s", symbol, e)
        return jsonify({'success': False, 'error': str(e)}), 500


@bots_app.route('/api/bots/individual-settings/<symbol>/copy-to-all', methods=['POST'])
def copy_individual_settings(symbol):
    """Копирует индивидуальные настройки монеты ко всем другим монетам"""
    try:
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol is required'}), 400

        payload = request.get_json(silent=True) or {}
        targets = payload.get('targets') if isinstance(payload, dict) else None

        if targets is not None and not isinstance(targets, list):
            return jsonify({'success': False, 'error': 'targets must be a list'}), 400

        copied_count = copy_individual_coin_settings_to_all(
            symbol,
            targets,
            persist=True
        )

        resp = {
            'success': True,
            'symbol': symbol.upper(),
            'copied_count': copied_count
        }
        if copied_count == 0:
            from bots_modules.imports_and_globals import get_individual_coin_settings
            if not get_individual_coin_settings(symbol):
                resp['message'] = 'У выбранной монеты нет индивидуальных настроек'
        return jsonify(resp)

    except KeyError as missing_error:
        logger.error(f" ❌ Настройки {symbol} не найдены для копирования: {missing_error}")
        return jsonify({'success': False, 'error': 'Individual settings not found'}), 404
    except Exception as e:
        logger.error(f" ❌ Ошибка копирования настроек {symbol}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def _get_candles_for_learn_exit_scam(symbol, timeframe=None):
    """Берёт свечи только из кэша или БД для этой монеты (без запросов к бирже). Расчёт по текущему ТФ.
    timeframe должен быть передан или доступен из конфига — дефолта нет."""
    try:
        from bot_engine.config_loader import get_current_timeframe
        tf = timeframe or get_current_timeframe()
    except Exception:
        raise RuntimeError("КРИТИЧЕСКАЯ ОШИБКА: таймфрейм не выбран или недоступен. Выберите таймфрейм в настройках.")
    if not tf or not str(tf).strip():
        raise RuntimeError("КРИТИЧЕСКАЯ ОШИБКА: таймфрейм не выбран или недоступен. Выберите таймфрейм в настройках.")
    candles = None
    with rsi_data_lock:
        candles_cache = coins_rsi_data.get('candles_cache', {})
        if symbol in candles_cache:
            symbol_cache = candles_cache[symbol]
            if isinstance(symbol_cache, dict) and tf in symbol_cache:
                candles = (symbol_cache[tf].get('candles') or [])[:]
            elif isinstance(symbol_cache, dict) and 'candles' in symbol_cache:
                if symbol_cache.get('timeframe') == tf:
                    candles = (symbol_cache.get('candles') or [])[:]
    if not candles:
        try:
            from bot_engine.storage import get_candles_for_symbol
            db_data = get_candles_for_symbol(symbol)
            if db_data and (db_data.get('timeframe') == tf or not db_data.get('timeframe')):
                candles = (db_data.get('candles') or [])[:]
        except Exception:
            pass
    return candles


@bots_app.route('/api/bots/individual-settings/<symbol>/learn-exit-scam', methods=['POST'])
def learn_exit_scam_for_coin(symbol):
    """Подбор параметров ExitScam по истории свечей для монеты. Свечи берутся из кэша/БД (не с биржи)."""
    try:
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol is required'}), 400
        normalized_symbol = symbol.upper()

        payload = request.get_json(silent=True) or {}
        aggressiveness = (payload.get('aggressiveness') or 'normal').strip().lower()
        if aggressiveness not in ('normal', 'conservative', 'aggressive'):
            aggressiveness = 'normal'
        timeframe = payload.get('timeframe') or None
        try:
            from bot_engine.config_loader import get_current_timeframe
            effective_tf = timeframe or get_current_timeframe()
        except Exception:
            return jsonify({
                'success': False,
                'error': 'КРИТИЧЕСКАЯ ОШИБКА: таймфрейм не выбран или недоступен. Выберите таймфрейм в настройках.'
            }), 500
        if not effective_tf or not str(effective_tf).strip():
            return jsonify({
                'success': False,
                'error': 'КРИТИЧЕСКАЯ ОШИБКА: таймфрейм не выбран или недоступен. Выберите таймфрейм в настройках.'
            }), 500
        candles = _get_candles_for_learn_exit_scam(normalized_symbol, timeframe=timeframe)
        if not candles or len(candles) < 50:
            return jsonify({
                'success': False,
                'error': f'Недостаточно свечей для таймфрейма {effective_tf} (есть {len(candles) or 0}, нужно минимум 50). Дождитесь обновления данных по текущему ТФ.',
                'symbol': normalized_symbol,
                'timeframe': effective_tf
            }), 400

        from bot_engine.ai.exit_scam_learner import compute_exit_scam_params
        params, stats = compute_exit_scam_params(candles, aggressiveness=aggressiveness)
        existing = get_individual_coin_settings(normalized_symbol) or {}
        merged = {**existing, **params}
        set_individual_coin_settings(normalized_symbol, merged, persist=True)
        logger.info(f" ExitScam для {normalized_symbol} обучен и сохранён: {params}")
        return jsonify({
            'success': True,
            'symbol': normalized_symbol,
            'params': params,
            'stats': stats,
        })
    except Exception as e:
        logger.error(f" Ошибка learn-exit-scam для {symbol}: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@bots_app.route('/api/bots/individual-settings/learn-exit-scam-all', methods=['POST'])
def learn_exit_scam_for_all_coins():
    """Ручной расчёт ExitScam по истории для всех монет из кэша/БД (по текущему ТФ)."""
    try:
        payload = request.get_json(silent=True) or {}
        aggressiveness = (payload.get('aggressiveness') or 'normal').strip().lower()
        if aggressiveness not in ('normal', 'conservative', 'aggressive'):
            aggressiveness = 'normal'
        timeframe = payload.get('timeframe') or None
        try:
            from bot_engine.config_loader import get_current_timeframe
            effective_tf = timeframe or get_current_timeframe()
        except Exception:
            return jsonify({
                'success': False,
                'error': 'КРИТИЧЕСКАЯ ОШИБКА: таймфрейм не выбран или недоступен. Выберите таймфрейм в настройках.'
            }), 500
        if not effective_tf or not str(effective_tf).strip():
            return jsonify({
                'success': False,
                'error': 'КРИТИЧЕСКАЯ ОШИБКА: таймфрейм не выбран или недоступен. Выберите таймфрейм в настройках.'
            }), 500

        with rsi_data_lock:
            symbols = list(coins_rsi_data.get('coins', {}).keys())
        if not symbols:
            return jsonify({
                'success': False,
                'error': 'Нет монет в кэше. Дождитесь обновления RSI данных.',
                'updated_count': 0,
                'failed_count': 0,
            }), 400

        from bot_engine.ai.exit_scam_learner import compute_exit_scam_params
        updated_count = 0
        failed_count = 0
        symbols_updated = []
        sample_params = []  # примеры посчитанных параметров по монетам (для проверки индивидуальности)
        for symbol in symbols:
            try:
                candles = _get_candles_for_learn_exit_scam(symbol, timeframe=timeframe)
                if not candles or len(candles) < 50:
                    failed_count += 1
                    continue
                # Явная копия, чтобы не передавать общий список между монетами
                candles = list(candles)
                params, _ = compute_exit_scam_params(candles, aggressiveness=aggressiveness)
                logger.info(
                    f" ExitScam {symbol}: single={params.get('exit_scam_single_candle_percent')}%%, "
                    f"multi N={params.get('exit_scam_multi_candle_count')} {params.get('exit_scam_multi_candle_percent')}%%"
                )
                existing = get_individual_coin_settings(symbol) or {}
                merged = {**existing, **params}
                set_individual_coin_settings(symbol, merged, persist=True)
                updated_count += 1
                symbols_updated.append(symbol)
                if len(sample_params) < 10:
                    sample_params.append({
                        'symbol': symbol,
                        'exit_scam_single_candle_percent': params.get('exit_scam_single_candle_percent'),
                        'exit_scam_multi_candle_percent': params.get('exit_scam_multi_candle_percent'),
                        'exit_scam_multi_candle_count': params.get('exit_scam_multi_candle_count'),
                        'exit_scam_candles': params.get('exit_scam_candles'),
                    })
            except Exception as e:
                logger.debug(f"learn-exit-scam-all {symbol}: {e}")
                failed_count += 1

        # Если ни по одной монете нет свечей в БД/кэше — явная ошибка
        if updated_count == 0:
            logger.warning(f" ExitScam для всех: нет свечей в БД/кэше по ТФ {effective_tf} (нужно мин. 50 на монету)")
            return jsonify({
                'success': False,
                'error': (
                    f'Нет свечей в БД/кэше для расчёта. Загрузите данные свечей по таймфрейму {effective_tf} '
                    '(обновление RSI или загрузка кэша свечей). Нужно минимум 50 свечей на монету.'
                ),
                'updated_count': 0,
                'failed_count': failed_count,
                'timeframe': effective_tf,
            }), 400

        logger.info(f" ExitScam для всех: обновлено {updated_count}, без свечей/ошибок {failed_count}")
        return jsonify({
            'success': True,
            'updated_count': updated_count,
            'failed_count': failed_count,
            'timeframe': effective_tf,
            'symbols_updated': symbols_updated[:50],
            'sample_params': sample_params,
        })
    except Exception as e:
        logger.error(f" Ошибка learn-exit-scam-all: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


EXIT_SCAM_INDIVIDUAL_KEYS = (
    'exit_scam_enabled', 'exit_scam_candles', 'exit_scam_single_candle_percent',
    'exit_scam_multi_candle_count', 'exit_scam_multi_candle_percent',
)


@bots_app.route('/api/bots/individual-settings/reset-exit-scam-all', methods=['POST'])
def reset_exit_scam_to_config_for_all():
    """Сброс индивидуальных настроек ExitScam для всех монет — будут использоваться значения из конфига."""
    try:
        with rsi_data_lock:
            symbols = list(coins_rsi_data.get('coins', {}).keys())
        if not symbols:
            return jsonify({
                'success': True,
                'reset_count': 0,
                'message': 'Нет монет в кэше',
            })
        reset_count = 0
        for symbol in symbols:
            existing = get_individual_coin_settings(symbol) or {}
            if not any(k in existing for k in EXIT_SCAM_INDIVIDUAL_KEYS):
                continue
            rest = {k: v for k, v in existing.items() if k not in EXIT_SCAM_INDIVIDUAL_KEYS}
            if rest:
                set_individual_coin_settings(symbol, rest, persist=True)
            else:
                remove_individual_coin_settings(symbol, persist=True)
            reset_count += 1
        logger.info(f" ExitScam сброшен к конфигу для {reset_count} монет")
        return jsonify({
            'success': True,
            'reset_count': reset_count,
            'message': f'ExitScam сброшен к общим настройкам для {reset_count} монет',
        })
    except Exception as e:
        logger.error(f" Ошибка reset-exit-scam-all: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@bots_app.route('/api/bots/individual-settings/reset-all', methods=['DELETE'])
def reset_all_individual_settings():
    """API для сброса всех индивидуальных настроек к глобальным настройкам"""
    try:
        removed_count = remove_all_individual_coin_settings(persist=True)
        logger.info(f" 🗑️ Сброшены индивидуальные настройки для всех монет ({removed_count} монет)")
        return jsonify({
            'success': True,
            'removed_count': removed_count,
            'message': f'Сброшены индивидуальные настройки для {removed_count} монет'
        })
    except Exception as e:
        logger.error(f" ❌ Ошибка сброса всех индивидуальных настроек: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def _patch_ai_config_after_auto_bot_save(data):
    """После сохранения auto-bot подмешивает AI-ключи в RiskConfig/AIConfig в configs/bot_config.py."""
    ai_keys = {
        'ai_optimal_entry_enabled': ('RiskConfig', 'AI_OPTIMAL_ENTRY_ENABLED'),
        'self_learning_enabled': ('AIConfig', 'AI_SELF_LEARNING_ENABLED'),
        'log_predictions': ('AIConfig', 'AI_LOG_PREDICTIONS'),
        'log_anomalies': ('AIConfig', 'AI_LOG_ANOMALIES'),
        'log_patterns': ('AIConfig', 'AI_LOG_PATTERNS'),
    }
    updates = {}
    for key, (cls_name, attr) in ai_keys.items():
        if key not in data:
            continue
        val = data[key]
        if isinstance(val, str) and val.lower() in ('false', '0', 'no', 'off'):
            val = False
        elif isinstance(val, str) and val.lower() in ('true', '1', 'yes', 'on'):
            val = True
        updates[(cls_name, attr)] = bool(val) if isinstance(val, (bool, int, float)) else val
    if not updates:
        return
    config_path = os.path.join('configs', 'bot_config.py')
    if not os.path.exists(config_path):
        return
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        in_risk, in_ai = False, False
        new_lines = []
        for line in lines:
            if 'class RiskConfig:' in line:
                in_risk, in_ai = True, False
            elif 'class AIConfig:' in line:
                in_risk, in_ai = False, True
            elif line.strip() and not line.strip().startswith('#'):
                if line.strip().startswith('class '):
                    in_risk, in_ai = False, False
            if in_risk and ('RiskConfig', 'AI_OPTIMAL_ENTRY_ENABLED') in updates:
                if 'AI_OPTIMAL_ENTRY_ENABLED =' in line:
                    line = f"    AI_OPTIMAL_ENTRY_ENABLED = {updates[('RiskConfig', 'AI_OPTIMAL_ENTRY_ENABLED')]}\n"
                    del updates[('RiskConfig', 'AI_OPTIMAL_ENTRY_ENABLED')]
            if in_ai:
                for (c, attr), val in list(updates.items()):
                    if c != 'AIConfig':
                        continue
                    if f'{attr} =' in line:
                        line = f"    {attr} = {val}\n"
                        updates.pop((c, attr), None)
                        break
            new_lines.append(line)
        with open(config_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        from bot_engine.config_loader import reload_config
        reload_config()
        logger.debug("[API] AI-настройки синхронизированы в RiskConfig/AIConfig")
    except Exception as e:
        logger.warning("[API] Синхронизация AI в configs/bot_config: %s", e)


@bots_app.route('/api/bots/auto-bot', methods=['GET', 'POST'])
def auto_bot_config():
    """Получить или обновить конфигурацию Auto Bot"""
    try:
        # ✅ Логируем только POST (изменения), GET не логируем (слишком часто)
        if request.method == 'POST':
            logger.info(f" 📝 Изменение конфигурации Auto Bot")
        
        if request.method == 'GET':
            # ✅ КРИТИЧНО: При каждом GET принудительно перечитываем конфиг из файла (источник истины),
            # иначе UI получает кэш и после сохранения другой секции «AI подтверждение» возвращается к старому значению
            from bots_modules.imports_and_globals import load_auto_bot_config
            if hasattr(load_auto_bot_config, '_last_mtime'):
                load_auto_bot_config._last_mtime = 0
            load_auto_bot_config()
            
            with bots_data_lock:
                config = bots_data['auto_bot_config'].copy()
                
                # ✅ Логируем ключевые значения на уровне INFO для отладки (после перезагрузки страницы)
                # Добавляем timestamp для отслеживания, что данные действительно свежие
                # Логирование конфигурации убрано для уменьшения спама (переведено в DEBUG если нужно)
                avoid_down_trend_val = config.get('avoid_down_trend')
                avoid_up_trend_val = config.get('avoid_up_trend')
                
                # ✅ Flask jsonify автоматически преобразует Python bool в JSON boolean
                # Проверяем, что ключевые булевы значения действительно булевы
                # (на случай если они пришли как строки из какого-то другого источника)
                # ❌ КРИТИЧЕСКИ ВАЖНО: НЕ используем bool("False") - это вернет True!
                # Вместо этого проверяем тип и преобразуем правильно
                if 'avoid_down_trend' in config and not isinstance(config['avoid_down_trend'], bool):
                    val = config['avoid_down_trend']
                    logger.warning(f" ⚠️ avoid_down_trend не булево: {type(val).__name__} = {val}, преобразуем правильно")
                    if isinstance(val, str):
                        # Строка "False", "false", "0" -> False, иначе True
                        config['avoid_down_trend'] = val.lower() in ('true', '1', 'yes', 'on')
                    elif isinstance(val, (int, float)):
                        # Число 0 -> False, иначе True
                        config['avoid_down_trend'] = bool(val)
                    else:
                        # Другие типы -> False по умолчанию
                        config['avoid_down_trend'] = False
                    logger.warning(f" ✅ avoid_down_trend преобразовано в: {config['avoid_down_trend']} (тип: {type(config['avoid_down_trend']).__name__})")
                
                if 'avoid_up_trend' in config and not isinstance(config['avoid_up_trend'], bool):
                    val = config['avoid_up_trend']
                    logger.warning(f" ⚠️ avoid_up_trend не булево: {type(val).__name__} = {val}, преобразуем правильно")
                    if isinstance(val, str):
                        # Строка "False", "false", "0" -> False, иначе True
                        config['avoid_up_trend'] = val.lower() in ('true', '1', 'yes', 'on')
                    elif isinstance(val, (int, float)):
                        # Число 0 -> False, иначе True
                        config['avoid_up_trend'] = bool(val)
                    else:
                        # Другие типы -> False по умолчанию
                        config['avoid_up_trend'] = False
                    logger.warning(f" ✅ avoid_up_trend преобразовано в: {config['avoid_up_trend']} (тип: {type(config['avoid_up_trend']).__name__})")
                
                # ✅ Синхронизация AI-настроек: они сохраняются в RiskConfig/AIConfig (POST /api/ai/config),
                # а DEFAULT_AUTO_BOT_CONFIG в файле не обновляется — подмешиваем актуальные значения из bot_config
                try:
                    from bot_engine.config_loader import RiskConfig, AIConfig
                    config['ai_optimal_entry_enabled'] = getattr(RiskConfig, 'AI_OPTIMAL_ENTRY_ENABLED', config.get('ai_optimal_entry_enabled', False))
                    config['self_learning_enabled'] = getattr(AIConfig, 'AI_SELF_LEARNING_ENABLED', config.get('self_learning_enabled', False))
                    config['log_predictions'] = getattr(AIConfig, 'AI_LOG_PREDICTIONS', config.get('log_predictions', False))
                    config['log_anomalies'] = getattr(AIConfig, 'AI_LOG_ANOMALIES', config.get('log_anomalies', False))
                    config['log_patterns'] = getattr(AIConfig, 'AI_LOG_PATTERNS', config.get('log_patterns', False))
                except Exception as _ai_merge_err:
                    logger.debug(f" AI-merge в auto-bot: {_ai_merge_err}")
                # ✅ FullAI: источник истины для UI — класс AutoBotConfig из bot_config.py (после reload в load_auto_bot_config)
                try:
                    from bot_engine.config_loader import AutoBotConfig
                    if getattr(AutoBotConfig, 'FULL_AI_CONTROL', None) is not None:
                        config['full_ai_control'] = bool(AutoBotConfig.FULL_AI_CONTROL)
                    if getattr(AutoBotConfig, 'FULLAI_ADAPTIVE_ENABLED', None) is not None:
                        config['fullai_adaptive_enabled'] = bool(AutoBotConfig.FULLAI_ADAPTIVE_ENABLED)
                    if getattr(AutoBotConfig, 'FULLAI_ADAPTIVE_DEAD_CANDLES', None) is not None:
                        config['fullai_adaptive_dead_candles'] = int(AutoBotConfig.FULLAI_ADAPTIVE_DEAD_CANDLES)
                    if getattr(AutoBotConfig, 'FULLAI_ADAPTIVE_VIRTUAL_SUCCESS', None) is not None:
                        config['fullai_adaptive_virtual_success_count'] = int(AutoBotConfig.FULLAI_ADAPTIVE_VIRTUAL_SUCCESS)
                    if getattr(AutoBotConfig, 'FULLAI_ADAPTIVE_REAL_LOSS', None) is not None:
                        config['fullai_adaptive_real_loss_to_retry'] = int(AutoBotConfig.FULLAI_ADAPTIVE_REAL_LOSS)
                    if getattr(AutoBotConfig, 'FULLAI_ADAPTIVE_ROUND_SIZE', None) is not None:
                        config['fullai_adaptive_virtual_round_size'] = int(AutoBotConfig.FULLAI_ADAPTIVE_ROUND_SIZE)
                    if getattr(AutoBotConfig, 'FULLAI_ADAPTIVE_MAX_FAILURES', None) is not None:
                        config['fullai_adaptive_virtual_max_failures'] = int(AutoBotConfig.FULLAI_ADAPTIVE_MAX_FAILURES)
                except Exception as _fullai_merge_err:
                    logger.debug("FullAI merge в auto-bot: %s", _fullai_merge_err)
                
                # ✅ Фильтры уже загружены в load_auto_bot_config() выше и находятся в bots_data['auto_bot_config']
                # Не нужно повторно загружать их из БД - используем уже загруженные значения
                # Если фильтров нет в config (на всякий случай), устанавливаем значения по умолчанию
                if 'whitelist' not in config:
                    config['whitelist'] = []
                if 'blacklist' not in config:
                    config['blacklist'] = []
                if 'scope' not in config:
                    config['scope'] = 'all'
                
                # ExitScam: лимиты как в конфиге (0.5 = 0.5%), без пересчёта по ТФ
                try:
                    from bots_modules.filters import get_exit_scam_effective_limits
                    from bot_engine.config_loader import get_config_value
                    single_pct = get_config_value(config, 'exit_scam_single_candle_percent')
                    multi_count = get_config_value(config, 'exit_scam_multi_candle_count')
                    multi_pct = get_config_value(config, 'exit_scam_multi_candle_percent')
                    tf_name, eff_single, eff_multi = get_exit_scam_effective_limits(single_pct, multi_count, multi_pct)
                    config['exit_scam_timeframe'] = tf_name
                    config['exit_scam_effective_single_pct'] = round(float(eff_single), 2)
                    config['exit_scam_effective_multi_pct'] = round(float(eff_multi), 2)
                except Exception:
                    from bot_engine.config_loader import get_current_timeframe
                    _tf = get_current_timeframe()
                    if not _tf or not str(_tf).strip():
                        return jsonify({
                            'success': False,
                            'error': 'КРИТИЧЕСКАЯ ОШИБКА: таймфрейм не выбран или недоступен. Выберите таймфрейм в настройках.'
                        }), 500
                    config['exit_scam_timeframe'] = _tf
                    config['exit_scam_effective_single_pct'] = get_config_value(config, 'exit_scam_single_candle_percent')
                    config['exit_scam_effective_multi_pct'] = get_config_value(config, 'exit_scam_multi_candle_percent')
                
                # Для UI: доступность переключателя «Полный Режим ИИ» только при ИИ и валидной лицензии
                try:
                    from bot_engine.ai import get_ai_manager
                    config['ai_license_valid'] = get_ai_manager().is_available()
                except Exception:
                    config['ai_license_valid'] = False
                
                return jsonify({
                    'success': True,
                    'config': config
                })
        
        elif request.method == 'POST':
            # Добавляем логирование для отладки
            try:
                data = request.get_json()
                pass
            except Exception as json_error:
                logger.error(f" ❌ Ошибка парсинга JSON: {json_error}")
                return jsonify({'success': False, 'error': f'Invalid JSON: {str(json_error)}'}), 400
            
            # ✅ Проверяем на None и пустой словарь
            if data is None or (isinstance(data, dict) and len(data) == 0):
                logger.warning(" ⚠️ Пустые данные или пустой объект, пропускаем обработку")
                return jsonify({
                    'success': True,
                    'message': 'Нет изменений в настройках',
                    'config': bots_data['auto_bot_config'].copy(),
                    'saved_to_file': True,
                    'changes_count': 0,
                    'changed_params': []
                }), 200
            
            # FullAI: явное логирование при получении переключателя из UI (для отладки «не включается»)
            if 'full_ai_control' in data:
                logger.info(f"[API] [FullAI] Получен full_ai_control = {data['full_ai_control']!r} (из UI)")
            
            # Проверяем изменение критериев зрелости
            maturity_params_changed = False
            maturity_keys = ['min_candles_for_maturity', 'min_rsi_low', 'max_rsi_high']
            changes_count = 0
            
            # ✅ Сохраняем старую конфигурацию для сравнения
            with bots_data_lock:
                old_config = bots_data['auto_bot_config'].copy()
            
            # ✅ Сначала проверяем какие изменения будут (только для критериев зрелости)
            for key in maturity_keys:
                if key in data and data[key] != old_config.get(key):
                    maturity_params_changed = True
                    logger.warning(f" ⚠️ Изменен критерий зрелости: {key} ({old_config.get(key)} → {data[key]})")
            
            # Примечание: changes_count будет подсчитан при формировании changed_data ниже
            
            # ✅ КРИТИЧЕСКИ ВАЖНО: Фильтруем только ИЗМЕНЕННЫЕ значения
            # Создаем словарь только с реально измененными параметрами
            changed_data = {}
            changed_params_list = []  # Список измененных параметров для toast
            # changes_count уже инициализирован выше, не сбрасываем
            
            with bots_data_lock:
                # ✅ КРИТИЧЕСКИ ВАЖНО: Логируем enabled до обновления
                enabled_before = bots_data['auto_bot_config'].get('enabled')
                logger.info(f"[API] 🔍 enabled ДО обновления: {enabled_before}, enabled в data: {data.get('enabled', 'НЕ ПЕРЕДАН')}")
                
                for key, value in data.items():
                    old_value = bots_data['auto_bot_config'].get(key)
                    
                    # ✅ НОРМАЛИЗУЕМ значения для корректного сравнения (int vs float, str vs int и т.д.)
                    def normalize_value(v):
                        if v is None:
                            return None
                        if isinstance(v, bool):
                            return v
                        if isinstance(v, (int, float)):
                            # Для чисел - сравниваем как float, но сохраняем исходный тип
                            return float(v)
                        if isinstance(v, str):
                            # Пробуем преобразовать строку в число, если возможно
                            try:
                                if '.' in v:
                                    return float(v)
                                else:
                                    return int(v)
                            except ValueError:
                                return v
                        return v
                    
                    normalized_old = normalize_value(old_value)
                    normalized_new = normalize_value(value)
                    
                    # ✅ КРИТИЧЕСКИ ВАЖНО: Добавляем в changed_data только если значение РЕАЛЬНО изменилось
                    if normalized_old != normalized_new:
                        changed_data[key] = value
                        bots_data['auto_bot_config'][key] = value
                        # ✅ Синхронизация break_even_trigger: при изменении break_even_trigger_percent обновляем оба ключа
                        if key == 'break_even_trigger_percent':
                            bots_data['auto_bot_config']['break_even_trigger'] = value
                            if 'break_even_trigger' not in changed_data:
                                changed_data['break_even_trigger'] = value
                        elif key == 'break_even_trigger':
                            bots_data['auto_bot_config']['break_even_trigger_percent'] = value
                            if 'break_even_trigger_percent' not in changed_data:
                                changed_data['break_even_trigger_percent'] = value
                        
                        # Формируем список изменений для логирования и toast
                        if key not in old_config:
                            # Новый ключ добавляется
                            changes_count += 1
                            display_name = CONFIG_NAMES.get(key, key)
                            changed_params_list.append(f"{display_name} = {value} (новый)")
                            logger.info(f" ✅ Добавлен новый ключ конфигурации: {display_name} = {value}")
                        else:
                            # Изменение существующего ключа
                            changes_count += 1
                            # Используем log_config_change для красивого логирования
                            log_config_change(key, old_value, value)
                            # Добавляем в список для toast
                            display_name = CONFIG_NAMES.get(key, key)
                            changed_params_list.append(f"{display_name}: {old_value} → {value}")
                
                # ✅ КРИТИЧЕСКИ ВАЖНО: Логируем enabled после обновления
                enabled_after = bots_data['auto_bot_config'].get('enabled')
                logger.info(f"[API] 🔍 enabled ПОСЛЕ обновления: {enabled_after}")
                logger.info(f"[API] 🔍 Изменено параметров: {len(changed_data)} из {len(data)}")
                
                # ✅ Логируем только ИЗМЕНЕННЫЕ параметры
                if len(changed_data) > 0:
                    logger.info(f"[API] 📋 Измененные параметры: {', '.join(changed_params_list[:10])}{'...' if len(changed_params_list) > 10 else ''}")
                    if 'full_ai_control' in changed_data:
                        logger.info(f"[API] [FullAI] full_ai_control применён в конфиг → {changed_data['full_ai_control']!r}, будет сохранён в файл")
                else:
                    logger.info(f"[API] ⏭️ Нет изменений: все {len(data)} параметров без изменений")
            
            # FullAI: при включении — авто-включение ИИ (если лицензия валидна); при выключении — восстановление настроек
            old_fullai = old_config.get('full_ai_control', False)
            new_fullai = bots_data['auto_bot_config'].get('full_ai_control', False)
            if new_fullai and not old_fullai:
                try:
                    from bots_modules.imports_and_globals import load_full_ai_config_from_db, save_full_ai_config_to_db
                    from copy import deepcopy
                    fullai_cfg = load_full_ai_config_from_db() or {}
                    # Снимок для восстановления при выключении FullAI (если включаем ИИ автоматически)
                    snapshot = fullai_cfg.get('user_ai_state_before_prii')
                    # Конфиг FullAI считаем пустым, если нет «нормальных» ключей (только user_ai_state_before_prii или вообще пусто)
                    has_real_config = any(k for k in (fullai_cfg or {}) if k != 'user_ai_state_before_prii')
                    if not has_real_config:
                        with bots_data_lock:
                            initial_fullai = deepcopy(bots_data.get('auto_bot_config', {}))
                        for _k in ('system_timeframe', 'timeframe', 'SYSTEM_TIMEFRAME'):
                            initial_fullai.pop(_k, None)
                        if snapshot is not None:
                            initial_fullai['user_ai_state_before_prii'] = snapshot
                        initial_fullai['fullai_adaptive_enabled'] = True  # при включении Full AI обкатка по умолчанию вкл
                        save_full_ai_config_to_db(initial_fullai)
                        logger.info("[FullAI] Конфиг FullAI инициализирован → configs/fullai_config.json и БД")
                    from bot_engine.ai import get_ai_manager
                    if get_ai_manager().is_available():
                        ai_before = bots_data['auto_bot_config'].get('ai_enabled', False)
                        if not ai_before:
                            fullai_cfg = load_full_ai_config_from_db() or {}
                            fullai_cfg['user_ai_state_before_prii'] = {'ai_enabled': False}
                            save_full_ai_config_to_db(fullai_cfg)
                            with bots_data_lock:
                                bots_data['auto_bot_config']['ai_enabled'] = True
                            changed_data['ai_enabled'] = True
                            changes_count += 1
                            changed_params_list.append('ai_enabled: False → True (FullAI)')
                            logger.info("[FullAI] ИИ автоматически включён при включении FullAI (лицензия валидна)")
                except Exception as e:
                    logger.debug(f"[FullAI] Авто-включение ИИ / инициализация конфига: {e}")
            elif not new_fullai and old_fullai:
                try:
                    from bots_modules.imports_and_globals import load_full_ai_config_from_db, save_full_ai_config_to_db
                    fullai_cfg = load_full_ai_config_from_db() or {}
                    snapshot = fullai_cfg.pop('user_ai_state_before_prii', None)
                    if snapshot and isinstance(snapshot, dict):
                        with bots_data_lock:
                            for k, v in snapshot.items():
                                bots_data['auto_bot_config'][k] = v
                                changed_data[k] = v
                                changes_count += 1
                                changed_params_list.append(f"{k} восстановлено (FullAI выкл)")
                        save_full_ai_config_to_db(fullai_cfg)
                        logger.info("[FullAI] Восстановлены настройки до включения FullAI: %s", list(snapshot.keys()))
                except Exception as e:
                    logger.debug(f"[FullAI] Восстановление настроек: {e}")
            
            # ✅ КРИТИЧЕСКИ ВАЖНО: Сохраняем фильтры (whitelist, blacklist, scope) в БД
            # И ВАЖНО: scope также должен быть в bots_data['auto_bot_config'] для сохранения в файл!
            filters_saved = False
            try:
                from bot_engine.bots_database import get_bots_database
                db = get_bots_database()
                
                # Извлекаем фильтры из data
                whitelist = data.get('whitelist') if 'whitelist' in data else None
                blacklist = data.get('blacklist') if 'blacklist' in data else None
                scope = data.get('scope') if 'scope' in data else None
                
                if whitelist is not None or blacklist is not None or scope is not None:
                    filters_saved = db.save_coin_filters(whitelist=whitelist, blacklist=blacklist, scope=scope)
                    if filters_saved:
                        logger.info(f"✅ Фильтры сохранены в БД: whitelist={len(whitelist) if whitelist else 'не изменен'}, blacklist={len(blacklist) if blacklist else 'не изменен'}, scope={scope if scope else 'не изменен'}")
            except Exception as e:
                logger.error(f"❌ Ошибка сохранения фильтров в БД: {e}")
            
            # ✅ КРИТИЧЕСКИ ВАЖНО: Сохраняем в файл ТОЛЬКО если есть изменения!
            # Если changed_data пустой, не сохраняем и возвращаем сообщение "Нет изменений"
            logger.info(f"[API] 🔍 Проверка changed_data: длина={len(changed_data)}, ключи={list(changed_data.keys())}")
            if len(changed_data) > 0:
                # КРИТИЧЕСКИ ВАЖНО: Сохраняем конфигурацию в файл (с перезагрузкой модуля)
                logger.info(f"[API] ✅ Есть изменения, вызываем save_auto_bot_config()")
                save_result = save_auto_bot_config(changed_data=changed_data)
                logger.info(f"✅ Auto Bot: изменено параметров: {changes_count}, конфигурация сохранена и перезагружена")
                # Синхронизация AI-настроек в RiskConfig/AIConfig (они пишутся в DEFAULT_AUTO_BOT_CONFIG, но UI читает из RiskConfig/AIConfig)
                _patch_ai_config_after_auto_bot_save(data)
            else:
                # Нет изменений - не сохраняем и не перезагружаем
                logger.info(f"[API] ⏭️ Нет изменений, НЕ вызываем save_auto_bot_config()")
                save_result = True  # Успех, но без сохранения
                logger.info("ℹ️  Auto Bot: изменений не обнаружено")
            
            # ✅ АВТОМАТИЧЕСКАЯ ОЧИСТКА при изменении критериев зрелости
            if maturity_params_changed:
                logger.warning("=" * 80)
                logger.warning(" 🔄 КРИТЕРИИ ЗРЕЛОСТИ ИЗМЕНЕНЫ!")
                logger.warning(" 🗑️ Очистка файла зрелых монет...")
                logger.warning("=" * 80)
                
                try:
                    # Очищаем хранилище зрелых монет
                    clear_mature_coins_storage()
                    logger.info(" ✅ Файл зрелых монет очищен")
                    logger.info(" 🔄 Монеты будут перепроверены при следующей загрузке RSI")
                except Exception as e:
                    logger.error(f" ❌ Ошибка очистки файла зрелых монет: {e}")
            
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
                    logger.info(" ✅ Немедленная проверка Auto Bot завершена")
                except Exception as e:
                    logger.error(f" ❌ Ошибка немедленной проверки Auto Bot: {e}")
            
            # КРИТИЧЕСКИ ВАЖНО: При отключении Auto Bot НЕ удаляем ботов!
            # Показываем блок только если enabled реально изменился с True на False
            if 'enabled' in data and old_config.get('enabled') == True and data['enabled'] == False:
                # ✅ ЯРКИЙ ЛОГ ВЫКЛЮЧЕНИЯ (КРАСНЫЙ)
                logger.info("=" * 80)
                print("\033[91m🔴 AUTO BOT ВЫКЛЮЧЕН! 🔴\033[0m")
                logger.info("=" * 80)
                
                with bots_data_lock:
                    bots_count = len(bots_data['bots'])
                    bots_in_position = sum(
                        1
                        for bot in bots_data['bots'].values()
                        if bot.get('status', '').lower() in ['in_position_long', 'in_position_short']
                    )
                
                if bots_count > 0:
                    logger.info("")
                    logger.info("✅ ЧТО БУДЕТ ДАЛЬШЕ:")
                    logger.info("   🔄 Существующие боты продолжат работать")
                    logger.info("   🛡️ Защитные механизмы активны (стоп-лосс, RSI выход)")
                    logger.info("   ❌ Новые боты НЕ будут создаваться")
                    logger.info("   🗑️ Для удаления используйте кнопку 'Удалить всё'")
                else:
                    logger.info("ℹ️  Нет активных ботов")
                
                logger.info("=" * 80)
                logger.warning("✅ АВТОБОТ ОСТАНОВЛЕН (боты сохранены)")
                logger.info("=" * 80)
        
            # ✅ Формируем сообщение в зависимости от наличия изменений
            if changes_count > 0:
                # Формируем детальное сообщение с перечислением изменений
                if len(changed_params_list) <= 5:
                    # Если изменений мало - показываем все
                    changes_text = ', '.join(changed_params_list)
                    message = f'Сохранено: {changes_text}'
                else:
                    # Если изменений много - показываем первые 3 и количество
                    first_changes = ', '.join(changed_params_list[:3])
                    message = f'Сохранено: {first_changes} и еще {len(changed_params_list) - 3} параметров'
            else:
                message = 'Нет изменений в настройках'
        
        return jsonify({
            'success': True,
            'message': message,
            'config': bots_data['auto_bot_config'].copy(),
            'saved_to_file': save_result,
            'changes_count': changes_count,  # ✅ Добавляем количество изменений для UI
            'changed_params': changed_params_list  # ✅ Добавляем список измененных параметров для UI
        })
        
    except Exception as e:
        logger.error(f" Ошибка конфигурации Auto Bot: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/auto-bot/restore-defaults', methods=['POST'])
def restore_auto_bot_defaults():
    """Восстанавливает дефолтную конфигурацию Auto Bot"""
    try:
        logger.info(" 🔄 Запрос на восстановление дефолтной конфигурации Auto Bot")
        
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
        logger.error(f" Ошибка восстановления дефолтной конфигурации: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/debug-init', methods=['GET'])
def debug_init_status():
    """Отладочный эндпоинт для проверки инициализации"""
    try:
        from bots_modules.continuous_data_loader import get_continuous_loader
        loader = get_continuous_loader()
        
        return jsonify({
            'success': True,
            'init_bot_service_called': 'init_bot_service' in globals(),
            'continuous_loader_running': loader is not None and loader.is_running if loader else False,
            'exchange_exists': exchange is not None,
            'bots_data_keys': list(bots_data.keys()) if 'bots_data' in globals() else 'not_initialized'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def _get_continuous_loader_status():
    """Helper для получения статуса Continuous Data Loader"""
    try:
        from bots_modules.continuous_data_loader import get_continuous_loader
        loader = get_continuous_loader()
        return loader is not None and loader.is_running if loader else False
    except:
        return False

@bots_app.route('/api/bots/process-state', methods=['GET'])
def get_process_state():
    """Получить состояние всех процессов системы"""
    try:
        return jsonify({
            'success': True,
            'process_state': process_state.copy(),
            'system_info': {
                'continuous_loader_running': _get_continuous_loader_status(),
                'exchange_initialized': exchange is not None,
                'total_bots': len(bots_data['bots']),
                'auto_bot_enabled': bots_data['auto_bot_config']['enabled'],
                'mature_coins_storage_size': len(mature_coins_storage)
            }
                })
        
    except Exception as e:
        logger.error(f" Ошибка получения состояния процессов: {str(e)}")
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
        logger.error(f" Ошибка получения зрелых монет: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/mature-coins/reload', methods=['POST'])
def reload_mature_coins():
    """Перезагрузить список зрелых монет из файла"""
    try:
        load_mature_coins_storage()
        logger.info(f" Перезагружено {len(mature_coins_storage)} зрелых монет")
        return jsonify({
            'success': True,
            'message': f'Перезагружено {len(mature_coins_storage)} зрелых монет',
            'data': {
                'mature_coins': list(mature_coins_storage.keys()),
                'count': len(mature_coins_storage)
            }
        })
    except Exception as e:
        logger.error(f" Ошибка перезагрузки зрелых монет: {str(e)}")
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
        logger.error(f" Ошибка удаления монеты {symbol} из хранилища: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/mature-coins/clear', methods=['POST'])
def clear_mature_coins_storage():
    """Очистка всего постоянного хранилища зрелых монет"""
    try:
        global mature_coins_storage
        mature_coins_storage = {}
        save_mature_coins_storage()
        logger.info(" Постоянное хранилище зрелых монет очищено")
        return jsonify({
            'success': True,
            'message': 'Постоянное хранилище зрелых монет очищено'
        })
    except Exception as e:
        logger.error(f" Ошибка очистки хранилища зрелых монет: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ❌ ОТКЛЮЧЕНО: Все Optimal EMA endpoints удалены (EMA фильтр убран из системы)
# @bots_app.route('/api/bots/optimal-ema', methods=['GET'])
# @bots_app.route('/api/bots/optimal-ema/<symbol>', methods=['GET'])
# @bots_app.route('/api/bots/optimal-ema/<symbol>/rescan', methods=['POST'])

# ❌ ОТКЛЮЧЕНО: Optimal EMA Worker больше не используется (EMA фильтр убран)
# @bots_app.route('/api/bots/optimal-ema-worker/status', methods=['GET'])
# def get_optimal_ema_worker_status():
#     """Получает статус воркера оптимальных EMA"""
#     return jsonify({
#         'success': False,
#         'error': 'Optimal EMA Worker отключен - EMA фильтр больше не используется'
#     }), 404

# @bots_app.route('/api/bots/optimal-ema-worker/force-update', methods=['POST'])
# def force_optimal_ema_update():
#     """Принудительно запускает обновление оптимальных EMA"""
#     return jsonify({
#         'success': False,
#         'error': 'Optimal EMA Worker отключен - EMA фильтр больше не используется'
#     }), 404

# @bots_app.route('/api/bots/optimal-ema-worker/set-interval', methods=['POST'])
# def set_optimal_ema_interval():
#     """Устанавливает интервал обновления воркера оптимальных EMA"""
#     return jsonify({
#         'success': False,
#         'error': 'Optimal EMA Worker отключен - EMA фильтр больше не используется'
#     }), 404

# Старый код (закомментирован):
# @bots_app.route('/api/bots/optimal-ema-worker/status', methods=['GET'])
# def get_optimal_ema_worker_status():
#     """Получает статус воркера оптимальных EMA"""
#     try:
#         from bot_engine.optimal_ema_worker import get_optimal_ema_worker
#         
#         worker = get_optimal_ema_worker()
#         if worker:
#             status = worker.get_status()
#             return jsonify({
#                 'success': True,
#                 'data': status
#             })
#         else:
#             return jsonify({
#                 'success': False,
#                 'error': 'Воркер оптимальных EMA не инициализирован'
#             }), 404
#     except Exception as e:
#         logger.error(f" Ошибка получения статуса воркера: {e}")
#         return jsonify({'success': False, 'error': str(e)}), 500
# 
# @bots_app.route('/api/bots/optimal-ema-worker/force-update', methods=['POST'])
# def force_optimal_ema_update():
#     """Принудительно запускает обновление оптимальных EMA"""
#     try:
#         from bot_engine.optimal_ema_worker import get_optimal_ema_worker
#         
#         worker = get_optimal_ema_worker()
#         if worker:
#             success = worker.force_update()
#             if success:
#                 return jsonify({
#                     'success': True,
#                     'message': 'Принудительное обновление оптимальных EMA запущено'
#                 })
#             else:
#                 return jsonify({
#                     'success': False,
#                     'error': 'Обновление уже выполняется'
#                 }), 409
#         else:
#             return jsonify({
#                 'success': False,
#                 'error': 'Воркер оптимальных EMA не инициализирован'
#             }), 404
#     except Exception as e:
#         logger.error(f" Ошибка принудительного обновления: {e}")
#         return jsonify({'success': False, 'error': str(e)}), 500
# 
# @bots_app.route('/api/bots/optimal-ema-worker/set-interval', methods=['POST'])
# def set_optimal_ema_interval():
#     """Устанавливает интервал обновления воркера оптимальных EMA"""
#     try:
#         from bot_engine.optimal_ema_worker import get_optimal_ema_worker
#         
#         data = request.get_json()
#         if not data or 'interval' not in data:
#             return jsonify({
#                 'success': False,
#                 'error': 'Не указан интервал обновления'
#             }), 400
#         
#         interval = int(data['interval'])
#         if interval < 300:  # Минимум 5 минут
#             return jsonify({
#                 'success': False,
#                 'error': 'Интервал не может быть меньше 300 секунд (5 минут)'
#             }), 400
#         
#         worker = get_optimal_ema_worker()
#         if worker:
#             success = worker.set_update_interval(interval)
#             if success:
#                 return jsonify({
#                     'success': True,
#                     'message': f'Интервал обновления изменен на {interval} секунд'
#                 })
#             else:
#                 return jsonify({
#                     'success': False,
#                     'error': 'Не удалось изменить интервал'
#                 })
#         else:
#             return jsonify({
#                 'success': False,
#                 'error': 'Воркер оптимальных EMA не инициализирован'
#             }), 404
#     except Exception as e:
#         logger.error(f" Ошибка изменения интервала: {e}")
#         return jsonify({'success': False, 'error': str(e)}), 500

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
        logger.error(f" Ошибка загрузки дефолтной конфигурации: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/auto-bot/test-signals', methods=['POST'])
def test_auto_bot_signals():
    """Тестовый эндпоинт для принудительной обработки Auto Bot сигналов - УДАЛЕНО!"""
    return jsonify({'success': False, 'message': 'Auto Bot отключен!'})
    try:
        logger.info(" 🧪 Принудительная обработка Auto Bot сигналов...")
        
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
        logger.error(f" Ошибка тестирования Auto Bot: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@bots_app.errorhandler(500)
def internal_error(error):
    logger.error(f" Внутренняя ошибка сервера: {str(error)}")
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


def _soft_restart_bots_service():
    """Перезапускает основные компоненты сервиса без остановки Flask."""
    try:
        logger.info(" ♻️ Мягкий перезапуск сервисных компонентов...")

        try:
            cleanup_bot_service()
        except Exception as cleanup_error:
            logger.warning(f" ⚠️ Ошибка при очистке перед перезапуском: {cleanup_error}")

        init_success = init_bot_service()
        if not init_success:
            logger.error(" ❌ init_bot_service вернул False при мягком перезапуске")
            return False, "init_bot_service() вернул False"

        try:
            start_async_processor()
        except Exception as start_error:
            logger.warning(f" ⚠️ Не удалось запустить асинхронный процессор после перезапуска: {start_error}")

        logger.info(" ✅ Мягкий перезапуск выполнен без остановки Flask")
        return True, None

    except Exception as exc:
        logger.error(f" ❌ Ошибка мягкого перезапуска: {exc}")
        return False, str(exc)

def signal_handler(signum, frame):
    """Обработчик сигналов завершения с принудительным завершением"""
    global graceful_shutdown
    print(f"\n 🛑 Получен сигнал {signum}, начинаем graceful shutdown...")
    logger.warning(f" 🛑 Получен сигнал {signum}, начинаем graceful shutdown...")
    graceful_shutdown = True
    shutdown_flag.set()
    
    # Запускаем принудительное завершение через таймер
    def force_exit():
        time.sleep(2.0)  # Даём 2 секунды на graceful shutdown
        print(" ⏱️ Таймаут graceful shutdown, принудительное завершение...")
        logger.warning(" ⏱️ Таймаут graceful shutdown, принудительное завершение...")
        os._exit(0)
    
    force_exit_thread = threading.Thread(target=force_exit, daemon=True)
    force_exit_thread.start()
    
    # Пытаемся выполнить graceful shutdown
    try:
        cleanup_bot_service()
        print(" ✅ Graceful shutdown завершен")
        logger.warning(" ✅ Graceful shutdown завершен")
        sys.exit(0)
    except Exception as e:
        print(f" ⚠️ Ошибка при graceful shutdown: {e}")
        logger.error(f" ⚠️ Ошибка при graceful shutdown: {e}")
        os._exit(1)

@bots_app.route('/api/system/reload-modules', methods=['POST'])
def reload_modules():
    """Умная горячая перезагрузка модулей с поддержкой Flask"""
    try:
        import importlib
        import sys
        # Определяем модули для перезагрузки в порядке зависимостей
        modules_to_reload = [
            'bot_engine.config_loader',
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
        
        logger.info(" 🔄 Начинаем умную горячую перезагрузку...")
        
        # Этап 1: Перезагружаем безопасные модули
        for module_name in modules_to_reload:
            try:
                if module_name == 'bot_engine.config_loader':
                    from bot_engine.config_loader import reload_config
                    logger.info(" 🔄 Перезагрузка конфига (config_loader + configs.bot_config)...")
                    reload_config()
                    reloaded.append(module_name)
                    logger.info(" ✅ Модуль bot_engine.config_loader перезагружен")
                elif module_name in sys.modules:
                    logger.info(f" 🔄 Перезагрузка модуля {module_name}...")
                    module = sys.modules[module_name]
                    importlib.reload(module)
                    reloaded.append(module_name)
                    logger.info(f" ✅ Модуль {module_name} перезагружен")
                else:
                    logger.warning(f" ⚠️ Модуль {module_name} не был загружен")
            except Exception as e:
                logger.error(f" ❌ Ошибка перезагрузки {module_name}: {e}")
                failed.append({'module': module_name, 'error': str(e)})
        
        # Этап 1.5: Проверяем состояние Flask после перезагрузки
        try:
            logger.info(" 🔍 Проверка состояния Flask после перезагрузки...")
            # Простая проверка что Flask все еще работает
            if hasattr(request, 'method') and hasattr(request, 'get_json'):
                logger.info(" ✅ Flask состояние корректно")
            else:
                logger.warning(" ⚠️ Flask состояние может быть нарушено")
        except Exception as e:
            logger.error(f" ❌ Ошибка проверки Flask: {e}")
        
        # Этап 2: Проверяем нужен ли перезапуск Flask сервера
        try:
            request_data = request.get_json(silent=True) or {}
            force_flask_restart = request_data.get('force_flask_restart', False)
            logger.info(f" 📋 Данные запроса: {request_data}")
        except Exception as e:
            logger.error(f" ❌ Ошибка парсинга JSON запроса: {e}")
            request_data = {}
            force_flask_restart = False
        
        restart_requested = force_flask_restart or any(module in sys.modules for module in flask_restart_modules)
        soft_restart_performed = False
        soft_restart_error = None

        if restart_requested:
            logger.info(" 🔄 Требуется мягкий перезапуск сервисных компонентов (Flask останется активным)...")
        else:
            logger.info(" ✅ Перезапуск Flask компонентов не требуется")
        
        # Этап 3: Перезагружаем конфигурацию
        try:
            from bots_modules.imports_and_globals import load_auto_bot_config
            load_auto_bot_config()
            logger.info(" ✅ Конфигурация Auto Bot перезагружена")
        except Exception as e:
            logger.error(f" ❌ Ошибка перезагрузки конфигурации: {e}")

        if restart_requested:
            soft_restart_performed, soft_restart_error = _soft_restart_bots_service()
            if soft_restart_performed:
                logger.info(" ✅ Мягкий перезапуск сервисных компонентов выполнен")
            else:
                logger.error(f" ❌ Ошибка мягкого перезапуска: {soft_restart_error}")
        
        # Формируем ответ
        response_data = {
            'success': True,
            'reloaded': reloaded,
            'failed': failed,
            'flask_restart_required': False,
            'restart_requested': restart_requested,
            'soft_restart_performed': soft_restart_performed,
            'message': f'Перезагружено {len(reloaded)} модулей'
        }
        
        if soft_restart_performed:
            response_data['message'] += '. Выполнен мягкий перезапуск сервисов без остановки Flask.'
        elif restart_requested:
            response_data['message'] += '. Требовался мягкий перезапуск, но завершился с ошибкой.'
            if soft_restart_error:
                response_data['soft_restart_error'] = soft_restart_error
        
        logger.info(f" ✅ Горячая перезагрузка завершена: {len(reloaded)} модулей")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f" ❌ Общая ошибка горячей перезагрузки: {e}")
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
        logger.error(f" ❌ Ошибка получения делистинговых монет: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bots_app.route('/api/bots/delisted-coins/force-scan', methods=['POST'])
def force_delisting_scan_api():
    """API для принудительного сканирования делистинговых монет"""
    try:
        from bots_modules.sync_and_cache import scan_all_coins_for_delisting
        
        logger.info(" 🔍 Принудительное сканирование делистинговых монет...")
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
        logger.error(f" ❌ Ошибка принудительного сканирования: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def cleanup_bot_service():
    """Очистка ресурсов при завершении сервиса"""
    global system_initialized
    
    # КРИТИЧЕСКИ ВАЖНО: Сбрасываем флаг, чтобы остановить торговлю
    system_initialized = False
    logger.warning(" 🛑 Флаг system_initialized сброшен - торговля остановлена")
    
    try:
        logger.info(" 🧹 Очистка ресурсов сервиса ботов...")
        
        # Останавливаем асинхронный процессор
        stop_async_processor()
        
        # Останавливаем Continuous Data Loader
        try:
            from bots_modules.continuous_data_loader import stop_continuous_loader
            logger.warning(" 🛑 Остановка Continuous Data Loader...")
            stop_continuous_loader()
        except Exception as e:
            logger.warning(f" ⚠️ Ошибка остановки Continuous Data Loader: {e}")
        
        # ❌ ОТКЛЮЧЕНО: Воркер оптимальных EMA больше не используется
        # try:
        #     from bot_engine.optimal_ema_worker import stop_optimal_ema_worker
        #     stop_optimal_ema_worker()
        #     logger.info(" 🛑 Остановка воркера оптимальных EMA...")
        # except Exception as e:
        #     logger.error(f" Ошибка остановки воркера оптимальных EMA: {e}")
        
        # Сохраняем все важные данные
        logger.info(" 💾 Финальное сохранение всех данных...")
        
        # 1. Сохраняем состояние ботов
        logger.info(" 📊 Сохранение состояния ботов...")
        save_bots_state()
        
        # 2. Сохраняем конфигурацию автобота
        logger.info(" ⚙️ Сохранение конфигурации автобота...")
        save_auto_bot_config()
        
        # 3. Сохраняем системную конфигурацию
        logger.info(" 🔧 Сохранение системной конфигурации...")
        system_config_data = {
            'bot_status_update_interval': SystemConfig.BOT_STATUS_UPDATE_INTERVAL,
            'position_sync_interval': SystemConfig.POSITION_SYNC_INTERVAL,
            'inactive_bot_cleanup_interval': SystemConfig.INACTIVE_BOT_CLEANUP_INTERVAL,
            'inactive_bot_timeout': SystemConfig.INACTIVE_BOT_TIMEOUT,
            'stop_loss_setup_interval': SystemConfig.STOP_LOSS_SETUP_INTERVAL
        }
        save_system_config(system_config_data)
        
        # 4. Сохраняем кэш RSI данных
        logger.info(" 📈 Сохранение кэша RSI данных...")
        save_rsi_cache()
        
        # 5. Сохраняем состояние процессов
        logger.info(" 🔄 Сохранение состояния процессов...")
        save_process_state()
        
        # 6. Сохраняем данные о зрелости монет
        logger.info(" 🪙 Сохранение данных о зрелости монет...")
        save_mature_coins_storage()
        
        logger.info(" ✅ Все данные сохранены, очистка завершена")
        
    except Exception as e:
        logger.error(f" ❌ Ошибка при очистке: {e}")
        import traceback
        logger.error(f" Traceback: {traceback.format_exc()}")

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
        
        # ❌ ОТКЛЮЧЕНО: Воркер оптимальных EMA - больше не используется
        # EMA фильтр убран, расчет оптимальных EMA не нужен
        logger.info("ℹ️ Воркер оптимальных EMA отключен (не используется)")
        
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
        logger.warning("[STOP] Получен сигнал прерывания...")
        cleanup_bot_service()
        os._exit(0)
    except Exception as e:
        logger.error(f" Ошибка запуска сервиса ботов: {str(e)}")
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
        logger.error(f" ❌ Ошибка получения детальной информации о ботах: {e}")
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
            logger.error(" bot_history_manager is None!")
            return jsonify({
                'success': False,
                'error': 'Bot history manager not initialized'
            }), 500
        
        symbol = request.args.get('symbol')
        action_type = request.args.get('action_type')
        limit = int(request.args.get('limit', 100))
        period = request.args.get('period')

        if symbol and symbol.lower() == 'all':
            symbol = None
        if action_type:
            action_type = action_type.upper()
        if period and period.lower() == 'all':
            period = None
        
        history = bot_history_manager.get_bot_history(
            symbol,
            action_type,
            limit,
            period
        )
        
        return jsonify({
            'success': True,
            'history': history,
            'count': len(history)
        })
        
    except Exception as e:
        logger.error(f" Ошибка получения истории ботов: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/trades', methods=['GET'])
def get_bot_trades():
    """Получает историю торговых сделок ботов"""
    try:
        symbol = request.args.get('symbol')
        trade_type = request.args.get('trade_type')
        limit = int(request.args.get('limit', 100))
        period = request.args.get('period')

        if symbol and symbol.lower() == 'all':
            symbol = None
        if trade_type:
            trade_type = trade_type.upper()
        if period and period.lower() == 'all':
            period = None
        
        trades = bot_history_manager.get_bot_trades(
            symbol,
            trade_type,
            limit,
            period
        )
        
        return jsonify({
            'success': True,
            'trades': trades,
            'count': len(trades)
        })
        
    except Exception as e:
        logger.error(f" Ошибка получения сделок ботов: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== AI MODULE ENDPOINTS ====================

@bots_app.route('/api/ai/decisions', methods=['GET'])
def get_ai_decisions():
    """Получает список решений AI"""
    try:
        status = request.args.get('status')  # SUCCESS, FAILED, PENDING
        symbol = request.args.get('symbol')
        limit = int(request.args.get('limit', 100))
        
        try:
            from bot_engine.ai.ai_data_storage import AIDataStorage
            storage = AIDataStorage()
            decisions = storage.get_ai_decisions(status=status, symbol=symbol)
            
            # Ограничиваем количество
            decisions = decisions[:limit]
            
            return jsonify({
                'success': True,
                'decisions': decisions,
                'count': len(decisions)
            })
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'AI data storage not available',
                'decisions': [],
                'count': 0
            })
        
    except Exception as e:
        logger.error(f" Ошибка получения решений AI: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# Endpoint /api/ai/performance регистрируется в bot_engine/api/endpoints_ai.py через register_ai_endpoints().
# Дубликат здесь удалён во избежание AssertionError: View function mapping is overwriting an existing endpoint.


@bots_app.route('/api/ai/training-history', methods=['GET'])
def get_ai_training_history():
    """Получает историю обучения AI"""
    try:
        limit = int(request.args.get('limit', 50))
        
        try:
            from bot_engine.ai.ai_data_storage import AIDataStorage
            storage = AIDataStorage()
            history = storage.get_training_history(limit=limit)
            
            return jsonify({
                'success': True,
                'history': history,
                'count': len(history)
            })
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'AI data storage not available',
                'history': [],
                'count': 0
            })
        
    except Exception as e:
        logger.error(f" Ошибка получения истории обучения AI: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/ai/stats', methods=['GET'])
def get_ai_stats():
    """Получает статистику AI vs скриптовые правила"""
    try:
        symbol = request.args.get('symbol')  # Опциональный фильтр по символу
        period = request.args.get('period')  # Опциональный период: 24h|7d|30d|all
        if period and period.lower() == 'all':
            period = 'all'
        if not period:
            period = 'all'
        
        # Получаем все сделки из истории
        if bot_history_manager is None:
            return jsonify({
                'success': False,
                'error': 'Bot history manager not initialized'
            }), 500
        
        # Берем максимально возможный срез и отключаем период, чтобы не потерять сделки
        # В памяти BotHistoryManager хранит до 5000 последних сделок; поднимем лимит и уберем период
        trades = bot_history_manager.get_bot_trades(
            symbol=symbol,
            trade_type=None,
            limit=500000,
            period=period
        )

        # Fallback: добираем сделки из action-истории (POSITION_CLOSED), если по каким-то причинам
        # массив trades содержит меньше записей (например, старые записи только в history)
        try:
            history_actions = bot_history_manager.get_bot_history(
                symbol=symbol,
                action_type=None,
                limit=500000,
                period=period
            )
            closed_actions = [
                h for h in history_actions
                if (h.get('action_type') or '').upper() == 'POSITION_CLOSED'
            ]
            # Преобразуем закрытия в формат trade-подобных словарей (если их нет в trades)
            def _to_trade_from_close(entry):
                return {
                    'id': entry.get('id'),
                    'timestamp': entry.get('timestamp'),
                    'bot_id': entry.get('bot_id'),
                    'symbol': entry.get('symbol'),
                    'direction': entry.get('direction'),
                    'size': entry.get('size'),
                    'entry_price': entry.get('entry_price'),
                    'exit_price': entry.get('exit_price'),
                    'pnl': entry.get('pnl'),
                    'roi': entry.get('roi'),
                    'status': 'CLOSED',
                    'close_timestamp': entry.get('timestamp'),
                    'decision_source': entry.get('decision_source', 'SCRIPT'),
                    'ai_decision_id': entry.get('ai_decision_id'),
                    'ai_confidence': entry.get('ai_confidence'),
                    'is_successful': entry.get('is_successful', (entry.get('pnl', 0) or 0) > 0),
                }
            if closed_actions:
                # Создаем множество существующих id, чтобы не дублировать
                existing_ids = {t.get('id') for t in trades if t.get('id')}
                converted = [_to_trade_from_close(a) for a in closed_actions if a.get('id') not in existing_ids]
                if converted:
                    trades.extend(converted)
        except Exception:
            # Безопасно игнорируем любые проблемы с fallback
            pass
        
        ai_trades = []
        script_trades = []
        
        for trade in trades:
            source_raw = trade.get('decision_source')
            source = source_raw.upper() if isinstance(source_raw, str) else 'SCRIPT'
            if source == 'AI':
                ai_trades.append(trade)
            else:
                # Все нерегистрированные или отсутствующие источники считаем скриптовыми
                script_trades.append(trade)
        
        def _compute_stats(items):
            total = len(items)
            successful = sum(1 for t in items if t.get('is_successful', (t.get('pnl') or 0) > 0))
            failed = total - successful
            total_pnl = sum((t.get('pnl') or 0) for t in items)
            avg_pnl = total_pnl / total if total else 0
            win_rate = (successful / total * 100) if total else 0
            return {
                'total': total,
                'successful': successful,
                'failed': failed,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'win_rate': win_rate
            }
        
        ai_stats = _compute_stats(ai_trades)
        script_stats = _compute_stats(script_trades)
        
        return jsonify({
            'success': True,
            'ai': ai_stats,
            'script': script_stats,
            'counts': {
                'ai_total': ai_stats['total'],
                'script_total': script_stats['total'],
                'all_total': len(trades)
            },
            'comparison': {
                'win_rate_diff': ai_stats['win_rate'] - script_stats['win_rate'],
                'avg_pnl_diff': ai_stats['avg_pnl'] - script_stats['avg_pnl'],
                'total_pnl_diff': ai_stats['total_pnl'] - script_stats['total_pnl']
            }
        })
        
    except Exception as e:
        logger.error(f" Ошибка получения статистики AI: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bots_app.route('/api/ai/optimizer/results', methods=['GET'])
def get_ai_optimizer_results():
    """Возвращает последние результаты оптимизатора и лучшие параметры по монетам."""
    try:
        results_dir = os.path.join('data', 'ai', 'optimization_results')
        optimized_path = os.path.join(results_dir, 'optimized_params.json')
        trade_patterns_path = os.path.join(results_dir, 'trade_patterns.json')
        best_params_path = os.path.join('data', 'ai', 'best_params_per_symbol.json')
        genomes_path = os.path.join('data', 'ai', 'optimizer_genomes.json')

        optimized_params, optimized_updated = _load_json_file(optimized_path)
        trade_patterns, patterns_updated = _load_json_file(trade_patterns_path)
        best_params, _ = _load_json_file(best_params_path)
        genome_meta, genome_updated = _load_json_file(genomes_path)

        top_symbols = []
        if isinstance(best_params, dict):
            for symbol, payload in best_params.items():
                rating = payload.get('rating')
                if rating is None:
                    continue
                top_symbols.append({
                    'symbol': symbol,
                    'rating': float(rating),
                    'win_rate': float(payload.get('win_rate', 0.0) or 0.0),
                    'total_pnl': float(payload.get('total_pnl', 0.0) or 0.0),
                    'updated_at': payload.get('updated_at'),
                })
            top_symbols = sorted(top_symbols, key=lambda item: item['rating'], reverse=True)[:10]

        metadata = {
            'optimized_params_updated_at': optimized_updated,
            'trade_patterns_updated_at': patterns_updated,
            'genome_version': (genome_meta or {}).get('version'),
            'genome_source': os.path.relpath(genomes_path) if os.path.exists(genomes_path) else None,
            'max_tests': (genome_meta or {}).get('max_tests'),
            'genome_updated_at': genome_updated,
            'total_symbols_optimized': len(best_params) if isinstance(best_params, dict) else 0,
        }

        return jsonify({
            'success': True,
            'optimized_params': optimized_params,
            'trade_patterns': trade_patterns,
            'top_symbols': top_symbols,
            'metadata': metadata,
        })
    except Exception as exc:
        logger.error(f"[AI_OPTIMIZER] Ошибка получения результатов оптимизации: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 500

@bots_app.route('/api/ai/models', methods=['GET'])
def get_ai_models():
    """Получает информацию о текущих моделях AI"""
    try:
        try:
            from bot_engine.ai.ai_data_storage import AIDataStorage
            import os
            import json
            
            storage = AIDataStorage()
            versions = storage.get_model_versions(limit=10)
            latest_version = storage.get_latest_model_version()
            
            # Проверяем наличие файлов моделей
            models_dir = os.path.join('data', 'ai', 'models')
            models_info = {}
            
            model_files = {
                'signal_predictor': 'signal_predictor.pkl',
                'profit_predictor': 'profit_predictor.pkl',
                'scaler': 'scaler.pkl'
            }
            
            for model_name, filename in model_files.items():
                filepath = os.path.join(models_dir, filename)
                metadata_path = os.path.join(models_dir, f"{model_name}_metadata.json")
                
                exists = os.path.exists(filepath)
                models_info[model_name] = {
                    'exists': exists,
                    'path': filepath,
                    'metadata': None
                }
                
                if exists and os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            models_info[model_name]['metadata'] = json.load(f)
                    except:
                        pass
            
            return jsonify({
                'success': True,
                'models': models_info,
                'versions': versions,
                'latest_version': latest_version
            })
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'AI data storage not available',
                'models': {},
                'versions': []
            })
        
    except Exception as e:
        logger.error(f" Ошибка получения информации о моделях AI: {e}")
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
            logger.warning(f" Не удалось проверить лицензию: {e}")
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
        logger.error(f" Ошибка получения стопов: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bots_app.route('/api/bots/statistics', methods=['GET'])
def get_bot_statistics():
    """Получает статистику по ботам"""
    try:
        symbol = request.args.get('symbol')
        
        period = request.args.get('period')

        if symbol and symbol.lower() == 'all':
            symbol = None
        if period and period.lower() == 'all':
            period = None

        statistics = bot_history_manager.get_bot_statistics(symbol, period)
        
        return jsonify({
            'success': True,
            'statistics': statistics
        })
        
    except Exception as e:
        logger.error(f" Ошибка получения статистики ботов: {e}")
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
        logger.error(f" Ошибка очистки истории ботов: {e}")
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
        logger.error(f" Ошибка создания демо-данных: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bots_app.route('/api/bots/analytics/sync-from-exchange', methods=['POST'])
def sync_trades_from_exchange():
    """Синхронизирует bot_trades_history с данными биржи: обновляет цены и PnL для совпавших сделок."""
    try:
        from bot_engine.trading_analytics import (
            run_full_analytics,
            exchange_trades_to_summaries,
            bot_trades_to_summaries,
            reconcile_trades,
        )
        from app.config import EXCHANGES, ACTIVE_EXCHANGE
        from exchanges.exchange_factory import ExchangeFactory
        from bot_engine.bots_database import get_bots_database

        exchange_name = ACTIVE_EXCHANGE
        cfg = EXCHANGES.get(exchange_name, {})
        if not cfg or not cfg.get('enabled', True):
            return jsonify({'success': False, 'error': f'Биржа {exchange_name} не настроена'}), 400
        api_key = cfg.get('api_key')
        api_secret = cfg.get('api_secret')
        passphrase = cfg.get('passphrase')
        if not api_key or not api_secret:
            return jsonify({'success': False, 'error': 'API ключи не заполнены'}), 400

        exchange = ExchangeFactory.create_exchange(exchange_name, api_key, api_secret, passphrase)
        exchange_trades = exchange.get_closed_pnl(sort_by='time', period='all') or []
        db = get_bots_database()
        bot_trades = db.get_bot_trades_history(status='CLOSED', limit=50000)

        if not bot_trades:
            return jsonify({'success': True, 'updated': 0, 'message': 'Нет сделок в БД для обновления'})

        ex_summaries = exchange_trades_to_summaries(exchange_trades)
        bot_summaries = bot_trades_to_summaries(bot_trades)
        if not ex_summaries:
            return jsonify({'success': True, 'updated': 0, 'message': 'Биржа не вернула сделок'})

        reconciliation = reconcile_trades(ex_summaries, bot_summaries)
        updated = 0
        for m in reconciliation.get('matched', []):
            db_id = m.get('bot_db_id')
            if db_id is None:
                continue
            try:
                db_id = int(db_id)
            except (TypeError, ValueError):
                continue
            ok = db.update_bot_trade_from_exchange(
                trade_id=db_id,
                entry_price=float(m.get('entry_price') or 0),
                exit_price=float(m.get('exit_price') or 0),
                pnl=float(m.get('pnl') or 0),
                position_size_usdt=float(m['position_size_usdt']) if m.get('position_size_usdt') is not None else None,
            )
            if ok:
                updated += 1
        return jsonify({'success': True, 'updated': updated, 'matched': len(reconciliation.get('matched', []))})
    except Exception as e:
        logger.exception('Ошибка синхронизации с биржей: %s', e)
        return jsonify({'success': False, 'error': str(e)}), 500


@bots_app.route('/api/bots/analytics', methods=['GET'])
def get_trading_analytics():
    """Запуск полной аналитики торговли (сделки ботов и опционально биржи)."""
    try:
        from bot_engine.trading_analytics import run_full_analytics
        include_exchange = request.args.get('include_exchange', '0').strip().lower() in ('1', 'true', 'yes')
        limit = int(request.args.get('limit', 10000))
        exchange_instance = None
        if include_exchange:
            try:
                from app.config import EXCHANGES, ACTIVE_EXCHANGE
                from exchanges.exchange_factory import ExchangeFactory
                exchange_name = ACTIVE_EXCHANGE
                cfg = EXCHANGES.get(exchange_name, {})
                if cfg and cfg.get('enabled', True):
                    api_key = cfg.get('api_key')
                    api_secret = cfg.get('api_secret')
                    passphrase = cfg.get('passphrase')
                    if api_key and api_secret:
                        exchange_instance = ExchangeFactory.create_exchange(
                            exchange_name, api_key, api_secret, passphrase
                        )
            except Exception as ex:
                logger.warning("Не удалось подключить биржу для аналитики: %s", ex)
        from bot_engine.trading_analytics import get_analytics_for_ai
        report = run_full_analytics(
            load_bot_trades_from_db=True,
            load_exchange_from_api=include_exchange,
            exchange_instance=exchange_instance,
            exchange_period='all',
            bots_db_limit=limit,
        )
        report['ai_summary'] = get_analytics_for_ai(report)
        return jsonify({'success': True, 'report': report})
    except Exception as e:
        logger.exception("Ошибка аналитики торговли: %s", e)
        return jsonify({'success': False, 'error': str(e)}), 500


@bots_app.route('/api/bots/analytics/rsi-audit', methods=['GET'])
def get_rsi_audit():
    """Аудит RSI входа/выхода: сделки с биржи, RSI в точке входа и выхода, сверка с текущим конфигом.
    LONG: вход корректен при RSI <= порог; SHORT: при RSI >= порог. Вне диапазона — ошибочные входы."""
    try:
        exchange = get_exchange()
        if not exchange:
            return jsonify({'success': False, 'error': 'Биржа не инициализирована'}), 503
        limit = request.args.get('limit', '500')
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 500
        limit = min(max(1, limit), 2000)
        from bot_engine.rsi_audit import run_rsi_audit
        report = run_rsi_audit(exchange, limit=limit, period='all')
        return jsonify({'success': True, 'report': report})
    except Exception as e:
        logger.exception("Ошибка аудита RSI: %s", e)
        return jsonify({'success': False, 'error': str(e)}), 500


# Последний результат ai-reanalyze (для отображения пользователю и polling)
_last_ai_reanalyze_result = {'ts': 0, 'fullai_changes': [], 'ai_retrain': None, 'running': False}


@bots_app.route('/api/bots/analytics/ai-reanalyze', methods=['POST'])
def ai_reanalyze_and_update():
    """
    Ручной запуск: ИИ анализирует закрытые сделки, находит ошибки и успехи, обновляет параметры.
    - Сбрасывает кеш аналитики
    - FullAI learner: анализ сделок, торговой аналитики, обучение на ошибках и успехах
    """
    global _last_ai_reanalyze_result
    try:
        import time
        from bot_engine.ai_analytics import invalidate_analytics_cache
        invalidate_analytics_cache()

        data = request.get_json(silent=True) or {}
        days_back = int(data.get('days_back') or data.get('period_days') or 7)
        symbol_filter = (data.get('symbol') or '').strip().upper() or None
        limit = min(5000, max(500, int(data.get('limit') or 2000)))

        fullai_changes = []
        insights = {'mistakes': [], 'successes': [], 'recommendations': []}
        fullai_msg = 'FullAI выключен, параметры не менялись'
        try:
            from bots_modules.imports_and_globals import bots_data, bots_data_lock
            with bots_data_lock:
                full_ai = (bots_data.get('auto_bot_config') or {}).get('full_ai_control', False)
            if full_ai:
                from bots_modules.fullai_trades_learner import run_fullai_trades_analysis
                r = run_fullai_trades_analysis(
                    days_back=days_back,
                    min_trades_per_symbol=2,
                    adjust_params=True,
                    symbol_filter=symbol_filter,
                    limit=limit,
                )
                fullai_changes = r.get('changes') or []
                insights = r.get('insights') or insights
                analyzed = r.get('analyzed', 0)
                updated_cnt = len(r.get('updated_symbols') or [])
                if fullai_changes:
                    fullai_msg = f"Проанализировано {analyzed} сделок. Обновлено {len(fullai_changes)} параметров по {updated_cnt} монетам."
                else:
                    fullai_msg = f"Проанализировано {analyzed} сделок. Изменений параметров нет."
            else:
                # Даже без FullAI — можно показать что анализ сделок выполнен (торговая аналитика)
                fullai_msg = 'FullAI выключен. Включите FullAI для обновления параметров по сделкам.'
        except Exception as e:
            logger.warning("ai-reanalyze fullai: %s", e)
            fullai_msg = f"FullAI: ошибка — {e}"

        _last_ai_reanalyze_result = {
            'ts': time.time(),
            'fullai_changes': fullai_changes,
            'insights': insights,
            'ai_retrain': None,
            'running': False,
        }

        return jsonify({
            'success': True,
            'message': fullai_msg,
            'started': True,
            'changes': fullai_changes,
            'insights': insights,
        })
    except Exception as e:
        logger.exception("ai-reanalyze: %s", e)
        return jsonify({'success': False, 'error': str(e)}), 500


@bots_app.route('/api/bots/analytics/ai-reanalyze/result', methods=['GET'])
def ai_reanalyze_result():
    """Последний результат ai-reanalyze (для polling после запуска)."""
    global _last_ai_reanalyze_result
    return jsonify({
        'success': True,
        'result': _last_ai_reanalyze_result,
    })


@bots_app.route('/api/bots/analytics/ai-context', methods=['GET'])
def get_ai_analytics_context():
    """Контекст аналитики для ИИ: проблемы, рекомендации, метрики, последние события (для обучения и решений)."""
    try:
        from bot_engine.ai_analytics import get_ai_analytics_context as _get_context
        symbol = request.args.get('symbol', '').strip().upper() or None
        hours = float(request.args.get('hours', 24))
        include_report = request.args.get('include_report', '1').strip().lower() in ('1', 'true', 'yes')
        ctx = _get_context(symbol=symbol, hours_back=hours, include_report=include_report)
        return jsonify({'success': True, 'context': ctx})
    except Exception as e:
        logger.exception("Ошибка получения контекста аналитики для ИИ: %s", e)
        return jsonify({'success': False, 'error': str(e)}), 500


@bots_app.route('/api/bots/analytics/fullai', methods=['GET'])
def get_fullai_analytics():
    """Аналитика FullAI: события и сводка из data/fullai_analytics.db + винрейт/PnL по сделкам бота из bots_data.db."""
    try:
        from bot_engine.fullai_analytics import get_events, get_summary, get_db_info
        symbol = request.args.get('symbol', '').strip().upper() or None
        from_ts = request.args.get('from_ts', type=float)
        to_ts = request.args.get('to_ts', type=float)
        limit = request.args.get('limit', type=int) or 500
        limit = min(max(1, limit), 2000)
        summary = get_summary(symbol=symbol, from_ts=from_ts, to_ts=to_ts)
        events = get_events(symbol=symbol, from_ts=from_ts, to_ts=to_ts, limit=limit)
        db_info = get_db_info()
        bot_trades_stats = _compute_bot_trades_stats(symbol=symbol, from_ts=from_ts, to_ts=to_ts)
        closed_trades = _get_closed_trades_for_table(symbol=symbol, from_ts=from_ts, to_ts=to_ts, limit=limit)
        fullai_configs = {}
        try:
            from bot_engine.bots_database import get_bots_database
            db = get_bots_database()
            fullai_configs = db.load_all_full_ai_configs_for_analytics()
        except Exception as cfg_err:
            pass
        return jsonify({
            'success': True,
            'summary': summary,
            'events': events,
            'closed_trades': closed_trades,
            'bot_trades_stats': bot_trades_stats,
            'fullai_configs': fullai_configs,
            'db_path': db_info['db_path'],
            'total_events': db_info['total_events'],
        })
    except Exception as e:
        logger.exception("Ошибка аналитики FullAI: %s", e)
        return jsonify({'success': False, 'error': str(e)}), 500


def _get_closed_trades_for_table(symbol=None, from_ts=None, to_ts=None, limit=500):
    """
    Закрытые сделки из bots_data.db + виртуальные из app_data.db.
    Отображаются вместе (реальные и виртуальные) с подписью «Виртуальная».
    """
    try:
        from bot_engine.bots_database import get_bots_database
        db = get_bots_database()
        trades = db.get_bot_trades_history(
            symbol=symbol,
            status='CLOSED',
            limit=min(limit, 500),
            from_ts_sec=from_ts,
            to_ts_sec=to_ts,
        ) or []
        out = []
        for t in trades:
            pnl = float(t.get('pnl') or 0)
            roi = t.get('roi')  # может быть % или коэффициент
            roi_pct = None
            if roi is not None:
                roi_f = float(roi)
                roi_pct = roi_f * 100 if 0 < abs(roi_f) < 1.5 else roi_f
            if roi_pct is None and t.get('entry_price') and t.get('exit_price'):
                ep, xp = float(t['entry_price']), float(t['exit_price'])
                if ep > 0 and t.get('direction', '').upper() == 'LONG':
                    roi_pct = ((xp - ep) / ep) * 100
                elif ep > 0 and t.get('direction', '').upper() == 'SHORT':
                    roi_pct = ((ep - xp) / ep) * 100
            reason = t.get('close_reason') or ''
            # ИИ-анализатор выводов: детальный, разнообразный анализ по сделке
            try:
                from bot_engine.ai.trade_conclusion_analyzer import analyze_trade_conclusion
                trade_for_analysis = {
                    **t,
                    'roi': roi_pct if roi_pct is not None else t.get('roi'),
                    'pnl': pnl,
                }
                conclusion = analyze_trade_conclusion(trade_for_analysis)
            except Exception:
                if pnl >= 0:
                    conclusion = 'Прибыль. ' + (reason if reason else 'Закрыто по условию')
                    if reason and any(x in reason.upper() for x in ('TP', 'TAKE_PROFIT', 'ТЕЙК')):
                        conclusion = 'Прибыль. Выход по TP — цель достигнута'
                    elif reason and any(x in reason.upper() for x in ('RSI', 'РСИ')):
                        conclusion = 'Прибыль. Выход по RSI в плюсе — сигнал сработал'
                else:
                    conclusion = 'Убыток. ' + (reason if reason else 'Закрыто по условию')
                    if reason and any(x in reason.upper() for x in ('SL', 'STOP', 'СЛОСС')):
                        conclusion = 'Убыток. Выход по SL — стоп сработал, фиксация убытка'
                    elif reason and any(x in reason.upper() for x in ('RSI', 'РСИ')):
                        conclusion = 'Убыток. Выход по RSI в минусе — возможно ранний выход'
            ts = t.get('exit_timestamp') or t.get('entry_timestamp')
            ts_iso = t.get('exit_time') or t.get('entry_time') or ''
            if ts and not ts_iso:
                try:
                    from datetime import datetime
                    dt = datetime.fromtimestamp(ts / 1000 if ts > 1e10 else ts)
                    ts_iso = dt.strftime('%Y-%m-%dT%H:%M:%S')
                except Exception:
                    pass
            out.append({
                'symbol': t.get('symbol'),
                'direction': t.get('direction'),
                'entry_price': t.get('entry_price'),
                'exit_price': t.get('exit_price'),
                'entry_time': t.get('entry_time'),
                'exit_time': t.get('exit_time'),
                'ts': ts,
                'ts_iso': ts_iso,
                'close_reason': reason,
                'pnl_usdt': round(pnl, 4),
                'roi_pct': round(roi_pct, 2) if roi_pct is not None else None,
                'is_successful': bool(t.get('is_successful')) or pnl > 0,
                'conclusion': conclusion,
                'is_virtual': False,
            })
        # Добавляем виртуальные сделки (FullAI) — в том же списке, с подписью «Виртуальная»
        try:
            from bot_engine.app_database import get_app_database
            app_db = get_app_database()
            virtual_trades = app_db.get_virtual_closed_trades(
                symbol=symbol,
                from_ts_sec=from_ts,
                to_ts_sec=to_ts,
                limit=min(limit, 500),
            )
            out.extend(virtual_trades)
        except Exception:
            pass
        # Сортируем по времени (новые первые) и ограничиваем
        out.sort(key=lambda x: int(x.get('ts') or 0), reverse=True)
        return out[:limit]
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug("closed_trades_for_table: %s", e)
        return []


def _compute_bot_trades_stats(symbol=None, from_ts=None, to_ts=None):
    """Винрейт и суммарный PnL по закрытым сделкам бота из bots_data.db (история ботов — истинный источник)."""
    try:
        from bot_engine.bots_database import get_bots_database
        db = get_bots_database()
        trades = db.get_bot_trades_history(
            symbol=symbol,
            status='CLOSED',
            limit=10000,
            from_ts_sec=from_ts,
            to_ts_sec=to_ts,
        )
        if not trades:
            return {'total': 0, 'wins': 0, 'losses': 0, 'win_rate_pct': None, 'total_pnl_usdt': 0.0}
        total = len(trades)
        wins = sum(1 for t in trades if t.get('is_successful') or (float(t.get('pnl') or 0) > 0))
        losses = total - wins
        total_pnl = sum(float(t.get('pnl') or 0) for t in trades)
        win_rate_pct = (wins / total * 100) if total else None
        return {
            'total': total,
            'wins': wins,
            'losses': losses,
            'win_rate_pct': round(win_rate_pct, 1) if win_rate_pct is not None else None,
            'total_pnl_usdt': round(total_pnl, 2),
        }
    except Exception as e:
        logger.debug("bot_trades_stats: %s", e)
        return {'total': 0, 'wins': 0, 'losses': 0, 'win_rate_pct': None, 'total_pnl_usdt': 0.0}


@bots_app.route('/api/ai/self-learning/stats', methods=['GET'])
def get_ai_self_learning_stats():
    """Получить статистику самообучения AI (доступно с любой лицензией)"""
    try:
        try:
            from bot_engine.ai.ai_self_learning import get_self_learning_system
            self_learning = get_self_learning_system()
            stats = self_learning.get_learning_stats()

            return jsonify({
                'success': True,
                'stats': stats
            })
        except Exception as e:
            logger.error(f"Ошибка получения статистики самообучения: {e}")
            return jsonify({
                'success': False,
                'error': f'Ошибка получения статистики: {str(e)}'
            })

    except Exception as e:
        logger.error(f"Ошибка в API самообучения: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


def _json_safe(obj):
    """Приводит объект к типам, сериализуемым в JSON (numpy.bool_, numpy.float64 и т.д. -> bool/float)."""
    try:
        import numpy as np
        # numpy scalar types (np.bool_ deprecated in 1.20+, но isinstance всё ещё работает)
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if type(obj).__module__ == 'numpy' and hasattr(obj, 'item'):
            return obj.item()  # numpy scalar -> Python scalar
        if isinstance(obj, (bool, int, float, str, type(None))):
            return obj
    except Exception:
        pass
    return obj


@bots_app.route('/api/ai/self-learning/performance', methods=['GET'])
def get_ai_self_learning_performance():
    """Получить метрики производительности AI (доступно с любой лицензией)"""
    try:
        try:
            from bot_engine.ai.ai_self_learning import get_self_learning_system
            self_learning = get_self_learning_system()

            # Получаем последние сделки для анализа
            try:
                from bot_engine.ai.ai_database import get_ai_database
                ai_db = get_ai_database()
                if ai_db:
                    # Получаем последние 100 сделок для анализа производительности
                    trades = ai_db.get_trades_for_training(limit=100)
                    if len(trades) >= 10:
                        performance = self_learning.evaluate_ai_performance(trades)
                        trends = self_learning.get_performance_trends()
                    else:
                        performance = {'error': 'Недостаточно данных для анализа (нужно минимум 10 сделок)'}
                        trends = {'error': 'Недостаточно данных для анализа трендов'}
                else:
                    performance = {'error': 'База данных недоступна'}
                    trends = {'error': 'База данных недоступна'}
            except Exception as db_error:
                logger.warning(f"Ошибка получения сделок из БД: {db_error}")
                performance = {'error': f'Ошибка базы данных: {str(db_error)}'}
                trends = {'error': f'Ошибка базы данных: {str(db_error)}'}

            # Приводим к типам, сериализуемым в JSON (numpy.bool_/float64 и т.д.)
            performance = _json_safe(performance)
            trends = _json_safe(trends)

            return jsonify({
                'success': True,
                'performance': performance,
                'trends': trends
            })
        except Exception as e:
            logger.error(f"Ошибка получения метрик производительности: {e}")
            return jsonify({
                'success': False,
                'error': f'Ошибка получения метрик: {str(e)}'
            })

    except Exception as e:
        logger.error(f"Ошибка в API производительности AI: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


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
    print("=" * 60)
    print("*** Запуск...")
    
    run_bots_service()
