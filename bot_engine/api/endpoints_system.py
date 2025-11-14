"""
API endpoints для системных операций
"""

from flask import request, jsonify
import logging
import importlib
import sys

logger = logging.getLogger('API_System')


def register_system_endpoints(app, state):
    """
    Регистрирует endpoints для системных операций
    
    Args:
        app: Flask приложение
        state: Словарь с зависимостями
    """
    
    @app.route('/api/bots/reload-modules', methods=['POST'])
    def reload_modules_endpoint():
        """Перезагружает модули без перезапуска сервера"""
        try:
            logger.info("[HOT_RELOAD] Начинаем перезагрузку модулей...")
            
            # Сохраняем важные переменные
            saved_exchange = state['exchange']
            saved_system_initialized = state.get('system_initialized', True)
            
            # Находим модули для перезагрузки
            modules_to_reload = [m for m in sys.modules.keys() 
                               if 'bot' in m.lower() and not m.startswith('_')]
            
            reloaded_count = 0
            for module_name in modules_to_reload:
                try:
                    if module_name in sys.modules:
                        importlib.reload(sys.modules[module_name])
                        reloaded_count += 1
                except Exception as e:
                    logger.warning(f"[HOT_RELOAD] Не удалось перезагрузить {module_name}: {e}")
            
            # Восстанавливаем переменные
            state['exchange'] = saved_exchange
            state['system_initialized'] = saved_system_initialized
            
            logger.info(f"[HOT_RELOAD] Перезагружено {reloaded_count} модулей")
            
            return jsonify({
                'success': True, 
                'message': f'Перезагружено {reloaded_count} модулей',
                'reloaded_modules': reloaded_count
            })
            
        except Exception as e:
            logger.error(f"[API] Ошибка перезагрузки модулей: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/bots/reload-config', methods=['POST'])
    def reload_config_endpoint():
        """Перезагружает конфигурацию из файла"""
        try:
            logger.info("[API] Перезагрузка конфигурации...")
            
            state['load_auto_bot_config_func']()
            state['load_system_config_func']()
            
            logger.info("[API] Конфигурация перезагружена")
            
            return jsonify({
                'success': True,
                'message': 'Конфигурация перезагружена из файла'
            })
            
        except Exception as e:
            logger.error(f"[API] Ошибка перезагрузки конфигурации: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/bots/restart-service', methods=['POST'])
    def restart_service_endpoint():
        """Перезапускает сервис ботов"""
        try:
            logger.info("[HOT_RELOAD] Перезапуск сервиса...")
            
            # Сбрасываем флаг инициализации
            state['system_initialized'] = False
            
            # Перезагружаем конфигурацию
            state['load_auto_bot_config_func']()
            state['load_system_config_func']()
            state['load_bots_state_func']()
            
            # Восстанавливаем флаг
            state['system_initialized'] = True
            
            logger.info("[HOT_RELOAD] Сервис перезапущен")
            
            return jsonify({
                'success': True, 
                'message': 'Сервис ботов перезапущен'
            })
            
        except Exception as e:
            logger.error(f"[API] Ошибка перезапуска сервиса: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/bots/cleanup-inactive', methods=['POST'])
    def cleanup_inactive_manual():
        """Принудительная очистка неактивных ботов"""
        try:
            logger.info("[API] Запуск очистки неактивных ботов")
            result = state['cleanup_inactive_func']()
            
            return jsonify({
                'success': True,
                'message': 'Очистка выполнена' if result else 'Неактивных ботов не найдено',
                'cleaned': result
            })
                
        except Exception as e:
            logger.error(f"[API] Ошибка очистки: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/bots/test-exit-scam/<symbol>', methods=['GET'])
    def test_exit_scam_endpoint(symbol):
        """Тестирует ExitScam фильтр для монеты"""
        try:
            state['test_exit_scam_func'](symbol)
            return jsonify({
                'success': True, 
                'message': f'Тест ExitScam для {symbol} выполнен'
            })
        except Exception as e:
            logger.error(f"[API] Ошибка тестирования ExitScam для {symbol}: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/bots/test-rsi-time-filter/<symbol>', methods=['GET'])
    def test_rsi_time_filter_endpoint(symbol):
        """Тестирует RSI временной фильтр для монеты"""
        try:
            state['test_rsi_time_func'](symbol)
            return jsonify({
                'success': True, 
                'message': f'Тест RSI временного фильтра для {symbol} выполнен'
            })
        except Exception as e:
            logger.error(f"[API] Ошибка тестирования RSI фильтра для {symbol}: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/bots/process-trading-signals', methods=['POST'])
    def process_trading_signals_endpoint():
        """Принудительно обработать торговые сигналы"""
        try:
            logger.info("[API] Принудительная обработка торговых сигналов...")
            
            state['process_signals_func'](exchange_obj=state['exchange'])
            
            with state['bots_data_lock']:
                active_bots = {symbol: bot for symbol, bot in state['bots_data']['bots'].items() 
                              if bot['status'] not in ['idle', 'paused']}
            
            logger.info(f"[API] Обработка завершена для {len(active_bots)} ботов")
            
            return jsonify({
                'success': True,
                'message': f'Обработка завершена для {len(active_bots)} ботов',
                'active_bots_count': len(active_bots)
            })
            
        except Exception as e:
            logger.error(f"[API] Ошибка обработки сигналов: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/bots/reset-update-flag', methods=['POST'])
    def reset_update_flag():
        """Сбросить флаг update_in_progress"""
        try:
            with state['rsi_data_lock']:
                was_in_progress = state['coins_rsi_data']['update_in_progress']
                state['coins_rsi_data']['update_in_progress'] = False
                
            logger.info(f"[API] Флаг update_in_progress сброшен (был: {was_in_progress})")
            
            return jsonify({
                'success': True,
                'message': 'Флаг сброшен',
                'was_in_progress': was_in_progress
            })
            
        except Exception as e:
            logger.error(f"[API] Ошибка сброса флага: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    logger.info("[API] System endpoints registered")


