"""
Health check и статус endpoints
"""

from flask import jsonify
from datetime import datetime
import logging

logger = logging.getLogger('API_Health')


def register_health_endpoints(app, get_state_func):
    """
    Регистрирует health check endpoints
    
    Args:
        app: Flask приложение
        get_state_func: Функция для получения состояния системы
    """
    
    @app.route('/health', methods=['GET'])
    @app.route('/api/bots/health', methods=['GET'])
    def health_check():
        """Проверка состояния сервиса"""
        try:
            state = get_state_func()
            return jsonify({
                'status': 'ok',
                'service': 'bots',
                'timestamp': datetime.now().isoformat(),
                'exchange_connected': state['exchange_connected'],
                'coins_loaded': state['coins_loaded'],
                'bots_active': state['bots_active']
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'service': 'bots',
                'error': str(e)
            }), 500
    
    @app.route('/api/status', methods=['GET'])
    def api_status():
        """API endpoint для проверки статуса сервиса ботов"""
        return jsonify({
            'status': 'online',
            'service': 'bots',
            'timestamp': datetime.now().isoformat(),
            'test': 'simple_endpoint'
        })
    
    @app.route('/api/bots/async-status', methods=['GET'])
    def get_async_status():
        """Получает статус асинхронного процессора"""
        try:
            state = get_state_func()
            async_processor = state.get('async_processor')
            async_processor_task = state.get('async_processor_task')
            
            status = {
                'available': state.get('async_available', False),
                'running': async_processor is not None and getattr(async_processor, 'is_running', False),
                'task_active': async_processor_task is not None and async_processor_task.is_alive(),
                'last_update': getattr(async_processor, 'last_update', 0) if async_processor else 0,
                'active_tasks': len(getattr(async_processor, 'active_tasks', [])) if async_processor else 0
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
    
    @app.route('/api/bots/process-state', methods=['GET'])
    def get_process_state():
        """Получить состояние всех процессов системы"""
        try:
            state = get_state_func()
            return jsonify({
                'success': True,
                'process_state': state.get('process_state', {}),
                'system_info': {
                    'smart_rsi_manager_running': state.get('smart_rsi_running', False),
                    'exchange_initialized': state.get('exchange_connected', False),
                    'total_bots': state.get('bots_active', 0),
                    'auto_bot_enabled': state.get('auto_bot_enabled', False),
                    'mature_coins_storage_size': state.get('mature_coins_count', 0)
                }
            })
        except Exception as e:
            logger.error(f"[ERROR] Ошибка получения состояния процессов: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    logger.info("[API] Health endpoints registered")

