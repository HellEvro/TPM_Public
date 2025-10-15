"""
API endpoints для конфигурации Auto Bot и системы
"""

from flask import request, jsonify
import logging
from datetime import datetime

logger = logging.getLogger('API_Config')


def register_config_endpoints(app, state):
    """
    Регистрирует endpoints для конфигурации
    
    Args:
        app: Flask приложение
        state: Словарь с зависимостями
    """
    
    @app.route('/api/bots/auto-bot', methods=['GET', 'POST'])
    def auto_bot_config():
        """Получить или обновить конфигурацию Auto Bot"""
        try:
            if request.method == 'GET':
                with state['bots_data_lock']:
                    config = state['bots_data']['auto_bot_config'].copy()
                    return jsonify({
                        'success': True,
                        'config': config
                    })
            
            elif request.method == 'POST':
                data = request.get_json()
                if not data:
                    return jsonify({'success': False, 'error': 'No data provided'}), 400
                
                logger.info(f"[CONFIG] Обновление конфигурации Auto Bot")
                
                # Проверяем изменение критериев зрелости
                maturity_params_changed = False
                maturity_keys = ['min_candles_for_maturity', 'min_rsi_low', 'max_rsi_high']
                
                with state['bots_data_lock']:
                    old_config = state['bots_data']['auto_bot_config'].copy()
                    
                    for key in maturity_keys:
                        if key in data and data[key] != old_config.get(key):
                            maturity_params_changed = True
                            logger.warning(f"[MATURITY] Изменен критерий: {key} ({old_config.get(key)} -> {data[key]})")
                    
                    # Обновляем конфигурацию
                    for key, value in data.items():
                        if key in state['bots_data']['auto_bot_config']:
                            old_value = state['bots_data']['auto_bot_config'][key]
                            state['bots_data']['auto_bot_config'][key] = value
                            logger.info(f"[CONFIG] {key}: {old_value} -> {value}")
                
                # Сохраняем конфигурацию
                save_result = state['save_auto_bot_config_func']()
                
                # Очищаем зрелые монеты если критерии изменились
                if maturity_params_changed:
                    logger.warning("[MATURITY] Критерии зрелости изменены - очистка файла")
                    state['clear_mature_coins_func']()
                
                # Логируем включение/выключение автобота
                if 'enabled' in data:
                    if data['enabled']:
                        logger.info("=" * 80)
                        logger.info("[AUTO_BOT] ВКЛЮЧЕН!")
                        logger.info("=" * 80)
                    else:
                        logger.info("=" * 80)
                        logger.info("[AUTO_BOT] ВЫКЛЮЧЕН!")
                        logger.info("=" * 80)
                
                return jsonify({
                    'success': True,
                    'message': 'Конфигурация Auto Bot обновлена',
                    'config': state['bots_data']['auto_bot_config'].copy(),
                    'saved_to_file': save_result
                })
                
        except Exception as e:
            logger.error(f"[ERROR] Ошибка конфигурации Auto Bot: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/bots/auto-bot/restore-defaults', methods=['POST'])
    def restore_auto_bot_defaults():
        """Восстанавливает дефолтную конфигурацию Auto Bot"""
        try:
            logger.info("[API] Запрос на восстановление дефолтной конфигурации")
            
            result = state['restore_default_config_func']()
            
            if result:
                with state['bots_data_lock']:
                    current_config = state['bots_data']['auto_bot_config'].copy()
                
                return jsonify({
                    'success': True,
                    'message': 'Дефолтная конфигурация восстановлена',
                    'config': current_config
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Ошибка восстановления'
                }), 500
                
        except Exception as e:
            logger.error(f"[ERROR] Ошибка восстановления: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/bots/default-config', methods=['GET'])
    def get_default_config():
        """Получить дефолтную конфигурацию Auto Bot"""
        try:
            default_config = state['load_default_config_func']()
            
            return jsonify({
                'success': True,
                'default_config': default_config,
                'message': 'Дефолтная конфигурация загружена'
            })
            
        except Exception as e:
            logger.error(f"[ERROR] Ошибка загрузки дефолтной конфигурации: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/bots/system-config', methods=['GET', 'POST'])
    def system_config():
        """Получить или обновить системные настройки"""
        try:
            if request.method == 'GET':
                config_data = state['get_system_config_func']()
                return jsonify({
                    'success': True,
                    'config': config_data
                })
            
            elif request.method == 'POST':
                data = request.get_json()
                if not data:
                    return jsonify({'success': False, 'error': 'No data provided'}), 400
                
                logger.info(f"[CONFIG] Обновление системных настроек")
                
                # Обновляем настройки
                state['update_system_config_func'](data)
                
                # Сохраняем
                saved = state['save_system_config_func'](data)
                
                return jsonify({
                    'success': True,
                    'message': 'Системные настройки обновлены',
                    'config': data,
                    'saved_to_file': saved
                })
                
        except Exception as e:
            logger.error(f"[ERROR] Ошибка настройки системы: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    logger.info("[API] Config endpoints registered")


