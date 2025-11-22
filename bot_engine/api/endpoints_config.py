"""
API endpoints для конфигурации Auto Bot и системы
"""

from flask import request, jsonify
import logging
from datetime import datetime

logger = logging.getLogger('API_Config')


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
        # Используем print напрямую для ANSI кодов
        print(f"\033[92m[CONFIG] ✓ {display_name}: {old_value} {arrow} {new_value}\033[0m")
        return True
    return False


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
                
                # Проверяем изменение критериев зрелости
                maturity_params_changed = False
                maturity_keys = ['min_candles_for_maturity', 'min_rsi_low', 'max_rsi_high']
                changes_count = 0
                
                with state['bots_data_lock']:
                    old_config = state['bots_data']['auto_bot_config'].copy()
                    
                    for key in maturity_keys:
                        if key in data and data[key] != old_config.get(key):
                            maturity_params_changed = True
                            logger.warning(f"[MATURITY] ⚠️ Изменен критерий зрелости: {key} ({old_config.get(key)} → {data[key]})")
                    
                    # Обновляем конфигурацию
                    for key, value in data.items():
                        if key in state['bots_data']['auto_bot_config']:
                            old_value = state['bots_data']['auto_bot_config'][key]
                            if old_value != value:
                                state['bots_data']['auto_bot_config'][key] = value
                                changes_count += 1
                                
                                # Используем log_config_change с названием из словаря
                                log_config_change(key, old_value, value)
                
                # Сохраняем конфигурацию
                save_result = state['save_auto_bot_config_func']()
                
                # Выводим итоговое сообщение
                if changes_count > 0:
                    print(f"\033[92m[CONFIG] ✅ Auto Bot: изменено параметров: {changes_count}, конфигурация сохранена\033[0m")
                else:
                    logger.info("[CONFIG] ℹ️  Auto Bot: изменений не обнаружено")
                
                # Очищаем зрелые монеты если критерии изменились
                if maturity_params_changed:
                    logger.warning("[MATURITY] Критерии зрелости изменены - очистка файла")
                    state['clear_mature_coins_func']()
                
                # Логируем включение/выключение автобота только при реальном изменении
                if 'enabled' in data and old_config.get('enabled') == False and data['enabled'] == True:
                    logger.info("=" * 80)
                    print("\033[92m🟢 AUTO BOT ВКЛЮЧЕН! 🟢\033[0m")
                    logger.info("=" * 80)
                elif 'enabled' in data and old_config.get('enabled') == True and data['enabled'] == False:
                    logger.info("=" * 80)
                    print("\033[91m🔴 AUTO BOT ВЫКЛЮЧЕН! 🔴\033[0m")
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
                
                # Получаем старую конфигурацию для сравнения
                old_config = state['get_system_config_func']()
                changes_count = 0
                
                # Проверяем изменения
                for key, new_value in data.items():
                    old_value = old_config.get(key)
                    if log_config_change(key, old_value, new_value):
                        changes_count += 1
                
                # Обновляем настройки
                state['update_system_config_func'](data)
                
                # Сохраняем
                saved = state['save_system_config_func'](data)
                
                # Выводим итоговое сообщение
                if changes_count > 0:
                    print(f"\033[92m[CONFIG] ✅ Изменено параметров: {changes_count}, конфигурация сохранена\033[0m")
                else:
                    logger.info("[CONFIG] ℹ️  Изменений не обнаружено")
                
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


