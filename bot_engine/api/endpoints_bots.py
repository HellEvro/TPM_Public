"""
API endpoints для управления ботами
"""

from flask import request, jsonify
import logging

logger = logging.getLogger('API_Bots')


def register_bots_endpoints(app, state):
    """
    Регистрирует endpoints для управления ботами
    
    Args:
        app: Flask приложение
        state: Словарь с зависимостями:
            - bots_data: данные ботов
            - bots_data_lock: блокировка
            - exchange: объект биржи
            - ensure_exchange_func: функция проверки биржи
            - create_bot_func: функция создания бота
            - save_bots_func: функция сохранения
            - update_cache_func: функция обновления кэша
            - log_bot_stop_func: функция логирования
            - check_maturity_func: функция проверки зрелости
            - BOT_STATUS: константы статусов
    """
    
    @app.route('/api/bots/list', methods=['GET'])
    def get_bots_list():
        """Получить список всех ботов"""
        try:
            with state['bots_data_lock']:
                bots_list = list(state['bots_data']['bots'].values())
                auto_bot_enabled = state['bots_data'].get('auto_bot_config', {}).get('enabled', False)
                last_update = state['bots_data'].get('last_update', 'Неизвестно')
            
            active_bots = sum(1 for bot in bots_list if bot.get('status') not in ['idle', 'paused'])
            
            return jsonify({
                'success': True,
                'bots': bots_list,
                'count': len(bots_list),
                'auto_bot_enabled': auto_bot_enabled,
                'last_update': last_update,
                'stats': {
                    'active_bots': active_bots,
                    'total_bots': len(bots_list)
                }
            })
            
        except Exception as e:
            logger.error(f"[API] Ошибка получения списка ботов: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'bots': [],
                'count': 0
            }), 500
    
    @app.route('/api/bots/create', methods=['POST'])
    def create_bot_endpoint():
        """Создать нового бота"""
        try:
            if not state['ensure_exchange_func']():
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
            
            # Проверяем зрелость если включена проверка
            enable_maturity_check = config.get('enable_maturity_check', True)
            if enable_maturity_check:
                chart_response = state['exchange'].get_chart_data(symbol, '6h', '30d')
                if chart_response and chart_response.get('success'):
                    candles = chart_response['data']['candles']
                    if candles and len(candles) >= 15:
                        with state['bots_data_lock']:
                            maturity_config = state['bots_data'].get('auto_bot_config', {})
                        
                        maturity_check = state['check_maturity_func'](symbol, candles, maturity_config)
                        if not maturity_check['is_mature']:
                            logger.warning(f"[BOT_CREATE] {symbol}: Монета не прошла проверку зрелости")
                            return jsonify({
                                'success': False, 
                                'error': f'Монета {symbol} не прошла проверку зрелости: {maturity_check["reason"]}',
                                'maturity_details': maturity_check['details']
                            }), 400
            
            # Создаем бота
            bot_config = state['create_bot_func'](symbol, config, exchange_obj=state['exchange'])
            
            logger.info(f"[BOT_CREATE] Бот для {symbol} создан")
            
            return jsonify({
                'success': True,
                'message': f'Бот для {symbol} создан успешно',
                'bot': bot_config
            })
            
        except Exception as e:
            logger.error(f"[ERROR] Ошибка создания бота: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/bots/start', methods=['POST'])
    def start_bot_endpoint():
        """Запустить бота"""
        try:
            data = request.get_json()
            if not data or not data.get('symbol'):
                return jsonify({'success': False, 'error': 'Symbol required'}), 400
            
            symbol = data['symbol']
            
            with state['bots_data_lock']:
                if symbol not in state['bots_data']['bots']:
                    return jsonify({'success': False, 'error': 'Bot not found'}), 404
                
                bot_data = state['bots_data']['bots'][symbol]
                BOT_STATUS = state['BOT_STATUS']
                
                if bot_data['status'] in [BOT_STATUS['PAUSED'], BOT_STATUS['IDLE']]:
                    bot_data['status'] = BOT_STATUS['RUNNING']
                    logger.info(f"[BOT] {symbol}: Бот запущен")
            
            return jsonify({
                'success': True,
                'message': f'Бот для {symbol} запущен'
            })
                
        except Exception as e:
            logger.error(f"[ERROR] Ошибка запуска бота: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/bots/stop', methods=['POST'])
    def stop_bot_endpoint():
        """Остановить бота"""
        try:
            data = request.get_json()
            if not data or not data.get('symbol'):
                return jsonify({'success': False, 'error': 'Symbol required'}), 400
            
            symbol = data['symbol']
            reason = data.get('reason', 'Остановлен пользователем')
            
            with state['bots_data_lock']:
                if symbol not in state['bots_data']['bots']:
                    return jsonify({'success': False, 'error': 'Bot not found'}), 404
                
                bot_data = state['bots_data']['bots'][symbol]
                BOT_STATUS = state['BOT_STATUS']
                
                bot_data['status'] = BOT_STATUS['PAUSED']
                bot_data['position_side'] = None
                bot_data['unrealized_pnl'] = 0.0
                
                logger.info(f"[BOT] {symbol}: Бот остановлен")
            
            # Логируем остановку
            state['log_bot_stop_func'](symbol, reason)
            
            # Сохраняем состояние
            state['save_bots_func']()
            state['update_cache_func']()
            
            return jsonify({
                'success': True, 
                'message': f'Бот для {symbol} остановлен'
            })
            
        except Exception as e:
            logger.error(f"[ERROR] Ошибка остановки бота: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/bots/pause', methods=['POST'])
    def pause_bot_endpoint():
        """Приостановить бота"""
        try:
            data = request.get_json()
            if not data or not data.get('symbol'):
                return jsonify({'success': False, 'error': 'Symbol required'}), 400
            
            symbol = data['symbol']
            
            with state['bots_data_lock']:
                if symbol not in state['bots_data']['bots']:
                    return jsonify({'success': False, 'error': 'Bot not found'}), 404
                
                bot_data = state['bots_data']['bots'][symbol]
                BOT_STATUS = state['BOT_STATUS']
                old_status = bot_data['status']
                
                bot_data['status'] = BOT_STATUS['PAUSED']
                logger.info(f"[BOT] {symbol}: Бот приостановлен (был: {old_status})")
            
            return jsonify({
                'success': True,
                'message': f'Бот для {symbol} приостановлен'
            })
            
        except Exception as e:
            logger.error(f"[ERROR] Ошибка приостановки бота: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/bots/delete', methods=['POST'])
    def delete_bot_endpoint():
        """Удалить бота"""
        try:
            data = request.get_json()
            if not data or not data.get('symbol'):
                return jsonify({'success': False, 'error': 'Symbol required'}), 400
            
            symbol = data['symbol']
            reason = data.get('reason', 'Удален пользователем')
            
            with state['bots_data_lock']:
                if symbol not in state['bots_data']['bots']:
                    return jsonify({'success': False, 'error': 'Bot not found'}), 404
                
                bot_data = state['bots_data']['bots'][symbol]
                del state['bots_data']['bots'][symbol]
                logger.info(f"[BOT] {symbol}: Бот удален")
                
                # Обновляем статистику
                state['bots_data']['global_stats']['active_bots'] = len([
                    bot for bot in state['bots_data']['bots'].values() 
                    if bot.get('status') in ['running', 'idle']
                ])
            
            # Логируем удаление
            state['log_bot_stop_func'](symbol, f"Удален: {reason}")
            
            # Сохраняем состояние
            state['save_bots_func']()
            state['update_cache_func']()
            
            return jsonify({
                'success': True,
                'message': f'Бот для {symbol} удален'
            })
            
        except Exception as e:
            logger.error(f"[ERROR] Ошибка удаления бота: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/bots/close-position', methods=['POST'])
    def close_position_endpoint():
        """Принудительно закрыть позицию бота"""
        try:
            data = request.get_json()
            if not data or not data.get('symbol'):
                return jsonify({'success': False, 'error': 'Symbol required'}), 400
            
            symbol = data['symbol']
            
            if not state['exchange']:
                return jsonify({'success': False, 'error': 'Exchange not initialized'}), 500
            
            # Получаем позиции с биржи
            positions_response = state['exchange'].get_positions()
            if not positions_response or not positions_response.get('success'):
                return jsonify({'success': False, 'error': 'Failed to get positions'}), 500
            
            positions = positions_response.get('data', [])
            
            # Ищем позиции для символа
            symbol_positions = [pos for pos in positions 
                              if pos['symbol'] == f"{symbol}USDT" and float(pos.get('size', 0)) > 0]
            
            if not symbol_positions:
                return jsonify({
                    'success': False, 
                    'message': f'Позиции для {symbol} не найдены'
                }), 404
            
            # Закрываем позиции
            closed_positions = []
            errors = []
            
            for pos in symbol_positions:
                try:
                    position_side = 'LONG' if pos['side'] == 'Buy' else 'SHORT'
                    position_size = float(pos['size'])
                    
                    close_result = state['exchange'].close_position(
                        symbol=symbol,
                        size=position_size,
                        side=position_side,
                        order_type="Market"
                    )
                    
                    if close_result and close_result.get('success'):
                        closed_positions.append({
                            'side': position_side,
                            'size': position_size
                        })
                        logger.info(f"[API] Позиция {position_side} для {symbol} закрыта")
                    else:
                        error_msg = close_result.get('message', 'Unknown error') if close_result else 'No response'
                        errors.append(f"Позиция {position_side}: {error_msg}")
                        
                except Exception as e:
                    errors.append(f"Позиция {pos['side']}: {str(e)}")
            
            # Обновляем данные бота
            with state['bots_data_lock']:
                if symbol in state['bots_data']['bots']:
                    bot_data = state['bots_data']['bots'][symbol]
                    if closed_positions:
                        bot_data['position_side'] = None
                        bot_data['unrealized_pnl'] = 0.0
                        bot_data['status'] = state['BOT_STATUS']['IDLE']
            
            state['save_bots_func']()
            state['update_cache_func']()
            
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
    
    logger.info("[API] Bots endpoints registered")


