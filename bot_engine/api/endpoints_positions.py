"""
API endpoints для работы с позициями
"""

from flask import request, jsonify
import logging

logger = logging.getLogger('API_Positions')


def register_positions_endpoints(app, state):
    """
    Регистрирует endpoints для работы с позициями
    
    Args:
        app: Flask приложение
        state: Словарь с зависимостями
    """
    
    @app.route('/api/bots/account-info', methods=['GET'])
    def get_account_info():
        """Получает информацию о торговом счете"""
        try:
            if not state['ensure_exchange_func']():
                return jsonify({
                    'success': False,
                    'error': 'Exchange not initialized'
                }), 500
            
            # Получаем данные с биржи
            account_info = state['exchange'].get_unified_account_info()
            if not account_info.get("success"):
                account_info = {
                    'success': False,
                    'error': 'Failed to get account info'
                }
            
            # Добавляем информацию о ботах
            with state['bots_data_lock']:
                bots_list = list(state['bots_data']['bots'].values())
                account_info["bots_count"] = len(bots_list)
                account_info["active_bots"] = sum(1 for bot in bots_list 
                                                if bot.get('status') not in ['idle', 'paused'])
            
            response = jsonify(account_info)
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
            
        except Exception as e:
            logger.error(f"[ERROR] Ошибка получения информации о счете: {str(e)}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @app.route('/api/bots/manual-positions/refresh', methods=['POST'])
    def refresh_manual_positions():
        """Обновить список монет с ручными позициями"""
        try:
            manual_positions = []
            if state['exchange']:
                exchange_positions = state['exchange'].get_positions()
                if isinstance(exchange_positions, tuple):
                    positions_list = exchange_positions[0] if exchange_positions else []
                else:
                    positions_list = exchange_positions if exchange_positions else []
                
                for pos in positions_list:
                    if abs(float(pos.get('size', 0))) > 0:
                        symbol = pos.get('symbol', '')
                        clean_symbol = symbol.replace('USDT', '') if symbol else ''
                        if clean_symbol and clean_symbol not in manual_positions:
                            manual_positions.append(clean_symbol)
                
                logger.info(f"[MANUAL_POSITIONS] Обновлено {len(manual_positions)} монет")
                
            return jsonify({
                'success': True,
                'count': len(manual_positions),
                'positions': manual_positions
            })
            
        except Exception as e:
            logger.error(f"[ERROR] Ошибка обновления ручных позиций: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/bots/sync-positions', methods=['POST'])
    def sync_positions_manual():
        """Принудительная синхронизация позиций с биржей"""
        try:
            result = state['sync_positions_func']()
            
            if result:
                return jsonify({
                    'success': True,
                    'message': 'Синхронизация позиций выполнена',
                    'synced': True
                })
            else:
                return jsonify({
                    'success': True,
                    'message': 'Синхронизация не потребовалась',
                    'synced': False
                })
                
        except Exception as e:
            logger.error(f"[MANUAL_SYNC] Ошибка синхронизации: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/bots/active-detailed', methods=['GET'])
    def get_active_bots_detailed():
        """Получает детальную информацию о активных ботах"""
        try:
            with state['bots_data_lock']:
                active_bots = []
                for symbol, bot_data in state['bots_data']['bots'].items():
                    if bot_data.get('status') in ['armed_up', 'armed_down', 'in_position_long', 'in_position_short']:
                        # Получаем текущую цену
                        current_price = None
                        with state['rsi_data_lock']:
                            coin_data = state['coins_rsi_data']['coins'].get(symbol)
                            if coin_data:
                                current_price = coin_data.get('price')
                        
                        active_bots.append({
                            'symbol': symbol,
                            'status': bot_data.get('status'),
                            'position_size': bot_data.get('position_size', 0),
                            'pnl': bot_data.get('pnl', 0),
                            'current_price': current_price,
                            'entry_price': bot_data.get('entry_price'),
                            'created_at': bot_data.get('created_at'),
                            'last_update': bot_data.get('last_update')
                        })
                
                return jsonify({
                    'success': True,
                    'bots': active_bots,
                    'total': len(active_bots)
                })
                
        except Exception as e:
            logger.error(f"[API] Ошибка получения детальной информации: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    logger.info("[API] Positions endpoints registered")


