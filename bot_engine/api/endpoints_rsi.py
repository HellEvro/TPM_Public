"""
API endpoints для RSI данных
"""

from flask import request, jsonify
import logging
import os
import time
import threading

logger = logging.getLogger('API_RSI')


def register_rsi_endpoints(app, state):
    """
    Регистрирует endpoints для RSI данных
    
    Args:
        app: Flask приложение
        state: Словарь с зависимостями
    """
    
    @app.route('/api/bots/coins-with-rsi', methods=['GET'])
    def get_coins_with_rsi():
        """Получить все монеты с RSI 6H данными"""
        try:
            # Проверяем параметр refresh_symbol
            refresh_symbol = request.args.get('refresh_symbol')
            if refresh_symbol:
                logger.info(f"[API] Запрос на обновление RSI для {refresh_symbol}")
                try:
                    if state['ensure_exchange_func']():
                        coin_data = state['get_coin_rsi_func'](refresh_symbol, state['exchange'])
                        if coin_data:
                            with state['rsi_data_lock']:
                                state['coins_rsi_data']['coins'][refresh_symbol] = coin_data
                            logger.info(f"[API] RSI для {refresh_symbol} обновлены")
                except Exception as e:
                    logger.error(f"[API] Ошибка обновления RSI для {refresh_symbol}: {e}")
            
            with state['rsi_data_lock']:
                # Проверяем возраст кэша
                cache_age = None
                RSI_CACHE_FILE = state.get('RSI_CACHE_FILE', 'data/rsi_cache.json')
                if os.path.exists(RSI_CACHE_FILE):
                    try:
                        cache_stat = os.path.getmtime(RSI_CACHE_FILE)
                        cache_age = (time.time() - cache_stat) / 60
                    except:
                        cache_age = None
                
                # Очищаем данные от несериализуемых объектов
                cleaned_coins = {}
                for symbol, coin_data in state['coins_rsi_data']['coins'].items():
                    cleaned_coin = coin_data.copy()
                    
                    # Очищаем enhanced_rsi
                    if 'enhanced_rsi' in cleaned_coin and cleaned_coin['enhanced_rsi']:
                        enhanced_rsi = cleaned_coin['enhanced_rsi'].copy()
                        
                        if 'confirmations' in enhanced_rsi:
                            confirmations = enhanced_rsi['confirmations'].copy()
                            for key, value in confirmations.items():
                                if hasattr(value, 'item'):
                                    confirmations[key] = value.item()
                                elif value is None:
                                    confirmations[key] = None
                            enhanced_rsi['confirmations'] = confirmations
                        
                        if 'adaptive_levels' in enhanced_rsi and enhanced_rsi['adaptive_levels']:
                            if isinstance(enhanced_rsi['adaptive_levels'], tuple):
                                enhanced_rsi['adaptive_levels'] = list(enhanced_rsi['adaptive_levels'])
                        
                        cleaned_coin['enhanced_rsi'] = enhanced_rsi
                    
                    # Добавляем эффективный сигнал
                    effective_signal = state['get_effective_signal_func'](cleaned_coin)
                    cleaned_coin['effective_signal'] = effective_signal
                    
                    cleaned_coins[symbol] = cleaned_coin
                
                # Получаем ручные позиции
                manual_positions = []
                try:
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
                except Exception as e:
                    logger.error(f"[ERROR] Ошибка получения ручных позиций: {str(e)}")
                
                result = {
                    'success': True,
                    'coins': cleaned_coins,
                    'total': len(cleaned_coins),
                    'last_update': state['coins_rsi_data']['last_update'],
                    'update_in_progress': state['coins_rsi_data']['update_in_progress'],
                    'manual_positions': manual_positions,
                    'cache_info': {
                        'cache_exists': os.path.exists(RSI_CACHE_FILE),
                        'cache_age_minutes': round(cache_age, 1) if cache_age else None,
                        'data_source': 'cache' if cache_age and cache_age < 360 else 'live'
                    },
                    'stats': {
                        'total_coins': state['coins_rsi_data']['total_coins'],
                        'successful_coins': state['coins_rsi_data']['successful_coins'],
                        'failed_coins': state['coins_rsi_data']['failed_coins']
                    }
                }
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"[ERROR] Ошибка получения монет с RSI: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/bots/force-rsi-update', methods=['POST'])
    def force_rsi_update():
        """Принудительно обновить RSI данные"""
        try:
            logger.info("[API] Принудительное обновление RSI данных...")
            
            def update_rsi():
                try:
                    state['load_all_coins_rsi_func']()
                    logger.info("[API] RSI данные обновлены")
                except Exception as e:
                    logger.error(f"[API] Ошибка обновления RSI: {e}")
            
            thread = threading.Thread(target=update_rsi, daemon=True)
            thread.start()
            
            return jsonify({
                'success': True,
                'message': 'Обновление RSI данных запущено'
            })
            
        except Exception as e:
            logger.error(f"[ERROR] Ошибка принудительного обновления RSI: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/bots/refresh-rsi/<symbol>', methods=['POST'])
    def refresh_rsi_for_coin(symbol):
        """Обновляет RSI данные для конкретной монеты"""
        try:
            logger.info(f"[API] Обновление RSI для {symbol}...")
            
            if not state['ensure_exchange_func']():
                return jsonify({'success': False, 'error': 'Биржа не инициализирована'}), 500
            
            coin_data = state['get_coin_rsi_func'](symbol, state['exchange'])
            
            if coin_data:
                with state['rsi_data_lock']:
                    state['coins_rsi_data']['coins'][symbol] = coin_data
                
                logger.info(f"[API] RSI для {symbol} обновлены")
                
                return jsonify({
                    'success': True,
                    'message': f'RSI для {symbol} обновлены',
                    'coin_data': coin_data
                })
            else:
                return jsonify({
                    'success': False,
                    'error': f'Не удалось получить RSI для {symbol}'
                }), 500
                
        except Exception as e:
            logger.error(f"[API] Ошибка обновления RSI для {symbol}: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/bots/clear-rsi-cache', methods=['POST'])
    def clear_rsi_cache_endpoint():
        """Очищает RSI кэш"""
        try:
            logger.info("[API] Очистка RSI кэша...")
            
            result = state['clear_rsi_cache_func']()
            
            if result:
                # Очищаем также данные в памяти
                with state['rsi_data_lock']:
                    state['coins_rsi_data']['coins'] = {}
                    state['coins_rsi_data']['last_update'] = None
                
                logger.info("[API] RSI кэш очищен")
                
                return jsonify({
                    'success': True,
                    'message': 'RSI кэш очищен'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Кэш не найден или уже очищен'
                })
                
        except Exception as e:
            logger.error(f"[ERROR] Ошибка очистки RSI кэша: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    logger.info("[API] RSI endpoints registered")


