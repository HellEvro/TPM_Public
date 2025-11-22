"""
API endpoints для работы с зрелыми монетами
"""

from flask import request, jsonify
import logging

logger = logging.getLogger('API_Mature')


def register_mature_endpoints(app, state):
    """
    Регистрирует endpoints для зрелых монет
    
    Args:
        app: Flask приложение
        state: Словарь с зависимостями
    """
    
    @app.route('/api/bots/mature-coins', methods=['GET'])
    def get_mature_coins():
        """Получение списка зрелых монет"""
        try:
            mature_storage = state['get_mature_coins_func']()
            
            return jsonify({
                'success': True,
                'data': {
                    'mature_coins': list(mature_storage.keys()),
                    'count': len(mature_storage),
                    'storage_details': mature_storage
                }
            })
        except Exception as e:
            logger.error(f"[API] Ошибка получения зрелых монет: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/bots/mature-coins-list', methods=['GET'])
    def get_mature_coins_list():
        """Получить список всех зрелых монет"""
        try:
            mature_storage = state['get_mature_coins_func']()
            mature_coins_list = list(mature_storage.keys())
            
            return jsonify({
                'success': True,
                'mature_coins': mature_coins_list,
                'total_count': len(mature_coins_list)
            })
            
        except Exception as e:
            logger.error(f"[API] Ошибка получения списка зрелых монет: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/bots/mature-coins/reload', methods=['POST'])
    def reload_mature_coins():
        """Перезагрузить список зрелых монет из файла"""
        try:
            state['load_mature_coins_func']()
            mature_storage = state['get_mature_coins_func']()
            
            logger.info(f"[API] Перезагружено {len(mature_storage)} зрелых монет")
            
            return jsonify({
                'success': True,
                'message': f'Перезагружено {len(mature_storage)} зрелых монет',
                'data': {
                    'mature_coins': list(mature_storage.keys()),
                    'count': len(mature_storage)
                }
            })
        except Exception as e:
            logger.error(f"[ERROR] Ошибка перезагрузки зрелых монет: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/bots/mature-coins/clear', methods=['POST'])
    def clear_mature_coins():
        """Очистка хранилища зрелых монет"""
        try:
            state['clear_mature_coins_func']()
            logger.info("[API] Хранилище зрелых монет очищено")
            
            return jsonify({
                'success': True,
                'message': 'Хранилище зрелых монет очищено'
            })
        except Exception as e:
            logger.error(f"[API] Ошибка очистки хранилища: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/bots/remove-mature-coins', methods=['POST'])
    def remove_mature_coins_api():
        """Удаление конкретных монет из зрелых"""
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
            
            result = state['remove_mature_coins_func'](coins_to_remove)
            
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
            logger.error(f"[API] Ошибка удаления монет: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/bots/mature-coins/<symbol>', methods=['DELETE'])
    def remove_mature_coin(symbol):
        """Удаление монеты из хранилища зрелых монет"""
        try:
            mature_storage = state['get_mature_coins_func']()
            
            if symbol in mature_storage:
                state['remove_mature_coin_func'](symbol)
                return jsonify({
                    'success': True,
                    'message': f'Монета {symbol} удалена из хранилища'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': f'Монета {symbol} не найдена'
                }), 404
        except Exception as e:
            logger.error(f"[API] Ошибка удаления монеты {symbol}: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    logger.info("[API] Positions endpoints registered")


