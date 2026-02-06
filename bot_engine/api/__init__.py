"""
API endpoints для управления ботами
"""

from .endpoints_health import register_health_endpoints
from .endpoints_bots import register_bots_endpoints
from .endpoints_config import register_config_endpoints
from .endpoints_rsi import register_rsi_endpoints
from .endpoints_positions import register_positions_endpoints
from .endpoints_mature import register_mature_endpoints
from .endpoints_system import register_system_endpoints

__all__ = [
    'register_health_endpoints',
    'register_bots_endpoints',
    'register_config_endpoints',
    'register_rsi_endpoints',
    'register_positions_endpoints',
    'register_mature_endpoints',
    'register_system_endpoints'
]


def register_all_endpoints(app, state):
    """
    Регистрирует все API endpoints
    
    Args:
        app: Flask приложение
        state: Словарь с зависимостями (глобальные переменные и функции)
    """
    register_health_endpoints(app, state.get('get_state_func'))
    register_bots_endpoints(app, state)
    register_config_endpoints(app, state)
    register_rsi_endpoints(app, state)
    register_positions_endpoints(app, state)
    register_mature_endpoints(app, state)
    register_system_endpoints(app, state)
    
    logger.info("[API] All endpoints registered successfully")

