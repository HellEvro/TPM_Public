"""
Модуль для расчета масштабируемых ордеров (лесенка)
"""
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class ScalingCalculator:
    """Калькулятор для расчета уровней лесенки"""
    
    def __init__(self, min_usdt_per_trade: float = 5.0):
        self.min_usdt_per_trade = min_usdt_per_trade
    
    def calculate_scaling_levels(
        self, 
        base_usdt: float, 
        price: float, 
        scaling_mode: str = "auto_double",
        start_percent: float = 1.0,
        max_levels: int = 5,
        manual_percentages: List[float] = None
    ) -> Dict:
        """
        Рассчитывает уровни лесенки с учетом минимального лимита биржи
        
        Args:
            base_usdt: Базовый объем в USDT
            price: Цена монеты
            scaling_mode: Режим расчета ("auto_double" или "manual_percentages")
            start_percent: Стартовый процент для auto_double
            max_levels: Максимум уровней
            manual_percentages: Ручные проценты для manual_percentages
            
        Returns:
            Словарь с результатами расчета
        """
        try:
            base_coins = base_usdt / price
            
            if scaling_mode == "auto_double":
                return self._calculate_auto_double_levels(
                    base_coins, price, start_percent, max_levels
                )
            elif scaling_mode == "manual_percentages":
                return self._calculate_manual_levels(
                    base_coins, price, manual_percentages or []
                )
            else:
                return {
                    'success': False,
                    'error': f'Неизвестный режим масштабирования: {scaling_mode}',
                    'levels': [],
                    'recommendation': None
                }
                
        except Exception as e:
            logger.error(f"[SCALING] Ошибка расчета лесенки: {e}")
            return {
                'success': False,
                'error': str(e),
                'levels': [],
                'recommendation': None
            }
    
    def _calculate_auto_double_levels(
        self, 
        base_coins: float, 
        price: float, 
        start_percent: float, 
        max_levels: int
    ) -> Dict:
        """Рассчитывает уровни с автоматическим удвоением и распределением 100%"""
        
        base_usdt = base_coins * price
        
        # Рассчитываем множители с удвоением: 1x, 2x, 4x, 8x, 16x...
        multipliers = []
        multiplier = 1
        for i in range(max_levels):
            multipliers.append(multiplier)
            multiplier *= 2
        
        # Сумма всех множителей (например, для 5 уровней: 1+2+4+8+16 = 31)
        total_multiplier = sum(multipliers)
        
        # Рассчитываем базовый процент (1x)
        base_percent = 100.0 / total_multiplier
        
        # Рассчитываем уровни
        levels = []
        total_percent = 0
        total_usdt_check = 0
        
        for i, mult in enumerate(multipliers):
            percent = base_percent * mult
            coins_amount = base_coins * (percent / 100)
            usdt_amount = coins_amount * price
            
            levels.append({
                'percent': percent,
                'coins': coins_amount,
                'usdt': usdt_amount,
                'multiplier': mult
            })
            
            total_percent += percent
            total_usdt_check += usdt_amount
        
        # Проверяем, что все уровни >= min_usdt_per_trade
        invalid_levels = [level for level in levels if level['usdt'] < self.min_usdt_per_trade]
        
        if invalid_levels:
            # Есть уровни меньше минимума - рассчитываем рекомендацию
            recommendation = self._calculate_recommendation(base_coins, price, max_levels)
            
            # Показываем, какие уровни не прошли
            invalid_info = [f"Уровень {i+1} ({level['multiplier']}x): {level['usdt']:.2f} USDT < {self.min_usdt_per_trade} USDT" 
                           for i, level in enumerate(levels) if level['usdt'] < self.min_usdt_per_trade]
            
            return {
                'success': False,
                'error': f'Недостаточный базовый объем для {max_levels} сделок. Минимум: {recommendation["min_base_usdt"]:.2f} USDT',
                'levels': levels,  # Возвращаем все уровни для отладки
                'invalid_levels': invalid_info,
                'recommendation': recommendation
            }
        
        # Все уровни валидны
        return {
            'success': True,
            'levels': levels,
            'total_percent': round(total_percent, 2),
            'total_usdt': round(total_usdt_check, 2),
            'recommendation': None
        }
    
    def _calculate_manual_levels(
        self, 
        base_coins: float, 
        price: float, 
        manual_percentages: List[float]
    ) -> Dict:
        """Рассчитывает уровни по ручным процентам с нормализацией до 100%"""
        
        if not manual_percentages:
            return {
                'success': False,
                'error': 'Не указаны ручные проценты',
                'levels': [],
                'recommendation': None
            }
        
        base_usdt = base_coins * price
        
        # Нормализуем проценты до 100%
        total_manual_percent = sum(manual_percentages)
        
        if total_manual_percent == 0:
            return {
                'success': False,
                'error': 'Сумма процентов равна 0',
                'levels': [],
                'recommendation': None
            }
        
        # Нормализуем каждый процент
        normalized_percentages = [(p / total_manual_percent) * 100 for p in manual_percentages]
        
        # Рассчитываем уровни
        levels = []
        total_percent = 0
        total_usdt_check = 0
        
        for i, percent in enumerate(normalized_percentages):
            coins_amount = base_coins * (percent / 100)
            usdt_amount = coins_amount * price
            
            levels.append({
                'percent': percent,
                'coins': coins_amount,
                'usdt': usdt_amount,
                'original_percent': manual_percentages[i]
            })
            
            total_percent += percent
            total_usdt_check += usdt_amount
        
        # Проверяем, что все уровни >= min_usdt_per_trade
        invalid_levels = [level for level in levels if level['usdt'] < self.min_usdt_per_trade]
        
        if invalid_levels:
            # Есть уровни меньше минимума - рассчитываем рекомендацию
            recommendation = self._calculate_recommendation(base_coins, price, len(manual_percentages))
            
            # Показываем, какие уровни не прошли
            invalid_info = [f"Уровень {i+1} ({level['original_percent']}% → {level['percent']:.2f}%): {level['usdt']:.2f} USDT < {self.min_usdt_per_trade} USDT" 
                           for i, level in enumerate(levels) if level['usdt'] < self.min_usdt_per_trade]
            
            return {
                'success': False,
                'error': f'Недостаточный базовый объем для {len(manual_percentages)} сделок. Минимум: {recommendation["min_base_usdt"]:.2f} USDT',
                'levels': levels,
                'invalid_levels': invalid_info,
                'recommendation': recommendation
            }
        
        # Все уровни валидны
        return {
            'success': True,
            'levels': levels,
            'total_percent': round(total_percent, 2),
            'total_usdt': round(total_usdt_check, 2),
            'recommendation': None
        }
    
    def _calculate_recommendation(
        self, 
        base_coins: float, 
        price: float, 
        max_levels: int
    ) -> Dict:
        """Рассчитывает рекомендацию минимального объема"""
        
        # Рассчитываем минимальный объем для 2 уровней с удвоением
        # Уровень 1: 5% = min_usdt_per_trade
        # Уровень 2: 10% = min_usdt_per_trade * 2
        
        min_percent_for_min_usdt = 5  # Минимальный процент для достижения min_usdt_per_trade
        
        # Рассчитываем минимальный базовый объем
        min_base_coins = (self.min_usdt_per_trade * 100) / (min_percent_for_min_usdt * price)
        min_base_usdt = min_base_coins * price
        
        # Рассчитываем для максимального количества уровней
        max_percent_needed = min_percent_for_min_usdt * (2 ** (max_levels - 1))
        max_base_coins = (self.min_usdt_per_trade * 100) / (min_percent_for_min_usdt * price)
        max_base_usdt = max_base_coins * price
        
        return {
            'min_base_usdt': min_base_usdt,
            'max_base_usdt': max_base_usdt,
            'min_levels': 2,
            'max_levels': max_levels,
            'example_levels': [
                {'percent': 5, 'usdt': self.min_usdt_per_trade},
                {'percent': 10, 'usdt': self.min_usdt_per_trade * 2},
                {'percent': 20, 'usdt': self.min_usdt_per_trade * 4}
            ]
        }

def get_scaling_hints(min_usdt_per_trade: float = 5.0) -> Dict:
    """
    Возвращает подсказки по минимальным объемам для разного количества сделок
    
    Args:
        min_usdt_per_trade: Минимальный объем на сделку
        
    Returns:
        Словарь с подсказками для 2, 3, 5, 10 сделок
    """
    hints = {}
    
    for num_trades in [2, 3, 5, 10]:
        # Рассчитываем прогрессию с удвоением, начиная с 1%
        # Уровень 1: 5% = min_usdt_per_trade
        # Значит base_usdt = min_usdt_per_trade * 100 / 5 = min_usdt_per_trade * 20
        
        # Но нужно найти такой стартовый процент, чтобы первая сделка была >= min_usdt_per_trade
        # Пробуем разные стартовые проценты
        for start_pct in [1, 2, 5, 10, 20, 50]:
            percentages = []
            current_pct = start_pct
            
            for i in range(num_trades):
                percentages.append(current_pct)
                current_pct *= 2
            
            # Для первого процента требуется: base_usdt * (start_pct / 100) >= min_usdt_per_trade
            # Значит: base_usdt >= min_usdt_per_trade * 100 / start_pct
            min_base_usdt = (min_usdt_per_trade * 100) / start_pct
            
            # Рассчитываем все уровни
            levels = []
            total_usdt = 0
            for pct in percentages:
                usdt_amount = min_base_usdt * (pct / 100)
                levels.append({
                    'percent': pct,
                    'usdt': usdt_amount
                })
                total_usdt += usdt_amount
            
            # Проверяем, что все уровни >= min_usdt_per_trade
            if all(level['usdt'] >= min_usdt_per_trade for level in levels):
                hints[f'{num_trades}_trades'] = {
                    'min_base_usdt': round(min_base_usdt, 2),
                    'total_usdt': round(total_usdt, 2),
                    'start_percent': start_pct,
                    'levels': [f"{level['percent']}% = {level['usdt']:.2f} USDT" for level in levels]
                }
                break
    
    return hints

def calculate_scaling_for_bot(
    base_usdt: float,
    price: float,
    scaling_config: Dict
) -> Dict:
    """
    Основная функция для расчета лесенки для бота
    
    Args:
        base_usdt: Базовый объем в USDT
        price: Цена монеты
        scaling_config: Конфигурация масштабирования
        
    Returns:
        Результат расчета лесенки
    """
    if not scaling_config.get('scaling_enabled', False):
        return {
            'success': False,
            'error': 'Масштабирование отключено',
            'levels': [],
            'recommendation': None
        }
    
    calculator = ScalingCalculator(
        min_usdt_per_trade=scaling_config.get('scaling_min_usdt_per_trade', 5.0)
    )
    
    # Определяем режим: автоматический или ручной
    manual_percentages = scaling_config.get('scaling_manual_percentages', [])
    number_of_trades = scaling_config.get('scaling_number_of_trades', 0)
    
    if manual_percentages:
        # РУЧНОЙ РЕЖИМ: используем указанные проценты
        scaling_mode = 'manual_percentages'
        max_levels = len(manual_percentages)
    elif number_of_trades > 0:
        # АВТОМАТИЧЕСКИЙ РЕЖИМ: рассчитываем прогрессию
        scaling_mode = 'auto_double'
        max_levels = number_of_trades
    else:
        return {
            'success': False,
            'error': 'Не указано ни количество сделок, ни проценты',
            'levels': [],
            'recommendation': None
        }
    
    return calculator.calculate_scaling_levels(
        base_usdt=base_usdt,
        price=price,
        scaling_mode=scaling_mode,
        start_percent=1.0,  # Начинаем с 1% для автоматического режима
        max_levels=max_levels,
        manual_percentages=manual_percentages
    )
