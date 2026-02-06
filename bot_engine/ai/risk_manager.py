"""
Dynamic Risk Manager - умное управление рисками

Адаптивно рассчитывает стоп-лоссы, тейк-профиты и размеры позиций
на основе анализа волатильности, силы тренда и вероятности разворота.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger('AI.RiskManager')


class DynamicRiskManager:
    """Умное управление рисками на основе AI анализа"""
    
    def __init__(self):
        """Инициализация менеджера рисков"""
        self.logger = logger
        
        # Базовые параметры (из конфигурации)
        from bot_engine.config_loader import RiskConfig
        
        self.base_sl_percent = RiskConfig.STOP_LOSS_PERCENT  # 15%
        self.base_tp_percent = RiskConfig.TRAILING_STOP_ACTIVATION  # 300%
        self.base_trailing_distance = RiskConfig.TRAILING_STOP_DISTANCE  # 150%
        
        # Диапазоны адаптации
        self.sl_min = 8.0    # Минимальный SL (очень низкая волатильность)
        self.sl_max = 25.0   # Максимальный SL (очень высокая волатильность)
        
        self.tp_min = 150.0  # Минимальный TP (слабый тренд)
        self.tp_max = 600.0  # Максимальный TP (очень сильный тренд)
        
        self.trailing_min = 80.0   # Минимальное расстояние (при риске разворота)
        self.trailing_max = 250.0  # Максимальное расстояние (при сильном тренде)
        
        # Параметры анализа
        self.volatility_lookback = 20  # Период для расчёта волатильности
        self.trend_strength_lookback = 10  # Период для оценки силы тренда
        self.reversal_lookback = 15  # Период для предсказания разворота
        
        logger.info("[RiskManager] ✅ Dynamic Risk Manager инициализирован")
    
    def calculate_volatility(self, candles: List[dict]) -> float:
        """
        Рассчитывает волатильность монеты
        
        Args:
            candles: Список свечей
        
        Returns:
            Коэффициент волатильности (0.0-1.0+)
            - 0.0-0.3 = низкая волатильность
            - 0.3-0.7 = средняя волатильность
            - 0.7+ = высокая волатильность
        """
        if len(candles) < self.volatility_lookback:
            return 0.5  # Средняя по умолчанию
        
        # Берём последние N свечей
        recent = candles[-self.volatility_lookback:]
        
        # Рассчитываем изменения цены между свечами
        price_changes = []
        for i in range(1, len(recent)):
            change = abs((recent[i]['close'] - recent[i-1]['close']) / recent[i-1]['close'])
            price_changes.append(change)
        
        # Средняя волатильность (коэффициент вариации)
        if price_changes:
            avg_change = np.mean(price_changes)
            std_change = np.std(price_changes)
            
            # Нормализуем: обычная волатильность ~0.02-0.05 (2-5% между свечами)
            volatility = (avg_change + std_change) / 0.05  # Нормируем к 5%
            
            return min(volatility, 2.0)  # Ограничиваем максимум
        
        return 0.5
    
    def calculate_trend_strength(self, candles: List[dict], direction: str) -> float:
        """
        Рассчитывает силу тренда
        
        Args:
            candles: Список свечей
            direction: 'UP' или 'DOWN'
        
        Returns:
            Сила тренда (0.0-1.0)
            - 0.0-0.3 = слабый тренд
            - 0.3-0.7 = средний тренд
            - 0.7-1.0 = сильный тренд
        """
        if len(candles) < self.trend_strength_lookback + 5:
            return 0.5  # Средняя по умолчанию
        
        recent = candles[-self.trend_strength_lookback:]
        
        # 1. Направленность движения (% свечей в нужном направлении)
        directional_candles = 0
        for candle in recent:
            if direction == 'UP' and candle['close'] > candle['open']:
                directional_candles += 1
            elif direction == 'DOWN' and candle['close'] < candle['open']:
                directional_candles += 1
        
        direction_ratio = directional_candles / len(recent)
        
        # 2. Последовательность движения (нет резких откатов)
        price_changes = []
        for i in range(1, len(recent)):
            change = (recent[i]['close'] - recent[i-1]['close']) / recent[i-1]['close']
            price_changes.append(change)
        
        # Проверяем консистентность направления
        if direction == 'UP':
            positive_changes = sum(1 for c in price_changes if c > 0)
            consistency = positive_changes / len(price_changes)
        else:
            negative_changes = sum(1 for c in price_changes if c < 0)
            consistency = negative_changes / len(price_changes)
        
        # 3. Величина движения
        total_move = abs((recent[-1]['close'] - recent[0]['close']) / recent[0]['close'])
        move_strength = min(total_move / 0.1, 1.0)  # Нормируем к 10%
        
        # Комбинируем факторы
        strength = (direction_ratio * 0.4 + consistency * 0.4 + move_strength * 0.2)
        
        return min(strength, 1.0)
    
    def calculate_dynamic_sl(self, 
                            symbol: str, 
                            candles: List[dict],
                            direction: str) -> Dict[str, Any]:
        """
        Рассчитывает адаптивный стоп-лосс на основе волатильности
        
        Args:
            symbol: Символ монеты
            candles: Список свечей
            direction: 'LONG' или 'SHORT'
        
        Returns:
            Словарь с параметрами стоп-лосса
        """
        # Рассчитываем волатильность
        volatility = self.calculate_volatility(candles)
        
        # Адаптируем SL: высокая волатильность = дальше SL
        if volatility < 0.3:
            # Низкая волатильность - можем поставить ближе
            sl_percent = self.sl_min + (self.base_sl_percent - self.sl_min) * (volatility / 0.3)
            reason = "Низкая волатильность - ближний SL для защиты"
        elif volatility < 0.7:
            # Средняя волатильность - стандартный SL
            sl_percent = self.base_sl_percent
            reason = "Средняя волатильность - стандартный SL"
        else:
            # Высокая волатильность - дальше SL чтобы не выбило
            sl_percent = self.base_sl_percent + (self.sl_max - self.base_sl_percent) * min((volatility - 0.7) / 0.3, 1.0)
            reason = "Высокая волатильность - дальний SL для избежания шума"
        
        # Округляем до 0.5%
        sl_percent = round(sl_percent * 2) / 2
        
        return {
            'sl_percent': sl_percent,
            'volatility': volatility,
            'reason': reason,
            'confidence': 0.80,  # Высокая уверенность в расчёте
            'base_sl': self.base_sl_percent
        }
    
    def calculate_dynamic_tp(self,
                            symbol: str,
                            candles: List[dict],
                            direction: str) -> Dict[str, Any]:
        """
        Рассчитывает адаптивный тейк-профит на основе силы тренда
        
        Args:
            symbol: Символ монеты
            candles: Список свечей
            direction: 'LONG' или 'SHORT' (направление позиции)
        
        Returns:
            Словарь с параметрами тейк-профита
        """
        # Определяем направление тренда
        trend_dir = 'UP' if direction == 'LONG' else 'DOWN'
        
        # Рассчитываем силу тренда
        trend_strength = self.calculate_trend_strength(candles, trend_dir)
        
        # Адаптируем TP: сильный тренд = дальше TP (больше профит)
        if trend_strength < 0.3:
            # Слабый тренд - скромный TP
            tp_percent = self.tp_min + (self.base_tp_percent - self.tp_min) * (trend_strength / 0.3)
            reason = "Слабый тренд - консервативный TP"
        elif trend_strength < 0.7:
            # Средний тренд - стандартный TP
            tp_percent = self.base_tp_percent
            reason = "Средний тренд - стандартный TP"
        else:
            # Сильный тренд - амбициозный TP
            tp_percent = self.base_tp_percent + (self.tp_max - self.base_tp_percent) * min((trend_strength - 0.7) / 0.3, 1.0)
            reason = "Сильный тренд - расширенный TP для максимизации прибыли"
        
        # Округляем до 10%
        tp_percent = round(tp_percent / 10) * 10
        
        return {
            'tp_percent': tp_percent,
            'trend_strength': trend_strength,
            'reason': reason,
            'confidence': 0.75
        }
    
    def predict_reversal(self, candles: List[dict], direction: str) -> Dict[str, Any]:
        """
        Предсказывает вероятность разворота тренда
        
        Args:
            candles: Список свечей
            direction: 'UP' или 'DOWN' (текущий тренд)
        
        Returns:
            Словарь с вероятностью разворота и рекомендациями
        """
        if len(candles) < self.reversal_lookback + 5:
            return {
                'reversal_probability': 0.5,
                'signals': [],
                'recommendation': 'HOLD',
                'confidence': 0.3
            }
        
        recent = candles[-self.reversal_lookback:]
        reversal_signals = []
        
        # 1. Проверка дивергенции RSI (если RSI доступен)
        # TODO: Добавить расчёт RSI если нужно
        
        # 2. Проверка замедления тренда
        # Сравниваем последние 5 свечей с предыдущими 5
        first_half = recent[:len(recent)//2]
        second_half = recent[len(recent)//2:]
        
        first_move = abs((first_half[-1]['close'] - first_half[0]['close']) / first_half[0]['close'])
        second_move = abs((second_half[-1]['close'] - second_half[0]['close']) / second_half[0]['close'])
        
        if second_move < first_move * 0.5:  # Замедление более чем в 2 раза
            reversal_signals.append('MOMENTUM_LOSS')
        
        # 3. Проверка длинных теней (rejection candles)
        rejection_count = 0
        for candle in recent[-5:]:
            body = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            
            if total_range > 0:
                body_ratio = body / total_range
                
                # Если тело < 40% от диапазона, это rejection
                if body_ratio < 0.4:
                    # Проверяем направление rejection
                    if direction == 'UP' and candle['high'] - max(candle['open'], candle['close']) > body:
                        rejection_count += 1
                    elif direction == 'DOWN' and min(candle['open'], candle['close']) - candle['low'] > body:
                        rejection_count += 1
        
        if rejection_count >= 2:
            reversal_signals.append('REJECTION_CANDLES')
        
        # 4. Проверка объёма на развороте
        volumes = [c['volume'] for c in recent]
        avg_volume = np.mean(volumes[:-1])
        current_volume = volumes[-1]
        
        # Всплеск объёма + противоположное направление свечи
        if current_volume > avg_volume * 1.5:
            last_candle = recent[-1]
            if direction == 'UP' and last_candle['close'] < last_candle['open']:
                reversal_signals.append('VOLUME_REVERSAL')
            elif direction == 'DOWN' and last_candle['close'] > last_candle['open']:
                reversal_signals.append('VOLUME_REVERSAL')
        
        # 5. Проверка экстремальных уровней
        closes = [c['close'] for c in recent]
        current_price = closes[-1]
        price_range = max(closes) - min(closes)
        
        if price_range > 0:
            position_in_range = (current_price - min(closes)) / price_range
            
            if direction == 'UP' and position_in_range > 0.90:
                reversal_signals.append('EXTREME_HIGH')
            elif direction == 'DOWN' and position_in_range < 0.10:
                reversal_signals.append('EXTREME_LOW')
        
        # Рассчитываем вероятность разворота
        reversal_probability = len(reversal_signals) / 5  # 5 возможных сигналов
        
        # Определяем рекомендацию
        if reversal_probability > 0.6:
            recommendation = 'CLOSE'  # Высокий риск разворота - закрыть
        elif reversal_probability > 0.4:
            recommendation = 'TIGHTEN_SL'  # Средний риск - ужесточить SL
        else:
            recommendation = 'HOLD'  # Низкий риск - держать
        
        confidence = 0.65 + (0.2 * (reversal_probability if reversal_probability > 0.5 else (1 - reversal_probability)))
        
        return {
            'reversal_probability': reversal_probability,
            'signals': reversal_signals,
            'recommendation': recommendation,
            'confidence': confidence,
            'direction': direction
        }
    
    def calculate_optimal_trailing(self,
                                  symbol: str,
                                  candles: List[dict],
                                  position_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Рассчитывает оптимальный trailing stop на основе риска разворота
        
        Args:
            symbol: Символ монеты
            candles: Список свечей
            position_data: Данные открытой позиции
        
        Returns:
            Словарь с параметрами trailing stop
        """
        direction = position_data.get('direction', 'LONG')
        trend_dir = 'UP' if direction == 'LONG' else 'DOWN'
        
        # Предсказываем вероятность разворота
        reversal = self.predict_reversal(candles, trend_dir)
        reversal_prob = reversal['reversal_probability']
        
        # Рассчитываем силу тренда
        trend_strength = self.calculate_trend_strength(candles, trend_dir)
        
        # Адаптируем trailing stop
        if reversal_prob > 0.6:
            # Высокий риск разворота - ужесточаем trailing
            trailing_distance = self.trailing_min
            should_tighten = True
            reason = f"Высокий риск разворота ({reversal_prob:.0%}) - ужесточаем trailing"
        elif reversal_prob > 0.4:
            # Средний риск - умеренный trailing
            trailing_distance = self.base_trailing_distance * 0.8
            should_tighten = True
            reason = f"Средний риск разворота ({reversal_prob:.0%}) - умеренный trailing"
        elif trend_strength > 0.7:
            # Сильный тренд - расширенный trailing
            trailing_distance = self.base_trailing_distance + (self.trailing_max - self.base_trailing_distance) * (trend_strength - 0.7) / 0.3
            should_tighten = False
            reason = f"Сильный тренд ({trend_strength:.0%}) - расширенный trailing"
        else:
            # Стандартный trailing
            trailing_distance = self.base_trailing_distance
            should_tighten = False
            reason = "Стандартные условия - базовый trailing"
        
        # Округляем до 10%
        trailing_distance = round(trailing_distance / 10) * 10
        
        return {
            'trailing_distance': trailing_distance,
            'should_tighten': should_tighten,
            'reversal_probability': reversal_prob,
            'trend_strength': trend_strength,
            'reason': reason,
            'confidence': 0.70
        }
    
    def calculate_position_size(self,
                               symbol: str,
                               candles: List[dict],
                               balance_usdt: float,
                               signal_confidence: float = 0.7) -> Dict[str, Any]:
        """
        Рассчитывает оптимальный размер позиции на основе уверенности и волатильности
        
        Args:
            symbol: Символ монеты
            candles: Список свечей
            balance_usdt: Доступный баланс в USDT
            signal_confidence: Уверенность в сигнале (0.0-1.0)
        
        Returns:
            Словарь с рекомендуемым размером позиции
        """
        from bot_engine.config_loader import DEFAULT_AUTO_BOT_CONFIG
        
        base_size_value = DEFAULT_AUTO_BOT_CONFIG.get('default_position_size')
        base_size_mode = DEFAULT_AUTO_BOT_CONFIG.get('default_position_mode')
        if base_size_mode == 'percent' and balance_usdt:
            base_size = balance_usdt * (base_size_value / 100.0)
        else:
            base_size = base_size_value
        
        # Рассчитываем волатильность
        volatility = self.calculate_volatility(candles)
        
        # Факторы для расчёта размера
        # 1. Уверенность в сигнале (0.5-1.0)
        confidence_factor = signal_confidence
        
        # 2. Волатильность (высокая волатильность = меньше размер)
        if volatility < 0.3:
            volatility_factor = 1.2  # Низкая волатильность - можем больше
        elif volatility < 0.7:
            volatility_factor = 1.0  # Средняя - стандартный размер
        else:
            volatility_factor = 0.7  # Высокая - уменьшаем риск
        
        # Комбинируем факторы
        size_multiplier = confidence_factor * volatility_factor
        
        # Рассчитываем итоговый размер
        position_size = base_size * size_multiplier
        
        # Ограничения
        min_size = base_size * 0.5   # Минимум 50% от базового
        max_size = base_size * 2.0   # Максимум 200% от базового
        
        position_size = max(min_size, min(position_size, max_size))
        
        # Формируем причину
        reason = f"Оптимально для уверенности {signal_confidence:.0%} и волатильности {volatility:.2f}"
        
        # Округляем до 0.1 USDT
        position_size = round(position_size, 1)
        
        return {
            'size_usdt': position_size,
            'size_multiplier': size_multiplier,
            'volatility': volatility,
            'confidence': signal_confidence,
            'reason': reason,
            'risk_percent': (position_size / balance_usdt * 100) if balance_usdt > 0 else 0
        }
    
    def get_hold_recommendation(self,
                               symbol: str,
                               candles: List[dict],
                               position_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Рекомендация по удержанию или закрытию позиции
        
        Args:
            symbol: Символ монеты
            candles: Список свечей
            position_data: Данные открытой позиции
        
        Returns:
            Словарь с рекомендацией
        """
        direction = position_data.get('direction', 'LONG')
        entry_price = position_data.get('entry_price', 0)
        current_price = candles[-1]['close']
        
        # Рассчитываем текущий PnL
        if direction == 'LONG':
            pnl_percent = ((current_price - entry_price) / entry_price) * 100
            trend_dir = 'UP'
        else:
            pnl_percent = ((entry_price - current_price) / entry_price) * 100
            trend_dir = 'DOWN'
        
        # Предсказываем разворот
        reversal = self.predict_reversal(candles, trend_dir)
        reversal_prob = reversal['reversal_probability']
        
        # Рассчитываем силу тренда
        trend_strength = self.calculate_trend_strength(candles, trend_dir)
        
        # Определяем действие
        if reversal_prob > 0.7:
            action = 'CLOSE'
            reason = f"Высокий риск разворота ({reversal_prob:.0%}), рекомендуется закрыть"
            risk_score = reversal_prob
        elif reversal_prob > 0.5:
            action = 'TIGHTEN_SL'
            reason = f"Средний риск разворота ({reversal_prob:.0%}), ужесточить SL"
            risk_score = reversal_prob
        elif trend_strength > 0.7 and pnl_percent > 0:
            action = 'HOLD'
            reason = f"Сильный тренд ({trend_strength:.0%}), продолжаем удержание"
            risk_score = 1 - trend_strength
        elif pnl_percent < -10:
            action = 'REVIEW'
            reason = f"Убыток {pnl_percent:.1f}%, требуется анализ"
            risk_score = 0.8
        else:
            action = 'HOLD'
            reason = "Стандартные условия, продолжаем удержание"
            risk_score = 0.4
        
        # Предсказываем ожидаемую прибыль
        if trend_strength > 0.6:
            expected_profit = pnl_percent + (trend_strength * 100)  # Упрощённый прогноз
        else:
            expected_profit = pnl_percent * 1.2
        
        return {
            'action': action,
            'reason': reason,
            'risk_score': risk_score,
            'reversal_probability': reversal_prob,
            'trend_strength': trend_strength,
            'current_pnl': pnl_percent,
            'expected_profit': expected_profit,
            'confidence': 0.65,
            'signals': reversal['signals']
        }
    
    def analyze_position(self,
                        symbol: str,
                        candles: List[dict],
                        position_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Комплексный анализ открытой позиции
        
        Args:
            symbol: Символ монеты
            candles: Список свечей
            position_data: Данные позиции
        
        Returns:
            Полный анализ с рекомендациями
        """
        direction = position_data.get('direction', 'LONG')
        
        # Получаем все компоненты анализа
        trailing = self.calculate_optimal_trailing(symbol, candles, position_data)
        hold_rec = self.get_hold_recommendation(symbol, candles, position_data)
        
        return {
            'symbol': symbol,
            'trailing_stop': trailing,
            'hold_recommendation': hold_rec,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Возвращает статус менеджера рисков
        
        Returns:
            Словарь со статусом
        """
        return {
            'active': True,
            'base_sl': self.base_sl_percent,
            'base_tp': self.base_tp_percent,
            'sl_range': [self.sl_min, self.sl_max],
            'tp_range': [self.tp_min, self.tp_max],
            'trailing_range': [self.trailing_min, self.trailing_max]
        }

