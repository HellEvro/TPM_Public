"""
Технические индикаторы для торговых ботов
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict
from .config_loader import (
    RSI_PERIOD, EMA_FAST, EMA_SLOW, TREND_CONFIRMATION_BARS,
    RSI_EXTREME_OVERSOLD, RSI_EXTREME_OVERBOUGHT, RSI_VOLATILITY_THRESHOLD_HIGH,
    RSI_VOLATILITY_THRESHOLD_LOW, RSI_DIVERGENCE_LOOKBACK, RSI_VOLUME_CONFIRMATION_MULTIPLIER,
    RSI_STOCH_PERIOD, RSI_EXTREME_ZONE_TIMEOUT
)


class TechnicalIndicators:
    """Класс для расчета технических индикаторов"""
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = RSI_PERIOD) -> Optional[float]:
        """
        Расчет RSI
        
        Args:
            prices: Список цен закрытия (должны быть упорядочены от старых к новым)
            period: Период RSI
            
        Returns:
            Значение RSI или None если недостаточно данных
        """
        if len(prices) < period + 1:
            return None
            
        # Дополнительная проверка: убеждаемся, что данные не пустые
        if not prices or all(p == 0 for p in prices):
            return None
            
        prices_array = np.array(prices)
        deltas = np.diff(prices_array)  # Разности между соседними ценами
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Первое значение - простое среднее
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        # Остальные значения - экспоненциальное сглаживание
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> Optional[float]:
        """
        Расчет EMA
        
        Args:
            prices: Список цен закрытия
            period: Период EMA
            
        Returns:
            Значение EMA или None если недостаточно данных
        """
        if len(prices) < period:
            return None
            
        prices_array = np.array(prices)
        alpha = 2 / (period + 1)
        
        # Первое значение EMA = SMA
        ema = np.mean(prices_array[:period])
        
        # Последующие значения
        for price in prices_array[period:]:
            ema = alpha * price + (1 - alpha) * ema
            
        return float(ema)
    
    @staticmethod
    def calculate_atr(candles_data: List[dict], period: int = 14) -> Optional[float]:
        """
        Расчет Average True Range (ATR) для измерения волатильности
        
        Args:
            candles_data: Список свечей с полями high, low, close
            period: Период ATR
            
        Returns:
            Значение ATR или None если недостаточно данных
        """
        if len(candles_data) < period + 1:
            return None
            
        true_ranges = []
        for i in range(1, len(candles_data)):
            high = float(candles_data[i]['high'])
            low = float(candles_data[i]['low'])
            prev_close = float(candles_data[i-1]['close'])
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_ranges.append(max(tr1, tr2, tr3))
        
        if len(true_ranges) < period:
            return None
            
        # Рассчитываем ATR как EMA от True Range
        atr = np.mean(true_ranges[:period])  # Первое значение - простое среднее
        alpha = 2.0 / (period + 1)
        
        for tr in true_ranges[period:]:
            atr = alpha * tr + (1 - alpha) * atr
            
        return float(atr)
    
    @staticmethod
    def calculate_adaptive_rsi_levels(candles_data: List[dict], 
                                    base_oversold: float = 29, 
                                    base_overbought: float = 71) -> Tuple[float, float]:
        """
        Рассчитывает адаптивные уровни RSI на основе волатильности
        
        Args:
            candles_data: Список свечей
            base_oversold: Базовый уровень перепроданности
            base_overbought: Базовый уровень перекупленности
            
        Returns:
            Tuple[oversold_level, overbought_level]
        """
        if len(candles_data) < 30:  # Минимум данных для анализа волатильности
            return base_oversold, base_overbought
            
        # Рассчитываем текущую волатильность
        current_atr = TechnicalIndicators.calculate_atr(candles_data[-14:])
        if current_atr is None:
            return base_oversold, base_overbought
            
        # Рассчитываем среднюю волатильность за более длинный период
        historical_atrs = []
        for i in range(14, len(candles_data), 14):
            if i + 14 <= len(candles_data):
                atr = TechnicalIndicators.calculate_atr(candles_data[i:i+14])
                if atr is not None:
                    historical_atrs.append(atr)
        
        if not historical_atrs:
            return base_oversold, base_overbought
            
        avg_atr = np.mean(historical_atrs)
        volatility_factor = current_atr / avg_atr if avg_atr > 0 else 1.0
        
        # Адаптируем уровни на основе волатильности
        if volatility_factor > RSI_VOLATILITY_THRESHOLD_HIGH:  # Высокая волатильность
            oversold = max(base_oversold - 5, RSI_EXTREME_OVERSOLD)
            overbought = min(base_overbought + 5, RSI_EXTREME_OVERBOUGHT)
        elif volatility_factor < RSI_VOLATILITY_THRESHOLD_LOW:  # Низкая волатильность
            oversold = min(base_oversold + 3, base_overbought - 10)
            overbought = max(base_overbought - 3, base_oversold + 10)
        else:
            oversold = base_oversold
            overbought = base_overbought
            
        return float(oversold), float(overbought)
    
    @staticmethod
    def detect_rsi_divergence(prices: List[float], 
                            rsi_values: List[float], 
                            lookback: int = RSI_DIVERGENCE_LOOKBACK) -> Optional[str]:
        """
        Обнаруживает дивергенцию между ценой и RSI
        
        Args:
            prices: Список цен закрытия
            rsi_values: Список значений RSI
            lookback: Период для поиска дивергенций
            
        Returns:
            'BULLISH_DIVERGENCE', 'BEARISH_DIVERGENCE' или None
        """
        if len(prices) < lookback or len(rsi_values) < lookback:
            return None
            
        recent_prices = prices[-lookback:]
        recent_rsi = rsi_values[-lookback:]
        
        # Находим локальные максимумы и минимумы
        price_trend = recent_prices[-1] - recent_prices[0]
        rsi_trend = recent_rsi[-1] - recent_rsi[0]
        
        # Bullish divergence: цена делает более низкий минимум, RSI - более высокий минимум
        if price_trend < -0.01 and rsi_trend > 1:  # Цена падает, RSI растет
            return "BULLISH_DIVERGENCE"
        
        # Bearish divergence: цена делает более высокий максимум, RSI - более низкий максимум
        if price_trend > 0.01 and rsi_trend < -1:  # Цена растет, RSI падает
            return "BEARISH_DIVERGENCE"
            
        return None
    
    @staticmethod
    def confirm_with_volume(volumes: List[float], lookback: int = 5) -> bool:
        """
        Подтверждает сигнал анализом объема
        
        Args:
            volumes: Список объемов
            lookback: Период для сравнения
            
        Returns:
            True если объем подтверждает сигнал
        """
        if len(volumes) < lookback * 2:
            return False
            
        recent_volume = np.mean(volumes[-lookback:])
        avg_volume = np.mean(volumes[-lookback*2:-lookback])
        
        # Объем должен быть выше среднего для подтверждения
        return bool(recent_volume > avg_volume * RSI_VOLUME_CONFIRMATION_MULTIPLIER)
    
    @staticmethod
    def calculate_stoch_rsi(rsi_values: List[float], 
                          stoch_period: int = 14,
                          k_smooth: int = 3, 
                          d_smooth: int = 3) -> Optional[Dict[str, float]]:
        """
        Рассчитывает Stochastic RSI (%K и %D) по стандартной формуле TradingView/Bybit
        
        Формула:
        1. Stoch RSI = (RSI - min(RSI, stoch_period)) / (max(RSI, stoch_period) - min(RSI, stoch_period))
        2. %K = SMA(Stoch RSI, k_smooth)
        3. %D = SMA(%K, d_smooth)
        
        Args:
            rsi_values: Список значений RSI (от старого к новому)
            stoch_period: Период для нахождения min/max RSI (обычно 14)
            k_smooth: Период сглаживания для %K (обычно 3)
            d_smooth: Период сглаживания для %D (обычно 3)
            
        Returns:
            Словарь с %K и %D или None
        """
        min_required = stoch_period + k_smooth + d_smooth
        if len(rsi_values) < min_required:
            return None
        
        # Шаг 1: Рассчитываем Stochastic RSI для каждой точки
        stoch_rsi_values = []
        for i in range(stoch_period - 1, len(rsi_values)):
            # Используем СКОЛЬЗЯЩЕЕ ОКНО из последних stoch_period значений RSI
            # period_rsi содержит RSI от позиции (i - stoch_period + 1) до позиции i
            period_rsi = rsi_values[i - stoch_period + 1:i + 1]
            highest_rsi = max(period_rsi)
            lowest_rsi = min(period_rsi)
            
            if highest_rsi == lowest_rsi:
                stoch_rsi_values.append(0.5)  # Нейтральное значение
            else:
                # Формула: (текущий RSI - min за период) / (max за период - min за период)
                stoch_rsi = (rsi_values[i] - lowest_rsi) / (highest_rsi - lowest_rsi)
                stoch_rsi_values.append(stoch_rsi)
        
        if len(stoch_rsi_values) < k_smooth + d_smooth:
            return None
        
        # Шаг 2: Рассчитываем %K как SMA от Stochastic RSI
        k_values = []
        for i in range(k_smooth - 1, len(stoch_rsi_values)):
            k_sma = sum(stoch_rsi_values[i - k_smooth + 1:i + 1]) / k_smooth
            k_values.append(k_sma * 100)  # Переводим в проценты
        
        if len(k_values) < d_smooth:
            return None
        
        # Шаг 3: Рассчитываем %D как SMA от %K
        d_values = []
        for i in range(d_smooth - 1, len(k_values)):
            d_sma = sum(k_values[i - d_smooth + 1:i + 1]) / d_smooth
            d_values.append(d_sma)
        
        return {
            'k': float(k_values[-1]),
            'd': float(d_values[-1])
        }
    
    @staticmethod
    def calculate_rsi_history(candles_data: List[dict], 
                            period: int = RSI_PERIOD, 
                            history_length: int = 50) -> List[float]:
        """
        Рассчитывает историю значений RSI
        
        Args:
            candles_data: Список свечей
            period: Период RSI
            history_length: Количество исторических значений (не используется, оставлено для совместимости)
            
        Returns:
            Список исторических значений RSI для всех свечей
        """
        if len(candles_data) < period:
            return []
            
        closes = np.array([float(candle['close']) for candle in candles_data])
        
        # ⚡ ОПТИМИЗАЦИЯ: Рассчитываем RSI ОДИН РАЗ для всего массива, а не N раз в цикле!
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        rsi_history = []
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        # Вычисляем RSI для каждой точки ОДНИМ проходом (было: N вызовов calculate_rsi)
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))
            
            rsi_history.append(float(rsi))
        
        # Возвращаем только последние history_length значений
        if len(rsi_history) > history_length:
            return rsi_history[-history_length:]
        return rsi_history
    
    @staticmethod
    def calculate_ema_slope(prices: List[float], period: int, lookback: int = 3) -> Optional[float]:
        """
        Расчет наклона EMA
        
        Args:
            prices: Список цен закрытия
            period: Период EMA
            lookback: Количество баров для расчета наклона
            
        Returns:
            Наклон EMA или None если недостаточно данных
        """
        if len(prices) < period + lookback:
            return None
            
        # Рассчитываем EMA для последних lookback баров
        ema_values = []
        for i in range(lookback):
            subset = prices[:-(lookback-i-1)] if lookback-i-1 > 0 else prices
            ema = TechnicalIndicators.calculate_ema(subset, period)
            if ema is not None:
                ema_values.append(ema)
        
        if len(ema_values) < 2:
            return None
            
        # Простой расчет наклона (разность между последним и первым значением)
        slope = ema_values[-1] - ema_values[0]
        return float(slope)


class TrendAnalyzer:
    """Анализатор тренда на основе EMA и подтверждений"""
    
    def __init__(self):
        self.fast_period = EMA_FAST
        self.slow_period = EMA_SLOW
        self.confirmation_bars = TREND_CONFIRMATION_BARS
    
    def analyze_trend(self, candles_data: List[dict]) -> Tuple[str, dict]:
        """
        Анализирует тренд на основе EMA и подтверждений
        
        Args:
            candles_data: Список свечей с полями high, low, open, close, volume
            
        Returns:
            Tuple[trend_direction, analysis_details]
        """
        if len(candles_data) < max(self.slow_period, self.confirmation_bars) + 10:
            return 'NEUTRAL', {'reason': 'insufficient_data'}
        
        closes = [float(candle['close']) for candle in candles_data]
        
        # Рассчитываем EMA
        ema50 = TechnicalIndicators.calculate_ema(closes, self.fast_period)
        ema200 = TechnicalIndicators.calculate_ema(closes, self.slow_period)
        ema200_slope = TechnicalIndicators.calculate_ema_slope(closes, self.slow_period)
        
        if not all([ema50, ema200, ema200_slope is not None]):
            return 'NEUTRAL', {'reason': 'indicator_calculation_failed'}
        
        current_close = closes[-1]
        
        # Проверяем последние N закрытий относительно EMA200
        recent_closes = closes[-self.confirmation_bars:]
        closes_above_ema200 = all(close > ema200 for close in recent_closes)
        closes_below_ema200 = all(close < ema200 for close in recent_closes)
        
        analysis = {
            'ema50': ema50,
            'ema200': ema200,
            'ema200_slope': ema200_slope,
            'current_close': current_close,
            'closes_above_ema200': closes_above_ema200,
            'closes_below_ema200': closes_below_ema200,
            'ema50_above_ema200': ema50 > ema200,
            'close_above_ema200': current_close > ema200,
            'ema200_slope_positive': ema200_slope > 0
        }
        
        # Определяем тренд UP
        if (current_close > ema200 and 
            ema50 > ema200 and 
            ema200_slope > 0 and 
            closes_above_ema200):
            return 'UP', analysis
        
        # Определяем тренд DOWN
        elif (current_close < ema200 and 
              ema50 < ema200 and 
              ema200_slope < 0 and 
              closes_below_ema200):
            return 'DOWN', analysis
        
        # Иначе NEUTRAL
        else:
            return 'NEUTRAL', analysis


class SignalGenerator:
    """Генератор торговых сигналов на основе RSI и тренда"""
    
    def __init__(self):
        self.trend_analyzer = TrendAnalyzer()
    
    def generate_signals(self, candles_data: List[dict]) -> dict:
        """
        Генерирует торговые сигналы с улучшенной логикой для сильных трендов
        
        Args:
            candles_data: Список свечей (должны быть упорядочены от старых к новым)
            
        Returns:
            Словарь с сигналами и анализом
        """
        if len(candles_data) < 50:  # Минимум данных для анализа
            return {
                'trend': 'NEUTRAL',
                'rsi': None,
                'signal': 'WAIT',
                'reason': 'insufficient_data'
            }
        
        # Извлекаем данные (candles_data уже перевернуты в get_coin_rsi_data)
        closes = [float(candle['close']) for candle in candles_data]
        volumes = [float(candle['volume']) for candle in candles_data]
        
        # Анализируем тренд
        trend, trend_analysis = self.trend_analyzer.analyze_trend(candles_data)
        
        # Рассчитываем основные индикаторы
        rsi = TechnicalIndicators.calculate_rsi(closes)
        if rsi is None:
            return {
                'trend': trend,
                'rsi': None,
                'signal': 'WAIT',
                'reason': 'rsi_calculation_failed',
                'trend_analysis': trend_analysis
            }
        
        # Рассчитываем дополнительные индикаторы
        rsi_history = TechnicalIndicators.calculate_rsi_history(candles_data, history_length=20)
        adaptive_levels = TechnicalIndicators.calculate_adaptive_rsi_levels(candles_data)
        divergence = TechnicalIndicators.detect_rsi_divergence(closes, rsi_history + [rsi]) if rsi_history else None
        volume_confirmation = TechnicalIndicators.confirm_with_volume(volumes)
        stoch_rsi_result = TechnicalIndicators.calculate_stoch_rsi(rsi_history + [rsi]) if rsi_history else None
        stoch_rsi = stoch_rsi_result['k'] if stoch_rsi_result else None
        
        # Создаем расширенный контекст для анализа
        enhanced_context = {
            'rsi': rsi,
            'adaptive_oversold': adaptive_levels[0],
            'adaptive_overbought': adaptive_levels[1],
            'divergence': divergence,
            'volume_confirmation': volume_confirmation,
            'stoch_rsi': stoch_rsi,
            'rsi_history': rsi_history,
            'trend': trend
        }
        
        # Генерируем сигналы с улучшенной логикой
        signal = self._determine_enhanced_signal(enhanced_context)
        
        return {
            'trend': trend,
            'rsi': rsi,
            'signal': signal['action'],
            'reason': signal['reason'],
            'trend_analysis': trend_analysis,
            'price': closes[-1],
            'enhanced_analysis': {
                'adaptive_levels': adaptive_levels,
                'divergence': divergence,
                'volume_confirmation': volume_confirmation,
                'stoch_rsi': stoch_rsi,
                'extreme_zone_duration': self._check_extreme_zone_duration(rsi_history + [rsi])
            }
        }
    
    def _determine_signal(self, trend: str, rsi: float) -> dict:
        """
        Определяет сигнал на основе RSI с учетом тренда
        
        Args:
            trend: Направление тренда (используется для фильтрации входов)
            rsi: Значение RSI
            
        Returns:
            Словарь с действием и причиной
        """
        from .config_loader import RSI_OVERSOLD, RSI_OVERBOUGHT
        
        # Логика с учетом тренда для входов
        if rsi <= RSI_OVERSOLD:  # RSI ≤ 29 
            # Для LONG проверяем тренд DOWN (избегаем нисходящего тренда)
            if trend == 'DOWN':
                return {'action': 'WAIT', 'reason': f'rsi_oversold_but_down_trend_{rsi:.1f}'}
            else:
                return {'action': 'ENTER_LONG', 'reason': f'rsi_oversold_{rsi:.1f}'}
        elif rsi >= RSI_OVERBOUGHT:  # RSI ≥ 71
            # Для SHORT проверяем тренд UP (избегаем восходящего тренда)
            if trend == 'UP':
                return {'action': 'WAIT', 'reason': f'rsi_overbought_but_up_trend_{rsi:.1f}'}
            else:
                return {'action': 'ENTER_SHORT', 'reason': f'rsi_overbought_{rsi:.1f}'}
        
        # ❌ ИСПРАВЛЕНИЕ: EXIT сигналы НЕ должны показываться для монет без ботов!
        # EXIT сигналы должны определяться только в контексте активных ботов
        # Здесь мы определяем только сигналы входа (ENTER_LONG, ENTER_SHORT, WAIT)
        
        # RSI между 30-70 - ждем
        else:
            return {'action': 'WAIT', 'reason': f'rsi_neutral_{rsi:.1f}'}
    
    def _determine_enhanced_signal(self, context: Dict) -> dict:
        """
        Определяет сигнал с учетом дополнительных фильтров для сильных трендов
        
        Args:
            context: Словарь с расширенными данными для анализа
            
        Returns:
            Словарь с действием и причиной
        """
        from .config_loader import RSI_OVERSOLD, RSI_OVERBOUGHT, SystemConfig
        
        # Проверяем, включена ли улучшенная система RSI
        if not SystemConfig.ENHANCED_RSI_ENABLED:
            # Возвращаемся к стандартной логике
            return self._determine_signal(context['trend'], context['rsi'])
        
        rsi = context['rsi']
        adaptive_oversold = context['adaptive_oversold']
        adaptive_overbought = context['adaptive_overbought']
        divergence = context['divergence']
        volume_confirmation = context['volume_confirmation']
        stoch_rsi = context['stoch_rsi']
        rsi_history = context['rsi_history']
        trend = context['trend']
        
        # Проверяем продолжительность нахождения в экстремальной зоне
        extreme_duration = self._check_extreme_zone_duration(rsi_history + [rsi])
        
        # Строгий режим с дивергенциями
        if SystemConfig.ENHANCED_RSI_REQUIRE_DIVERGENCE_CONFIRMATION:
            # В строгом режиме требуем обязательную дивергенцию
            if rsi <= adaptive_oversold and divergence == "BULLISH_DIVERGENCE":
                return {'action': 'ENTER_LONG', 'reason': f'strict_mode_bullish_divergence_{rsi:.1f}'}
            elif rsi >= adaptive_overbought and divergence == "BEARISH_DIVERGENCE":
                return {'action': 'ENTER_SHORT', 'reason': f'strict_mode_bearish_divergence_{rsi:.1f}'}
            else:
                return {'action': 'WAIT', 'reason': f'strict_mode_no_divergence_{rsi:.1f}'}

        # Логика входа в LONG позицию
        if rsi <= adaptive_oversold:
            confirmation_factors = []
            
            # Основное условие - RSI в зоне перепроданности
            if rsi <= RSI_OVERSOLD:
                confirmation_factors.append("base_oversold")
            
            # Дополнительные подтверждения (с учетом настроек)
            if divergence == "BULLISH_DIVERGENCE":
                confirmation_factors.append("bullish_divergence")
                
            if volume_confirmation and SystemConfig.ENHANCED_RSI_REQUIRE_VOLUME_CONFIRMATION:
                confirmation_factors.append("volume_confirm")
                
            if stoch_rsi and stoch_rsi < 20 and SystemConfig.ENHANCED_RSI_USE_STOCH_RSI:
                confirmation_factors.append("stoch_oversold")
                
            # Если RSI слишком долго в экстремальной зоне, требуем дополнительные подтверждения
            if extreme_duration > RSI_EXTREME_ZONE_TIMEOUT:
                if len(confirmation_factors) >= 2:  # Нужно минимум 2 подтверждения
                    return {
                        'action': 'ENTER_LONG', 
                        'reason': f'enhanced_oversold_{rsi:.1f}_confirmed_{"_".join(confirmation_factors)}'
                    }
                else:
                    return {
                        'action': 'WAIT', 
                        'reason': f'oversold_but_insufficient_confirmation_{rsi:.1f}_duration_{extreme_duration}'
                    }
            else:
                # RSI недавно вошел в зону перепроданности
                if len(confirmation_factors) >= 1:
                    return {
                        'action': 'ENTER_LONG', 
                        'reason': f'fresh_oversold_{rsi:.1f}_{"_".join(confirmation_factors)}'
                    }
        
        # Логика входа в SHORT позицию
        elif rsi >= adaptive_overbought:
            confirmation_factors = []
            
            # Основное условие - RSI в зоне перекупленности
            if rsi >= RSI_OVERBOUGHT:
                confirmation_factors.append("base_overbought")
            
            # Дополнительные подтверждения (с учетом настроек)
            if divergence == "BEARISH_DIVERGENCE":
                confirmation_factors.append("bearish_divergence")
                
            if volume_confirmation and SystemConfig.ENHANCED_RSI_REQUIRE_VOLUME_CONFIRMATION:
                confirmation_factors.append("volume_confirm")
                
            if stoch_rsi and stoch_rsi > 80 and SystemConfig.ENHANCED_RSI_USE_STOCH_RSI:
                confirmation_factors.append("stoch_overbought")
                
            # Если RSI слишком долго в экстремальной зоне, требуем дополнительные подтверждения
            if extreme_duration > RSI_EXTREME_ZONE_TIMEOUT:
                if len(confirmation_factors) >= 2:  # Нужно минимум 2 подтверждения
                    return {
                        'action': 'ENTER_SHORT', 
                        'reason': f'enhanced_overbought_{rsi:.1f}_confirmed_{"_".join(confirmation_factors)}'
                    }
                else:
                    return {
                        'action': 'WAIT', 
                        'reason': f'overbought_but_insufficient_confirmation_{rsi:.1f}_duration_{extreme_duration}'
                    }
            else:
                # RSI недавно вошел в зону перекупленности
                if len(confirmation_factors) >= 1:
                    return {
                        'action': 'ENTER_SHORT', 
                        'reason': f'fresh_overbought_{rsi:.1f}_{"_".join(confirmation_factors)}'
                    }
        
        # Логика выхода (остается как было, но с адаптивными уровнями)
        # ❌ ИСПРАВЛЕНИЕ: EXIT сигналы НЕ должны показываться для монет без ботов!
        # EXIT сигналы должны определяться только в контексте активных ботов
        # Здесь мы определяем только сигналы входа (ENTER_LONG, ENTER_SHORT, WAIT)
        
        # По умолчанию ждем
        return {'action': 'WAIT', 'reason': f'enhanced_neutral_{rsi:.1f}'}
    
    def _check_extreme_zone_duration(self, rsi_values: List[float]) -> int:
        """
        Проверяет, сколько периодов RSI находится в экстремальной зоне
        
        Args:
            rsi_values: Список значений RSI (от старых к новым)
            
        Returns:
            Количество последовательных периодов в экстремальной зоне
        """
        if not rsi_values:
            return 0
            
        current_rsi = rsi_values[-1]
        duration = 0
        
        # Определяем, в какой экстремальной зоне находимся
        if current_rsi <= RSI_EXTREME_OVERSOLD:
            # Считаем назад, пока RSI <= 20
            for i in range(len(rsi_values) - 1, -1, -1):
                if rsi_values[i] <= RSI_EXTREME_OVERSOLD:
                    duration += 1
                else:
                    break
        elif current_rsi >= RSI_EXTREME_OVERBOUGHT:
            # Считаем назад, пока RSI >= 80
            for i in range(len(rsi_values) - 1, -1, -1):
                if rsi_values[i] >= RSI_EXTREME_OVERBOUGHT:
                    duration += 1
                else:
                    break
                    
        return duration
