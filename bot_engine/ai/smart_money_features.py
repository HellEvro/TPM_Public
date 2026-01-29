"""
Smart Money Concepts (SMC) - Институциональный анализ рынка

Модуль реализует концепции Smart Money для определения:
- Order Blocks (зоны накопления крупных игроков)
- Fair Value Gaps (FVG) - зоны дисбаланса
- Liquidity Zones (зоны ликвидности)
- Break of Structure (BOS) - пробой структуры
- Change of Character (CHoCH) - смена характера тренда
- Market Structure (HH, HL, LH, LL)

Основа: RSI + SMC = минимум шума, максимум качества сигналов
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger('SMC')

class SignalType(Enum):
    """Типы торговых сигналов"""
    LONG = "LONG"
    SHORT = "SHORT"
    WAIT = "WAIT"

class TrendType(Enum):
    """Типы тренда"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    RANGING = "ranging"
    UNDEFINED = "undefined"

class ZoneType(Enum):
    """Типы ценовых зон"""
    PREMIUM = "premium"
    DISCOUNT = "discount"
    EQUILIBRIUM = "equilibrium"

@dataclass
class OrderBlock:
    """Структура Order Block"""
    type: str  # 'bullish' или 'bearish'
    high: float
    low: float
    index: int
    strength: float
    tested: bool
    timestamp: Optional[str] = None

@dataclass
class FairValueGap:
    """Структура Fair Value Gap"""
    type: str  # 'bullish' или 'bearish'
    top: float
    bottom: float
    index: int
    size_pct: float
    mitigated: bool
    timestamp: Optional[str] = None

@dataclass
class LiquidityZone:
    """Структура Liquidity Zone"""
    type: str  # 'buy_side' или 'sell_side'
    price: float
    strength: int
    indices: List[int]

@dataclass
class SwingPoint:
    """Структура Swing Point"""
    price: float
    index: int
    type: str  # 'high' или 'low'

class SmartMoneyFeatures:
    """
    Smart Money Concepts (SMC) для институционального анализа рынка

    Основа: RSI 6H + Order Blocks + FVG + Liquidity + Market Structure
    Минимум шума, максимум качества сигналов
    """

    def __init__(
        self,
        rsi_period: int = 14,
        swing_lookback: int = 5,
        impulse_threshold: float = 0.02,
        fvg_min_size: float = 0.001,
        equal_level_tolerance: float = 0.002
    ):
        """
        Инициализация Smart Money Features

        Args:
            rsi_period: Период RSI (по умолчанию 14)
            swing_lookback: Количество свечей для определения swing point
            impulse_threshold: Порог для определения импульсного движения (2%)
            fvg_min_size: Минимальный размер FVG (0.1%)
            equal_level_tolerance: Допуск для определения equal levels (0.2%)
        """
        self.rsi_period = rsi_period
        self.swing_lookback = swing_lookback
        self.impulse_threshold = impulse_threshold
        self.fvg_min_size = fvg_min_size
        self.equal_level_tolerance = equal_level_tolerance

        logger.info(f"SmartMoneyFeatures инициализирован: RSI={rsi_period}, swing={swing_lookback}")

    # ==================== RSI (ОСНОВА) ====================

    def compute_rsi(self, df: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
        """
        Вычисляет RSI - основной индикатор для входов

        Args:
            df: DataFrame со свечами (должен содержать 'close')
            period: Период RSI (если не указан, используется self.rsi_period)

        Returns:
            Series с значениями RSI
        """
        period = period or self.rsi_period

        if 'close' not in df.columns:
            logger.error("DataFrame должен содержать колонку 'close'")
            return pd.Series(dtype=float)

        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # Избегаем деления на ноль
        rs = np.where(loss != 0, gain / loss, 100)
        rsi = 100 - (100 / (1 + rs))

        return pd.Series(rsi, index=df.index)

    def get_rsi_signal(self, df: pd.DataFrame) -> Dict:
        """
        Получает сигнал на основе RSI

        Args:
            df: DataFrame со свечами

        Returns:
            Dict с сигналом RSI:
            - value: текущее значение RSI
            - zone: 'oversold', 'overbought', 'neutral'
            - signal: 'LONG', 'SHORT', 'WAIT'
            - divergence: тип дивергенции
        """
        rsi = self.compute_rsi(df)

        if rsi.empty or len(rsi) < 2:
            return {
                'value': None,
                'zone': 'neutral',
                'signal': 'WAIT',
                'divergence': None
            }

        current_rsi = rsi.iloc[-1]

        # Определяем зону RSI
        if current_rsi <= 30:
            zone = 'oversold'
            signal = 'LONG'
        elif current_rsi >= 70:
            zone = 'overbought'
            signal = 'SHORT'
        else:
            zone = 'neutral'
            signal = 'WAIT'

        # Проверяем дивергенцию
        divergence = self._detect_rsi_divergence(df, rsi)

        return {
            'value': float(current_rsi),
            'zone': zone,
            'signal': signal,
            'divergence': divergence
        }

    def _detect_rsi_divergence(
        self,
        df: pd.DataFrame,
        rsi: pd.Series,
        lookback: int = 10
    ) -> Optional[str]:
        """
        Детектирует дивергенции RSI

        Bullish divergence: цена делает LL, RSI делает HL
        Bearish divergence: цена делает HH, RSI делает LH

        Args:
            df: DataFrame со свечами
            rsi: Series с RSI
            lookback: Количество свечей для поиска дивергенции

        Returns:
            'bullish', 'bearish' или None
        """
        if len(df) < lookback or len(rsi) < lookback:
            return None

        try:
            # Получаем данные за lookback период
            price_now = df['low'].iloc[-1]
            price_prev = df['low'].iloc[-lookback]
            rsi_now = rsi.iloc[-1]
            rsi_prev = rsi.iloc[-lookback]

            # Bullish divergence: цена ниже, RSI выше
            if price_now < price_prev and rsi_now > rsi_prev:
                return 'bullish'

            # Bearish divergence: цена выше, RSI ниже
            price_high_now = df['high'].iloc[-1]
            price_high_prev = df['high'].iloc[-lookback]

            if price_high_now > price_high_prev and rsi_now < rsi_prev:
                return 'bearish'

        except Exception as e:
                        pass

        return None

    # ==================== ORDER BLOCKS ====================

    def find_order_blocks(
        self,
        df: pd.DataFrame,
        lookback: int = 50
    ) -> List[OrderBlock]:
        """
        Находит Order Blocks - зоны накопления крупных игроков

        Bullish OB: последняя медвежья свеча перед импульсным ростом
        Bearish OB: последняя бычья свеча перед импульсным падением

        Args:
            df: DataFrame со свечами (open, high, low, close обязательны)
            lookback: Сколько свечей назад искать

        Returns:
            Список OrderBlock объектов
        """
        if len(df) < 5:
            return []

        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"DataFrame должен содержать колонки: {required_cols}")
            return []

        order_blocks = []

        # Определяем импульсные движения
        for i in range(3, min(lookback, len(df) - 3)):
            idx = len(df) - i - 1

            if idx < 0 or idx + 3 >= len(df):
                continue

            # Вычисляем движение за 3 свечи
            move = (df['close'].iloc[idx + 3] - df['close'].iloc[idx]) / df['close'].iloc[idx]

            # Bullish Order Block: импульс вверх
            if move > self.impulse_threshold:
                # Ищем последнюю медвежью свечу перед импульсом
                for j in range(idx, max(idx - 5, 0), -1):
                    if df['close'].iloc[j] < df['open'].iloc[j]:  # Медвежья свеча
                        ob = OrderBlock(
                            type='bullish',
                            high=float(df['high'].iloc[j]),
                            low=float(df['low'].iloc[j]),
                            index=j,
                            strength=float(move),
                            tested=self._is_ob_tested(df, j, 'bullish'),
                            timestamp=str(df.index[j]) if isinstance(df.index[j], (pd.Timestamp, str)) else None
                        )
                        order_blocks.append(ob)
                        break

            # Bearish Order Block: импульс вниз
            elif move < -self.impulse_threshold:
                # Ищем последнюю бычью свечу перед падением
                for j in range(idx, max(idx - 5, 0), -1):
                    if df['close'].iloc[j] > df['open'].iloc[j]:  # Бычья свеча
                        ob = OrderBlock(
                            type='bearish',
                            high=float(df['high'].iloc[j]),
                            low=float(df['low'].iloc[j]),
                            index=j,
                            strength=float(abs(move)),
                            tested=self._is_ob_tested(df, j, 'bearish'),
                            timestamp=str(df.index[j]) if isinstance(df.index[j], (pd.Timestamp, str)) else None
                        )
                        order_blocks.append(ob)
                        break

        # Сортируем по индексу (от новых к старым)
        order_blocks.sort(key=lambda x: x.index, reverse=True)

        return order_blocks

    def _is_ob_tested(self, df: pd.DataFrame, ob_index: int, ob_type: str) -> bool:
        """
        Проверяет, был ли Order Block протестирован

        Args:
            df: DataFrame со свечами
            ob_index: Индекс Order Block
            ob_type: Тип OB ('bullish' или 'bearish')

        Returns:
            True если OB был протестирован
        """
        if ob_index >= len(df) - 1:
            return False

        ob_high = df['high'].iloc[ob_index]
        ob_low = df['low'].iloc[ob_index]

        for i in range(ob_index + 1, len(df)):
            if ob_type == 'bullish':
                # Bullish OB тестируется когда цена возвращается в зону
                if df['low'].iloc[i] <= ob_high and df['low'].iloc[i] >= ob_low:
                    return True
            else:  # bearish
                if df['high'].iloc[i] >= ob_low and df['high'].iloc[i] <= ob_high:
                    return True

        return False

    def get_active_order_blocks(
        self,
        df: pd.DataFrame,
        current_price: float,
        max_distance_pct: float = 5.0
    ) -> Dict:
        """
        Получает активные (непротестированные) Order Blocks рядом с текущей ценой

        Args:
            df: DataFrame со свечами
            current_price: Текущая цена
            max_distance_pct: Максимальное расстояние до OB в %

        Returns:
            Dict с nearest_bullish и nearest_bearish OB
        """
        order_blocks = self.find_order_blocks(df)

        # Фильтруем только непротестированные
        active_obs = [ob for ob in order_blocks if not ob.tested]

        nearest_bullish = None
        nearest_bearish = None
        min_bullish_dist = float('inf')
        min_bearish_dist = float('inf')

        for ob in active_obs:
            ob_mid = (ob.high + ob.low) / 2
            distance_pct = abs(current_price - ob_mid) / current_price * 100

            if distance_pct > max_distance_pct:
                continue

            if ob.type == 'bullish' and distance_pct < min_bullish_dist:
                min_bullish_dist = distance_pct
                nearest_bullish = ob
            elif ob.type == 'bearish' and distance_pct < min_bearish_dist:
                min_bearish_dist = distance_pct
                nearest_bearish = ob

        return {
            'nearest_bullish': nearest_bullish,
            'nearest_bearish': nearest_bearish,
            'total_active': len(active_obs),
            'in_bullish_ob': self._price_in_ob(current_price, nearest_bullish) if nearest_bullish else False,
            'in_bearish_ob': self._price_in_ob(current_price, nearest_bearish) if nearest_bearish else False
        }

    def _price_in_ob(self, price: float, ob: Optional[OrderBlock]) -> bool:
        """Проверяет, находится ли цена внутри Order Block"""
        if ob is None:
            return False
        return ob.low <= price <= ob.high

    # ==================== FAIR VALUE GAPS (FVG) ====================

    def find_fvg(self, df: pd.DataFrame) -> List[FairValueGap]:
        """
        Находит Fair Value Gaps (FVG) - зоны дисбаланса

        Bullish FVG: gap между high[i-2] и low[i] (цена не заполнила зону)
        Bearish FVG: gap между low[i-2] и high[i]

        Args:
            df: DataFrame со свечами

        Returns:
            Список FairValueGap объектов
        """
        if len(df) < 3:
            return []

        fvg_list = []

        for i in range(2, len(df)):
            # Bullish FVG: low текущей свечи выше high свечи 2 назад
            if df['low'].iloc[i] > df['high'].iloc[i - 2]:
                gap_size = (df['low'].iloc[i] - df['high'].iloc[i - 2]) / df['close'].iloc[i]

                if gap_size > self.fvg_min_size:
                    fvg = FairValueGap(
                        type='bullish',
                        top=float(df['low'].iloc[i]),
                        bottom=float(df['high'].iloc[i - 2]),
                        index=i,
                        size_pct=float(gap_size * 100),
                        mitigated=self._is_fvg_mitigated(
                            df, i, 'bullish',
                            df['high'].iloc[i - 2], df['low'].iloc[i]
                        ),
                        timestamp=str(df.index[i]) if isinstance(df.index[i], (pd.Timestamp, str)) else None
                    )
                    fvg_list.append(fvg)

            # Bearish FVG: high текущей свечи ниже low свечи 2 назад
            if df['high'].iloc[i] < df['low'].iloc[i - 2]:
                gap_size = (df['low'].iloc[i - 2] - df['high'].iloc[i]) / df['close'].iloc[i]

                if gap_size > self.fvg_min_size:
                    fvg = FairValueGap(
                        type='bearish',
                        top=float(df['low'].iloc[i - 2]),
                        bottom=float(df['high'].iloc[i]),
                        index=i,
                        size_pct=float(gap_size * 100),
                        mitigated=self._is_fvg_mitigated(
                            df, i, 'bearish',
                            df['high'].iloc[i], df['low'].iloc[i - 2]
                        ),
                        timestamp=str(df.index[i]) if isinstance(df.index[i], (pd.Timestamp, str)) else None
                    )
                    fvg_list.append(fvg)

        # Сортируем по индексу (от новых к старым)
        fvg_list.sort(key=lambda x: x.index, reverse=True)

        return fvg_list

    def _is_fvg_mitigated(
        self,
        df: pd.DataFrame,
        fvg_index: int,
        fvg_type: str,
        bottom: float,
        top: float
    ) -> bool:
        """
        Проверяет, был ли FVG заполнен (mitigated)

        Args:
            df: DataFrame со свечами
            fvg_index: Индекс FVG
            fvg_type: Тип FVG
            bottom: Нижняя граница
            top: Верхняя граница

        Returns:
            True если FVG заполнен
        """
        if fvg_index >= len(df) - 1:
            return False

        for i in range(fvg_index + 1, len(df)):
            if fvg_type == 'bullish':
                # Bullish FVG mitigated когда цена опускается в gap
                if df['low'].iloc[i] <= top:
                    return True
            else:  # bearish
                # Bearish FVG mitigated когда цена поднимается в gap
                if df['high'].iloc[i] >= bottom:
                    return True

        return False

    def get_unfilled_fvg(
        self,
        df: pd.DataFrame,
        current_price: float,
        max_distance_pct: float = 5.0
    ) -> Dict:
        """
        Получает незаполненные FVG рядом с текущей ценой

        Args:
            df: DataFrame со свечами
            current_price: Текущая цена
            max_distance_pct: Максимальное расстояние до FVG в %

        Returns:
            Dict с nearest_bullish и nearest_bearish FVG
        """
        fvg_list = self.find_fvg(df)

        # Фильтруем только незаполненные
        unfilled = [fvg for fvg in fvg_list if not fvg.mitigated]

        nearest_bullish = None
        nearest_bearish = None
        min_bullish_dist = float('inf')
        min_bearish_dist = float('inf')

        for fvg in unfilled:
            fvg_mid = (fvg.top + fvg.bottom) / 2
            distance_pct = abs(current_price - fvg_mid) / current_price * 100

            if distance_pct > max_distance_pct:
                continue

            if fvg.type == 'bullish' and distance_pct < min_bullish_dist:
                min_bullish_dist = distance_pct
                nearest_bullish = fvg
            elif fvg.type == 'bearish' and distance_pct < min_bearish_dist:
                min_bearish_dist = distance_pct
                nearest_bearish = fvg

        return {
            'nearest_bullish': nearest_bullish,
            'nearest_bearish': nearest_bearish,
            'total_unfilled': len(unfilled),
            'in_bullish_fvg': self._price_in_fvg(current_price, nearest_bullish) if nearest_bullish else False,
            'in_bearish_fvg': self._price_in_fvg(current_price, nearest_bearish) if nearest_bearish else False
        }

    def _price_in_fvg(self, price: float, fvg: Optional[FairValueGap]) -> bool:
        """Проверяет, находится ли цена внутри FVG"""
        if fvg is None:
            return False
        return fvg.bottom <= price <= fvg.top

    # ==================== LIQUIDITY ZONES ====================

    def find_liquidity_zones(self, df: pd.DataFrame, lookback: int = 50) -> List[LiquidityZone]:
        """
        Находит зоны ликвидности - места со скоплением стоп-лоссов

        - Equal highs/lows (двойные/тройные вершины)
        - Зоны выше swing high / ниже swing low

        Args:
            df: DataFrame со свечами
            lookback: Сколько свечей назад искать

        Returns:
            Список LiquidityZone объектов
        """
        liquidity_zones = []

        # Находим swing points
        swing_highs = self._find_swing_highs(df)
        swing_lows = self._find_swing_lows(df)

        # Ищем equal highs (buy-side liquidity - стопы шортов)
        for i, sh1 in enumerate(swing_highs):
            for sh2 in swing_highs[i + 1:]:
                price_diff = abs(sh1.price - sh2.price) / sh1.price
                if price_diff < self.equal_level_tolerance:
                    lz = LiquidityZone(
                        type='buy_side',
                        price=float(max(sh1.price, sh2.price)),
                        strength=2,  # Двойная вершина
                        indices=[sh1.index, sh2.index]
                    )
                    liquidity_zones.append(lz)

        # Ищем equal lows (sell-side liquidity - стопы лонгов)
        for i, sl1 in enumerate(swing_lows):
            for sl2 in swing_lows[i + 1:]:
                price_diff = abs(sl1.price - sl2.price) / sl1.price
                if price_diff < self.equal_level_tolerance:
                    lz = LiquidityZone(
                        type='sell_side',
                        price=float(min(sl1.price, sl2.price)),
                        strength=2,
                        indices=[sl1.index, sl2.index]
                    )
                    liquidity_zones.append(lz)

        return liquidity_zones

    # ==================== MARKET STRUCTURE ====================

    def _find_swing_highs(self, df: pd.DataFrame) -> List[SwingPoint]:
        """
        Находит swing highs (локальные максимумы)

        Args:
            df: DataFrame со свечами

        Returns:
            Список SwingPoint объектов
        """
        swing_highs = []
        lookback = self.swing_lookback

        if len(df) < lookback * 2 + 1:
            return swing_highs

        for i in range(lookback, len(df) - lookback):
            window_start = max(0, i - lookback)
            window_end = min(len(df), i + lookback + 1)

            if df['high'].iloc[i] == df['high'].iloc[window_start:window_end].max():
                swing_highs.append(SwingPoint(
                    price=float(df['high'].iloc[i]),
                    index=i,
                    type='high'
                ))

        return swing_highs

    def _find_swing_lows(self, df: pd.DataFrame) -> List[SwingPoint]:
        """
        Находит swing lows (локальные минимумы)

        Args:
            df: DataFrame со свечами

        Returns:
            Список SwingPoint объектов
        """
        swing_lows = []
        lookback = self.swing_lookback

        if len(df) < lookback * 2 + 1:
            return swing_lows

        for i in range(lookback, len(df) - lookback):
            window_start = max(0, i - lookback)
            window_end = min(len(df), i + lookback + 1)

            if df['low'].iloc[i] == df['low'].iloc[window_start:window_end].min():
                swing_lows.append(SwingPoint(
                    price=float(df['low'].iloc[i]),
                    index=i,
                    type='low'
                ))

        return swing_lows

    def analyze_market_structure(self, df: pd.DataFrame) -> Dict:
        """
        Анализирует рыночную структуру

        HH + HL = Bullish (восходящий тренд)
        LH + LL = Bearish (нисходящий тренд)

        Args:
            df: DataFrame со свечами

        Returns:
            Dict с информацией о структуре рынка
        """
        swing_highs = self._find_swing_highs(df)
        swing_lows = self._find_swing_lows(df)

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {
                'trend': TrendType.UNDEFINED.value,
                'structure': [],
                'higher_high': False,
                'higher_low': False,
                'lower_high': False,
                'lower_low': False,
                'last_swing_high': swing_highs[-1].price if swing_highs else None,
                'last_swing_low': swing_lows[-1].price if swing_lows else None
            }

        # Анализируем последние swing points
        hh = swing_highs[-1].price > swing_highs[-2].price  # Higher High
        hl = swing_lows[-1].price > swing_lows[-2].price    # Higher Low
        lh = swing_highs[-1].price < swing_highs[-2].price  # Lower High
        ll = swing_lows[-1].price < swing_lows[-2].price    # Lower Low

        # Определяем тренд
        if hh and hl:
            trend = TrendType.BULLISH.value
        elif lh and ll:
            trend = TrendType.BEARISH.value
        else:
            trend = TrendType.RANGING.value

        return {
            'trend': trend,
            'higher_high': hh,
            'higher_low': hl,
            'lower_high': lh,
            'lower_low': ll,
            'last_swing_high': float(swing_highs[-1].price),
            'last_swing_low': float(swing_lows[-1].price),
            'swing_highs_count': len(swing_highs),
            'swing_lows_count': len(swing_lows)
        }

    def detect_bos(self, df: pd.DataFrame) -> Dict:
        """
        Детектирует Break of Structure (BOS)

        Bullish BOS: пробой предыдущего swing high
        Bearish BOS: пробой предыдущего swing low

        Args:
            df: DataFrame со свечами

        Returns:
            Dict с информацией о BOS
        """
        swing_highs = self._find_swing_highs(df)
        swing_lows = self._find_swing_lows(df)

        if not swing_highs and not swing_lows:
            return {'type': 'none', 'broken_level': None, 'strength': 0}

        current_price = df['close'].iloc[-1]

        # Проверяем пробой последнего swing high
        if swing_highs:
            last_sh = swing_highs[-1].price
            if current_price > last_sh:
                strength = (current_price - last_sh) / last_sh * 100
                return {
                    'type': 'bullish',
                    'broken_level': float(last_sh),
                    'current_price': float(current_price),
                    'strength': float(strength)
                }

        # Проверяем пробой последнего swing low
        if swing_lows:
            last_sl = swing_lows[-1].price
            if current_price < last_sl:
                strength = (last_sl - current_price) / last_sl * 100
                return {
                    'type': 'bearish',
                    'broken_level': float(last_sl),
                    'current_price': float(current_price),
                    'strength': float(strength)
                }

        return {'type': 'none', 'broken_level': None, 'strength': 0}

    def detect_choch(self, df: pd.DataFrame) -> Dict:
        """
        Детектирует Change of Character (CHoCH) - первый признак смены тренда

        В восходящем тренде: первый LL (пробой swing low)
        В нисходящем тренде: первый HH (пробой swing high)

        Args:
            df: DataFrame со свечами

        Returns:
            Dict с информацией о CHoCH
        """
        structure = self.analyze_market_structure(df)
        bos = self.detect_bos(df)

        # CHoCH = BOS против текущего тренда
        if structure['trend'] == TrendType.BULLISH.value and bos['type'] == 'bearish':
            return {
                'detected': True,
                'type': 'bearish_choch',
                'message': 'Смена тренда с бычьего на медвежий',
                'broken_level': bos.get('broken_level')
            }
        elif structure['trend'] == TrendType.BEARISH.value and bos['type'] == 'bullish':
            return {
                'detected': True,
                'type': 'bullish_choch',
                'message': 'Смена тренда с медвежьего на бычий',
                'broken_level': bos.get('broken_level')
            }

        return {'detected': False, 'type': None, 'message': None, 'broken_level': None}

    # ==================== PRICE ZONES ====================

    def get_price_zone(self, df: pd.DataFrame, lookback: int = 50) -> Dict:
        """
        Определяет зону цены: Premium, Discount или Equilibrium

        Premium (верхние 50%): зона для продаж
        Discount (нижние 50%): зона для покупок
        Equilibrium: середина диапазона

        Args:
            df: DataFrame со свечами
            lookback: Сколько свечей использовать для определения диапазона

        Returns:
            Dict с информацией о ценовой зоне
        """
        if len(df) < 2:
            return {
                'zone': ZoneType.EQUILIBRIUM.value,
                'position_pct': 50.0,
                'equilibrium': None,
                'range_high': None,
                'range_low': None,
                'recommendation': 'Недостаточно данных'
            }

        recent = df.iloc[-lookback:] if len(df) >= lookback else df
        high = recent['high'].max()
        low = recent['low'].min()
        current = df['close'].iloc[-1]

        range_size = high - low

        if range_size <= 0:
            return {
                'zone': ZoneType.EQUILIBRIUM.value,
                'position_pct': 50.0,
                'equilibrium': float(current),
                'range_high': float(high),
                'range_low': float(low),
                'recommendation': 'Нейтральная зона (низкая волатильность)'
            }

        equilibrium = low + range_size * 0.5
        position = (current - low) / range_size

        if position > 0.7:
            zone = ZoneType.PREMIUM.value
            recommendation = 'Зона для продаж (SHORT)'
        elif position < 0.3:
            zone = ZoneType.DISCOUNT.value
            recommendation = 'Зона для покупок (LONG)'
        else:
            zone = ZoneType.EQUILIBRIUM.value
            recommendation = 'Нейтральная зона'

        return {
            'zone': zone,
            'position_pct': float(position * 100),
            'equilibrium': float(equilibrium),
            'range_high': float(high),
            'range_low': float(low),
            'recommendation': recommendation
        }

    # ==================== КОМПЛЕКСНЫЙ СИГНАЛ ====================

    def get_smc_signal(self, df: pd.DataFrame) -> Dict:
        """
        Возвращает комплексный сигнал на основе всех SMC факторов

        Args:
            df: DataFrame со свечами (минимум 60 свечей рекомендуется)

        Returns:
            Dict с комплексным сигналом:
            - signal: 'LONG', 'SHORT', 'WAIT'
            - score: -100 до +100
            - confidence: 0-100
            - reasons: список причин
            - entry_zone: оптимальная зона входа
            - rsi, structure, price_zone и другие детали
        """
        if len(df) < 10:
            return {
                'signal': SignalType.WAIT.value,
                'score': 0,
                'confidence': 0,
                'reasons': ['Недостаточно данных для анализа'],
                'rsi': None,
                'structure': None,
                'price_zone': None
            }

        current_price = df['close'].iloc[-1]
        reasons = []
        score = 0  # -100 до +100

        # === RSI ===
        rsi_data = self.get_rsi_signal(df)
        current_rsi = rsi_data['value']

        if current_rsi is not None:
            if current_rsi <= 30:
                score += 30
                reasons.append(f'RSI перепродан ({current_rsi:.1f})')
            elif current_rsi >= 70:
                score -= 30
                reasons.append(f'RSI перекуплен ({current_rsi:.1f})')

            # RSI дивергенция
            if rsi_data['divergence'] == 'bullish':
                score += 20
                reasons.append('Bullish RSI дивергенция')
            elif rsi_data['divergence'] == 'bearish':
                score -= 20
                reasons.append('Bearish RSI дивергенция')

        # === Order Blocks ===
        ob_data = self.get_active_order_blocks(df, current_price)

        if ob_data['in_bullish_ob']:
            score += 25
            reasons.append('Цена в зоне Bullish Order Block')
        if ob_data['in_bearish_ob']:
            score -= 25
            reasons.append('Цена в зоне Bearish Order Block')

        # === FVG ===
        fvg_data = self.get_unfilled_fvg(df, current_price)

        if fvg_data['in_bullish_fvg']:
            score += 15
            reasons.append('Цена в Bullish FVG (незаполненный)')
        if fvg_data['in_bearish_fvg']:
            score -= 15
            reasons.append('Цена в Bearish FVG (незаполненный)')

        # === Market Structure ===
        structure = self.analyze_market_structure(df)

        if structure['trend'] == TrendType.BULLISH.value:
            score += 10
            reasons.append('Бычья структура рынка (HH+HL)')
        elif structure['trend'] == TrendType.BEARISH.value:
            score -= 10
            reasons.append('Медвежья структура рынка (LH+LL)')

        # === CHoCH ===
        choch = self.detect_choch(df)

        if choch['detected']:
            if choch['type'] == 'bullish_choch':
                score += 30
                reasons.append('CHoCH: разворот на бычий')
            elif choch['type'] == 'bearish_choch':
                score -= 30
                reasons.append('CHoCH: разворот на медвежий')

        # === Price Zone ===
        price_zone = self.get_price_zone(df)

        if price_zone['zone'] == ZoneType.DISCOUNT.value and score > 0:
            score += 10
            reasons.append('Цена в Discount зоне (хорошо для LONG)')
        elif price_zone['zone'] == ZoneType.PREMIUM.value and score < 0:
            score -= 10
            reasons.append('Цена в Premium зоне (хорошо для SHORT)')

        # === Определяем сигнал ===
        if score >= 40:
            signal = SignalType.LONG.value
            confidence = min(abs(score), 100)
        elif score <= -40:
            signal = SignalType.SHORT.value
            confidence = min(abs(score), 100)
        else:
            signal = SignalType.WAIT.value
            confidence = 100 - abs(score)

        # === Определяем зону входа ===
        entry_zone = None
        if signal == SignalType.LONG.value:
            if ob_data['nearest_bullish']:
                entry_zone = {
                    'type': 'order_block',
                    'high': ob_data['nearest_bullish'].high,
                    'low': ob_data['nearest_bullish'].low
                }
            elif fvg_data['nearest_bullish']:
                entry_zone = {
                    'type': 'fvg',
                    'high': fvg_data['nearest_bullish'].top,
                    'low': fvg_data['nearest_bullish'].bottom
                }
        elif signal == SignalType.SHORT.value:
            if ob_data['nearest_bearish']:
                entry_zone = {
                    'type': 'order_block',
                    'high': ob_data['nearest_bearish'].high,
                    'low': ob_data['nearest_bearish'].low
                }
            elif fvg_data['nearest_bearish']:
                entry_zone = {
                    'type': 'fvg',
                    'high': fvg_data['nearest_bearish'].top,
                    'low': fvg_data['nearest_bearish'].bottom
                }

        return {
            'signal': signal,
            'score': score,
            'confidence': confidence,
            'reasons': reasons,
            'entry_zone': entry_zone,
            'rsi': current_rsi,
            'rsi_zone': rsi_data['zone'],
            'structure': structure['trend'],
            'price_zone': price_zone['zone'],
            'order_blocks_active': ob_data['total_active'],
            'fvg_unfilled': fvg_data['total_unfilled'],
            'current_price': float(current_price)
        }

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Вычисляет все SMC признаки и добавляет их в DataFrame

        Args:
            df: DataFrame со свечами

        Returns:
            DataFrame с добавленными SMC признаками
        """
        features = df.copy()

        # RSI
        features['rsi'] = self.compute_rsi(df)

        # RSI зоны
        features['rsi_oversold'] = (features['rsi'] <= 30).astype(int)
        features['rsi_overbought'] = (features['rsi'] >= 70).astype(int)

        # Price zone
        price_zone = self.get_price_zone(df)
        features['price_position_pct'] = price_zone['position_pct']
        features['in_discount'] = (price_zone['zone'] == ZoneType.DISCOUNT.value)
        features['in_premium'] = (price_zone['zone'] == ZoneType.PREMIUM.value)

        # Market structure
        structure = self.analyze_market_structure(df)
        features['trend_bullish'] = (structure['trend'] == TrendType.BULLISH.value)
        features['trend_bearish'] = (structure['trend'] == TrendType.BEARISH.value)

        # BOS
        bos = self.detect_bos(df)
        features['bos_bullish'] = (bos['type'] == 'bullish')
        features['bos_bearish'] = (bos['type'] == 'bearish')

        return features

# ==================== ТЕСТОВЫЙ КОД ====================

if __name__ == '__main__':
    import sys

    print("=" * 60)
    print("Smart Money Concepts (SMC) - Тест модуля")
    print("=" * 60)

    # Создаем тестовые данные
    np.random.seed(42)
    n_candles = 100

    # Генерируем реалистичные свечи
    base_price = 100.0
    prices = [base_price]

    for i in range(1, n_candles):
        change = np.random.randn() * 0.02  # 2% волатильность
        prices.append(prices[-1] * (1 + change))

    # Создаем OHLCV данные
    data = {
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'volume': []
    }

    for i in range(n_candles):
        o = prices[i]
        c = o * (1 + np.random.randn() * 0.01)
        h = max(o, c) * (1 + abs(np.random.randn() * 0.005))
        l = min(o, c) * (1 - abs(np.random.randn() * 0.005))
        v = np.random.randint(1000, 10000)

        data['open'].append(o)
        data['high'].append(h)
        data['low'].append(l)
        data['close'].append(c)
        data['volume'].append(v)

    df = pd.DataFrame(data)

    # Тестируем SMC
    smc = SmartMoneyFeatures()

    print("\n1. RSI сигнал:")
    rsi_signal = smc.get_rsi_signal(df)
    print(f"   RSI: {rsi_signal['value']:.2f}")
    print(f"   Зона: {rsi_signal['zone']}")
    print(f"   Сигнал: {rsi_signal['signal']}")
    print(f"   Дивергенция: {rsi_signal['divergence']}")

    print("\n2. Order Blocks:")
    obs = smc.find_order_blocks(df)
    print(f"   Найдено Order Blocks: {len(obs)}")
    for ob in obs[:3]:  # Показываем первые 3
        print(f"   - {ob.type}: {ob.low:.2f} - {ob.high:.2f}, сила: {ob.strength:.2%}, протестирован: {ob.tested}")

    print("\n3. Fair Value Gaps:")
    fvgs = smc.find_fvg(df)
    print(f"   Найдено FVG: {len(fvgs)}")
    for fvg in fvgs[:3]:
        print(f"   - {fvg.type}: {fvg.bottom:.2f} - {fvg.top:.2f}, размер: {fvg.size_pct:.2f}%, заполнен: {fvg.mitigated}")

    print("\n4. Market Structure:")
    structure = smc.analyze_market_structure(df)
    print(f"   Тренд: {structure['trend']}")
    print(f"   HH: {structure['higher_high']}, HL: {structure['higher_low']}")
    print(f"   LH: {structure['lower_high']}, LL: {structure['lower_low']}")
    print(f"   Последний swing high: {structure['last_swing_high']:.2f}")
    print(f"   Последний swing low: {structure['last_swing_low']:.2f}")

    print("\n5. Break of Structure:")
    bos = smc.detect_bos(df)
    print(f"   Тип: {bos['type']}")
    if bos['broken_level']:
        print(f"   Пробитый уровень: {bos['broken_level']:.2f}")
        print(f"   Сила: {bos['strength']:.2f}%")

    print("\n6. Change of Character:")
    choch = smc.detect_choch(df)
    print(f"   Обнаружен: {choch['detected']}")
    if choch['detected']:
        print(f"   Тип: {choch['type']}")
        print(f"   Сообщение: {choch['message']}")

    print("\n7. Price Zone:")
    zone = smc.get_price_zone(df)
    print(f"   Зона: {zone['zone']}")
    print(f"   Позиция: {zone['position_pct']:.1f}%")
    print(f"   Рекомендация: {zone['recommendation']}")

    print("\n8. Liquidity Zones:")
    liq = smc.find_liquidity_zones(df)
    print(f"   Найдено зон ликвидности: {len(liq)}")
    for lz in liq[:3]:
        print(f"   - {lz.type}: {lz.price:.2f}, сила: {lz.strength}")

    print("\n" + "=" * 60)
    print("КОМПЛЕКСНЫЙ СИГНАЛ SMC")
    print("=" * 60)

    signal = smc.get_smc_signal(df)
    print(f"\nСигнал: {signal['signal']}")
    print(f"Score: {signal['score']}")
    print(f"Уверенность: {signal['confidence']}%")
    print(f"\nПричины:")
    for reason in signal['reasons']:
        print(f"   - {reason}")

    if signal['entry_zone']:
        print(f"\nЗона входа:")
        print(f"   Тип: {signal['entry_zone']['type']}")
        print(f"   Диапазон: {signal['entry_zone']['low']:.2f} - {signal['entry_zone']['high']:.2f}")

    print(f"\nДетали:")
    rsi_value = f"{signal['rsi']:.2f}" if signal['rsi'] else 'N/A'
    print(f"   RSI: {rsi_value}")
    print(f"   Структура: {signal['structure']}")
    print(f"   Ценовая зона: {signal['price_zone']}")
    print(f"   Активных OB: {signal['order_blocks_active']}")
    print(f"   Незаполненных FVG: {signal['fvg_unfilled']}")

    print("\n" + "=" * 60)
    print("[OK] Все тесты пройдены успешно!")
    print("=" * 60)
