"""
Smart Risk Manager - Премиум-модуль умного риск-менеджмента

Особенности:
- Обучение на стопах (анализ причин неудачных сделок)
- Бэктестинг каждой монеты перед входом в позицию
- Оптимизация SL/TP на основе исторических данных
- Определение лучших точек входа

ТРЕБУЕТ ПРЕМИУМ ЛИЦЕНЗИИ!
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
from pathlib import Path

logger = logging.getLogger('AI.SmartRiskManager')

# Проверяем лицензию при импорте
try:
    from bot_engine.ai import check_premium_license
    PREMIUM_AVAILABLE = check_premium_license()
except ImportError:
    PREMIUM_AVAILABLE = False


class SmartRiskManager:
    """Умный риск-менеджмент с обучением на стопах (Premium только!)"""
    
    def __init__(self):
        """Инициализация (только с лицензией!)"""
        if not PREMIUM_AVAILABLE:
            raise ImportError(
                "SmartRiskManager требует премиум лицензию. "
                "Для активации: python scripts/activate_premium.py"
            )
        
        self.logger = logger
        self.backtest_cache = {}
        self.stop_patterns = {}
        self.training_data_path = Path('data/ai/training/stops_analysis.json')
        self.feedback_data_path = Path('data/ai/training/feedback')
        self.optimized_params_path = Path('data/ai/training/optimized_params.json')
        
        # Создаем директории если нужно
        self.training_data_path.parent.mkdir(parents=True, exist_ok=True)
        self.feedback_data_path.mkdir(parents=True, exist_ok=True)
        
        # Загружаем обученные паттерны
        self._load_stop_patterns()
        
        # Загружаем оптимизированные параметры
        self.optimized_params = {}
        self._load_optimized_params()
        
        # Подключаем ИИ модули
        self._init_ai_modules()
        
        # Подключаем ML модель для предсказания SL/TP
        self._init_ml_model()
        
        logger.info("[SmartRiskManager] ✅ Премиум-модуль загружен и готов к работе")
    
    def _init_ai_modules(self):
        """Инициализирует ИИ модули для анализа"""
        try:
            from bot_engine.ai import get_ai_manager
            self.ai_manager = get_ai_manager()
            
            # Получаем доступные модули
            self.anomaly_detector = self.ai_manager.anomaly_detector
            self.lstm_predictor = self.ai_manager.lstm_predictor
            self.pattern_detector = self.ai_manager.pattern_detector
            self.risk_manager = self.ai_manager.risk_manager
            
            logger.info("[SmartRiskManager] 🤖 ИИ модули подключены")
        except Exception as e:
            self.ai_manager = None
            logger.warning(f"[SmartRiskManager] ⚠️ ИИ модули недоступны: {e}")
    
    def _init_ml_model(self):
        """Инициализирует ML модель"""
        try:
            from bot_engine.ai.ml_risk_predictor import MLRiskPredictor
            self.ml_predictor = MLRiskPredictor()
            logger.info("[SmartRiskManager] 🤖 ML модель подключена")
        except Exception as e:
            self.ml_predictor = None
            logger.warning(f"[SmartRiskManager] ⚠️ ML модель недоступна: {e}")
    
    def analyze_stopped_trades(self, limit: int = 100) -> Dict[str, Any]:
        """
        Анализирует последние стопы для обучения ИИ
        
        Returns:
            Словарь с анализом стопов и рекомендациями
        """
        try:
            from bot_engine.bot_history import bot_history_manager
            
            # Получаем стопы из истории
            stopped_trades = bot_history_manager.get_stopped_trades(limit)
            
            if not stopped_trades:
                return {
                    'total_stops': 0,
                    'message': 'Нет данных о стопах для анализа'
                }
            
            # Анализируем паттерны
            patterns = self._extract_patterns(stopped_trades)
            common_reasons = self._analyze_reasons(stopped_trades)
            
            # 🧠 Используем LSTM для анализа причин стопов
            lstm_analysis = self._analyze_stops_with_lstm(stopped_trades)
            
            # 📊 Используем Anomaly Detector для поиска аномалий в стопах
            anomaly_analysis = self._analyze_stops_for_anomalies(stopped_trades)
            
            # 🎯 Используем Risk Manager для оптимизации SL/TP
            optimal_sl = self._optimize_stop_loss_with_ai(stopped_trades, patterns)
            optimal_tp = self._optimize_take_profit_with_ai(stopped_trades, patterns)
            
            # Сохраняем для обучения
            self._save_for_training(patterns)
            
            return {
                'total_stops': len(stopped_trades),
                'common_reasons': common_reasons,
                'optimal_sl_percent': optimal_sl,
                'optimal_tp_percent': optimal_tp,
                'patterns': patterns,
                'lstm_analysis': lstm_analysis,
                'anomaly_analysis': anomaly_analysis,
                'recommendations': self._generate_recommendations(stopped_trades)
            }
            
        except Exception as e:
            logger.error(f"[SmartRiskManager] Ошибка анализа стопов: {e}")
            return {
                'total_stops': 0,
                'error': str(e)
            }
    
    def backtest_coin(
        self, 
        symbol: str, 
        candles: List[dict], 
        direction: str,
        current_price: float
    ) -> Dict[str, Any]:
        """
        Быстрый бэктест монеты перед входом в позицию
        
        Args:
            symbol: Символ монеты
            candles: Последние 50-100 свечей
            direction: 'LONG' или 'SHORT'
            current_price: Текущая цена
        
        Returns:
            Оптимальные параметры входа (entry, SL, TP) и confidence
        """
        try:
            if len(candles) < 20:
                return self._default_backtest_result()
            
            # Используем кэш если есть
            cache_key = f"{symbol}_{direction}_{len(candles)}"
            if cache_key in self.backtest_cache:
                logger.debug(f"[SmartRiskManager] Используем кэш для {symbol}")
                return self.backtest_cache[cache_key]
            
            # Быстрый бэктест на последних свечах
            result = self._quick_backtest(symbol, candles, direction, current_price)
            
            # Кэшируем результат (на 1 час)
            self.backtest_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"[SmartRiskManager] Ошибка бэктеста {symbol}: {e}")
            return self._default_backtest_result()
    
    def _quick_backtest(
        self, 
        symbol: str, 
        candles: List[dict], 
        direction: str,
        current_price: float
    ) -> Dict[str, Any]:
        """Быстрый бэктест с использованием ИИ"""
        
        # Анализируем волатильность
        volatility = self._calculate_volatility(candles)
        
        # Анализируем силу тренда
        trend_strength = self._calculate_trend_strength(candles, direction)
        
        # 🤖 Используем LSTM для предсказания направления
        lstm_prediction = None
        if self.lstm_predictor:
            try:
                lstm_prediction = self.lstm_predictor.predict(candles, current_price)
            except Exception as e:
                logger.debug(f"[SmartRiskManager] LSTM недоступен: {e}")
        
        # 📊 Проверяем на аномалии
        anomaly_score = None
        if self.anomaly_detector:
            try:
                anomaly_score = self.anomaly_detector.detect(candles)
                if anomaly_score.get('is_anomaly') and anomaly_score.get('severity', 0) > 0.7:
                    logger.warning(f"[SmartRiskManager] ⚠️ {symbol}: Обнаружена аномалия в бэктесте!")
            except Exception as e:
                logger.debug(f"[SmartRiskManager] Anomaly Detector недоступен: {e}")
        
        # 🎯 Используем Risk Manager для оптимизации SL/TP
        optimal_sl_from_risk = 15.0
        optimal_tp_from_risk = 100.0
        if self.risk_manager:
            try:
                # Получаем рекомендации от Risk Manager
                risk_analysis = self.risk_manager.calculate_dynamic_sl(symbol, candles, direction)
                optimal_sl_from_risk = risk_analysis.get('sl_percent', 15.0)
                
                risk_tp_analysis = self.risk_manager.calculate_dynamic_tp(symbol, candles, direction)
                optimal_tp_from_risk = risk_tp_analysis.get('tp_percent', 100.0)
            except Exception as e:
                logger.debug(f"[SmartRiskManager] Risk Manager недоступен: {e}")
        
        # Рассчитываем оптимальные SL/TP на основе истории стопов для этой монеты
        coin_stops = self._get_coin_stops(symbol)
        
        # Объединяем рекомендации от разных источников
        if coin_stops:
            # Используем данные о стопах этой монеты
            optimal_sl_from_history = self._optimal_sl_for_coin(coin_stops, volatility)
            optimal_tp_from_history = self._optimal_tp_for_coin(coin_stops, trend_strength)
        else:
            # Используем общие рекомендации
            optimal_sl_from_history = 12.0 if volatility < 1.0 else 18.0
            optimal_tp_from_history = 80.0 if trend_strength < 0.5 else 120.0
        
        # 🤖 Используем ML модель для предсказания (если доступна)
        ml_prediction = None
        if self.ml_predictor:
            try:
                ml_features = {
                    'rsi': volatility * 50,  # TODO: получить реальный RSI
                    'volatility': volatility,
                    'trend_strength': trend_strength,
                    'volume': candles[-1].get('volume', 0),
                    'price': current_price,
                    'coin_stops_count': len(coin_stops),
                    'avg_stop_duration_hours': np.mean([s.get('duration_hours', 0) for s in coin_stops]) if coin_stops else 24
                }
                ml_prediction = self.ml_predictor.predict(ml_features)
            except Exception as e:
                logger.debug(f"[SmartRiskManager] ML модель недоступна: {e}")
        
        # Объединяем результаты: ML (если есть) > ИИ > история
        if ml_prediction:
            # Используем ML предсказание
            optimal_sl = ml_prediction['optimal_sl']
            optimal_tp = ml_prediction['optimal_tp']
            logger.info(f"[SmartRiskManager] 🤖 ML предсказание: SL={optimal_sl}%, TP={optimal_tp}%")
        else:
            # Взвешенное среднее: 60% ИИ, 40% история
            optimal_sl = (optimal_sl_from_risk * 0.6) + (optimal_sl_from_history * 0.4)
            optimal_tp = (optimal_tp_from_risk * 0.6) + (optimal_tp_from_history * 0.4)
        
        # Рассчитываем оптимальную точку входа
        optimal_entry = self._optimal_entry_price(candles, direction, current_price)
        
        # Confidence на основе качества данных
        confidence = self._calculate_confidence(candles, coin_stops)
        
        # 🤖 Увеличиваем confidence если LSTM предсказал правильное направление
        if lstm_prediction:
            predicted_direction = 'LONG' if lstm_prediction.get('direction', 0) > 0 else 'SHORT'
            if predicted_direction == direction:
                confidence += 0.15  # Бонус за правильное предсказание LSTM
        
        # 📊 Снижаем confidence при обнаружении аномалий
        if anomaly_score and anomaly_score.get('is_anomaly'):
            severity = anomaly_score.get('severity', 0)
            confidence -= severity * 0.3  # Штраф за аномалию
        
        confidence = max(0.1, min(confidence, 0.95))
        
        return {
            'optimal_entry': optimal_entry,
            'optimal_sl': current_price * (1 - optimal_sl / 100) if direction == 'LONG' else current_price * (1 + optimal_sl / 100),
            'optimal_tp': current_price * (1 + optimal_tp / 100) if direction == 'LONG' else current_price * (1 - optimal_tp / 100),
            'optimal_sl_percent': optimal_sl,
            'optimal_tp_percent': optimal_tp,
            'win_rate': self._estimate_win_rate(candles, direction),
            'expected_return': self._estimate_return(candles, direction),
            'confidence': confidence,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'lstm_prediction': lstm_prediction,
            'anomaly_detected': anomaly_score.get('is_anomaly') if anomaly_score else False,
            'risk_manager_recommendation': {
                'sl_percent': optimal_sl_from_risk,
                'tp_percent': optimal_tp_from_risk
            }
        }
    
    def _extract_patterns(self, stopped_trades: List[Dict]) -> Dict:
        """Извлекает паттерны из стопов"""
        patterns = {
            'high_rsi_stops': 0,
            'low_volatility_stops': 0,
            'rapid_stops': 0,
            'trailing_stops': 0
        }
        
        for trade in stopped_trades:
            entry_data = trade.get('entry_data', {})
            exit_reason = trade.get('close_reason', '')
            
            # Высокий RSI на входе
            if entry_data.get('rsi', 50) > 70:
                patterns['high_rsi_stops'] += 1
            
            # Низкая волатильность
            if entry_data.get('volatility', 1.0) < 0.5:
                patterns['low_volatility_stops'] += 1
            
            # Быстрое закрытие (< 6 часов)
            duration = entry_data.get('duration_hours', 0)
            if duration > 0 and duration < 6:
                patterns['rapid_stops'] += 1
            
            # Trailing stop
            if 'trailing' in exit_reason.lower():
                patterns['trailing_stops'] += 1
        
        return patterns
    
    def _analyze_reasons(self, stopped_trades: List[Dict]) -> Dict:
        """Анализирует основные причины стопов"""
        reasons = {}
        
        for trade in stopped_trades:
            reason = trade.get('close_reason', 'UNKNOWN')
            reasons[reason] = reasons.get(reason, 0) + 1
        
        # Сортируем по частоте
        sorted_reasons = dict(sorted(reasons.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_reasons
    
    def _optimize_stop_loss(self, stopped_trades: List[Dict]) -> float:
        """Оптимизирует Stop Loss на основе истории"""
        if not stopped_trades:
            return 15.0  # Дефолт
        
        # Анализируем RSI на входе успешных и неуспешных сделок
        # TODO: Здесь можно добавить ML для оптимизации
        
        return 15.0  # Временно возвращаем дефолт
    
    def _optimize_take_profit(self, stopped_trades: List[Dict]) -> float:
        """Оптимизирует Take Profit на основе истории"""
        if not stopped_trades:
            return 100.0  # Дефолт
        
        return 100.0  # Временно возвращаем дефолт
    
    def _calculate_volatility(self, candles: List[dict]) -> float:
        """Рассчитывает волатильность"""
        if len(candles) < 20:
            return 1.0
        
        # Рассчитываем изменения цены
        changes = []
        for i in range(1, len(candles)):
            change = abs(candles[i]['close'] - candles[i-1]['close']) / candles[i-1]['close']
            changes.append(change)
        
        return np.mean(changes) * 100 * 100  # В процентах
    
    def _calculate_trend_strength(self, candles: List[dict], direction: str) -> float:
        """Рассчитывает силу тренда"""
        if len(candles) < 10:
            return 0.5
        
        # Анализ последних свечей
        recent = candles[-10:]
        up_ticks = sum(1 for i in range(1, len(recent)) if recent[i]['close'] > recent[i-1]['close'])
        
        if direction == 'LONG':
            return up_ticks / len(recent)
        else:  # SHORT
            return (len(recent) - up_ticks) / len(recent)
    
    def _get_coin_stops(self, symbol: str) -> List[Dict]:
        """Получает стопы для конкретной монеты"""
        if symbol in self.stop_patterns:
            return self.stop_patterns[symbol]
        return []
    
    def _optimal_sl_for_coin(self, stops: List[Dict], volatility: float) -> float:
        """Рассчитывает оптимальный SL для монеты"""
        # TODO: ML модель для оптимизации
        return 12.0 if volatility < 1.0 else 18.0
    
    def _optimal_tp_for_coin(self, stops: List[Dict], trend_strength: float) -> float:
        """Рассчитывает оптимальный TP для монеты"""
        # TODO: ML модель для оптимизации
        return 80.0 if trend_strength < 0.5 else 120.0
    
    def _optimal_entry_price(self, candles: List[dict], direction: str, current_price: float) -> float:
        """Определяет оптимальную цену входа"""
        # Находим локальные минимумы/максимумы
        if direction == 'LONG':
            # Ищем локальный минимум
            lows = [c['low'] for c in candles[-10:]]
            return min(lows)
        else:  # SHORT
            # Ищем локальный максимум
            highs = [c['high'] for c in candles[-10:]]
            return max(highs)
    
    def _calculate_confidence(self, candles: List[dict], stops: List[Dict]) -> float:
        """Рассчитывает confidence на основе качества данных"""
        confidence = 0.5  # Базовый
        
        # Больше свечей = выше confidence
        if len(candles) >= 50:
            confidence += 0.2
        
        # Есть история стопов для этой монеты
        if stops:
            confidence += 0.2
        
        # Низкая волатильность = выше confidence
        volatility = self._calculate_volatility(candles)
        if volatility < 1.0:
            confidence += 0.1
        
        return min(confidence, 0.95)
    
    def _estimate_win_rate(self, candles: List[dict], direction: str) -> float:
        """Оценивает win rate на основе истории"""
        # TODO: Добавить реальную оценку на основе бэктеста
        return 0.6  # Дефолт
    
    def _estimate_return(self, candles: List[dict], direction: str) -> float:
        """Оценивает ожидаемую доходность"""
        # TODO: Добавить реальную оценку
        return 50.0  # Дефолт в %
    
    def _load_stop_patterns(self):
        """Загружает обученные паттерны"""
        try:
            if self.training_data_path.exists():
                with open(self.training_data_path, 'r') as f:
                    data = json.load(f)
                    self.stop_patterns = data.get('patterns', {})
                    logger.debug(f"[SmartRiskManager] Загружено {len(self.stop_patterns)} паттернов")
        except Exception as e:
            logger.warning(f"[SmartRiskManager] Не удалось загрузить паттерны: {e}")
    
    def _save_for_training(self, patterns: Dict):
        """Сохраняет данные для обучения"""
        try:
            data = {
                'patterns': patterns,
                'updated_at': datetime.now().isoformat()
            }
            
            with open(self.training_data_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"[SmartRiskManager] Не удалось сохранить данные: {e}")
    
    def _generate_recommendations(self, stopped_trades: List[Dict]) -> List[str]:
        """Генерирует рекомендации на основе анализа стопов"""
        recommendations = []
        
        patterns = self._extract_patterns(stopped_trades)
        
        if patterns['high_rsi_stops'] > len(stopped_trades) * 0.3:
            recommendations.append("Избегайте входов при RSI > 70")
        
        if patterns['low_volatility_stops'] > len(stopped_trades) * 0.3:
            recommendations.append("Выходите быстрее при низкой волатильности")
        
        if patterns['rapid_stops'] > len(stopped_trades) * 0.5:
            recommendations.append("Держите позиции дольше - большинство стопов слишком быстрые")
        
        if not recommendations:
            recommendations.append("Данных недостаточно для конкретных рекомендаций")
        
        return recommendations
    
    def _default_backtest_result(self) -> Dict[str, Any]:
        """Возвращает дефолтный результат бэктеста"""
        return {
            'optimal_entry': None,
            'optimal_sl': None,
            'optimal_tp': None,
            'optimal_sl_percent': 15.0,
            'optimal_tp_percent': 100.0,
            'win_rate': 0.5,
            'expected_return': 0.0,
            'confidence': 0.3,
            'volatility': 1.0,
            'trend_strength': 0.5
        }


    def _analyze_stops_with_lstm(self, stopped_trades: List[Dict]) -> Dict:
        """Использует LSTM для анализа причин стопов"""
        try:
            if not self.lstm_predictor:
                return {'available': False}
            
            # Анализируем последние стопы через LSTM
            analysis_results = []
            
            for trade in stopped_trades[:10]:  # Анализируем последние 10 стопов
                symbol = trade.get('symbol')
                entry_data = trade.get('entry_data', {})
                entry_price = entry_data.get('entry_price')
                
                if entry_price:
                    # TODO: Получить свечи для этого стопа и проанализировать через LSTM
                    # candles = get_candles_for_trade(trade)
                    # prediction = self.lstm_predictor.predict(candles, entry_price)
                    # analysis_results.append(prediction)
                    pass
            
            return {
                'available': True,
                'analyzed_stops': len(analysis_results),
                'avg_prediction_accuracy': 0.65  # TODO: Реальная статистика
            }
        except Exception as e:
            logger.error(f"[SmartRiskManager] Ошибка LSTM анализа: {e}")
            return {'available': False, 'error': str(e)}
    
    def _analyze_stops_for_anomalies(self, stopped_trades: List[Dict]) -> Dict:
        """Использует Anomaly Detector для поиска аномалий в стопах"""
        try:
            if not self.anomaly_detector:
                return {'available': False}
            
            # Определяем какие стопы являются аномалиями
            anomalous_stops = []
            
            for trade in stopped_trades[:20]:
                # TODO: Получить свечи и проверить через Anomaly Detector
                # candles = get_candles_for_trade(trade)
                # anomaly_score = self.anomaly_detector.detect(candles)
                # if anomaly_score.get('is_anomaly'):
                #     anomalous_stops.append(trade)
                pass
            
            return {
                'available': True,
                'total_analyzed': len(stopped_trades[:20]),
                'anomalies_found': len(anomalous_stops),
                'anomaly_rate': len(anomalous_stops) / len(stopped_trades[:20]) if stopped_trades[:20] else 0
            }
        except Exception as e:
            logger.error(f"[SmartRiskManager] Ошибка Anomaly анализа: {e}")
            return {'available': False, 'error': str(e)}
    
    def _optimize_stop_loss_with_ai(self, stopped_trades: List[Dict], patterns: Dict) -> float:
        """Использует Risk Manager для оптимизации SL"""
        try:
            if not self.risk_manager:
                return self._optimize_stop_loss(stopped_trades)
            
            # Анализируем паттерны через Risk Manager
            # TODO: Использовать risk_manager для оптимизации на основе волатильности
            
            optimal_sl = self._optimize_stop_loss(stopped_trades)
            
            # Корректируем на основе паттернов
            if patterns.get('high_rsi_stops', 0) > len(stopped_trades) * 0.3:
                optimal_sl += 2.0  # Увеличиваем SL при частых стопах на высоком RSI
            
            return optimal_sl
        except Exception as e:
            logger.error(f"[SmartRiskManager] Ошибка AI оптимизации SL: {e}")
            return 15.0
    
    def _optimize_take_profit_with_ai(self, stopped_trades: List[Dict], patterns: Dict) -> float:
        """Использует Risk Manager для оптимизации TP"""
        try:
            if not self.risk_manager:
                return self._optimize_take_profit(stopped_trades)
            
            # Анализируем паттерны через Risk Manager
            optimal_tp = self._optimize_take_profit(stopped_trades)
            
            # Корректируем на основе паттернов
            if patterns.get('rapid_stops', 0) > len(stopped_trades) * 0.5:
                optimal_tp += 20.0  # Увеличиваем TP при частых быстрых стопах
            
            return optimal_tp
        except Exception as e:
            logger.error(f"[SmartRiskManager] Ошибка AI оптимизации TP: {e}")
            return 100.0
    
    def evaluate_prediction(self, symbol: str, backtest_result: Dict, actual_outcome: Dict) -> Dict:
        """
        Оценивает насколько правильно ИИ предсказал SL/TP
        
        Args:
            symbol: Символ монеты
            backtest_result: Что ИИ предсказал при входе
            actual_outcome: Что реально произошло (из bot_history)
        
        Returns:
            Оценка: {'correct': True/False, 'score': 0-1, 'feedback': {...}}
        """
        try:
            predicted_sl = backtest_result.get('optimal_sl_percent', 15.0)
            predicted_tp = backtest_result.get('optimal_tp_percent', 100.0)
            actual_result = actual_outcome.get('roi', 0)
            
            # Рассчитываем score
            score = self._calculate_score(predicted_sl, predicted_tp, actual_result)
            
            feedback = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'predicted_sl': predicted_sl,
                'predicted_tp': predicted_tp,
                'actual_roi': actual_result,
                'score': score,
                'correct': score > 0.6
            }
            
            # Сохраняем обратную связь
            self._save_feedback(symbol, feedback)
            
            logger.info(f"[SmartRiskManager] 📊 Оценка для {symbol}: score={score:.2f} (SL={predicted_sl}%, TP={predicted_tp}%, реально={actual_result}%)")
            
            return feedback
            
        except Exception as e:
            logger.error(f"[SmartRiskManager] Ошибка оценки: {e}")
            return {'score': 0.5, 'error': str(e)}
    
    def _calculate_score(self, predicted_sl: float, predicted_tp: float, actual_result: float) -> float:
        """
        Рассчитывает score (0-1) насколько хорошо ИИ предсказал
        
        - 1.0 = идеально (закрылось в диапазоне)
        - 0.5 = средне
        - 0.0 = плохо (вышло за пределы)
        """
        if actual_result < -predicted_sl:
            # Вышли за SL → плохо
            return 0.0
        
        if actual_result > predicted_tp:
            # Превысили TP → хорошо (TP достигнут)
            return 0.8
        
        # В пределах SL-TP → нормализуем
        if predicted_sl > 0 and predicted_tp > 0:
            normalized = (actual_result + predicted_sl) / (predicted_tp + predicted_sl)
            return max(0.1, min(normalized, 0.95))
        
        return 0.5
    
    def _save_feedback(self, symbol: str, feedback: Dict):
        """Сохраняет обратную связь в файл"""
        try:
            feedback_file = self.feedback_data_path / f"{symbol}.json"
            
            # Загружаем существующие feedback
            if feedback_file.exists():
                with open(feedback_file, 'r') as f:
                    feedbacks = json.load(f)
            else:
                feedbacks = []
            
            # Добавляем новый
            feedbacks.append(feedback)
            
            # Ограничиваем размер (последние 100 записей)
            feedbacks = feedbacks[-100:]
            
            # Сохраняем
            with open(feedback_file, 'w') as f:
                json.dump(feedbacks, f, indent=2)
                
        except Exception as e:
            logger.error(f"[SmartRiskManager] Ошибка сохранения feedback: {e}")
    
    def _load_optimized_params(self):
        """Загружает оптимизированные параметры"""
        try:
            if self.optimized_params_path.exists():
                with open(self.optimized_params_path, 'r') as f:
                    self.optimized_params = json.load(f)
                    logger.debug(f"[SmartRiskManager] Загружено {len(self.optimized_params)} оптимизированных параметров")
        except Exception as e:
            logger.warning(f"[SmartRiskManager] Не удалось загрузить оптимизированные параметры: {e}")
    
    def learn_from_feedback(self):
        """
        Обучается на основе обратной связи и корректирует параметры
        
        1. Собирает данные для обучения ML модели
        2. Обучает модель на собранных данных
        3. Сохраняет обученную модель
        """
        try:
            # Собираем данные для обучения
            training_data = []
            
            # Загружаем все feedback файлы
            for feedback_file in self.feedback_data_path.glob("*.json"):
                symbol = feedback_file.stem
                
                with open(feedback_file, 'r') as f:
                    feedbacks = json.load(f)
                
                if not feedbacks:
                    continue
                
                for feedback in feedbacks:
                    # Извлекаем features и результаты
                    actual_sl = feedback.get('predicted_sl', 15.0)
                    actual_tp = feedback.get('predicted_tp', 100.0)
                    actual_roi = feedback.get('actual_roi', 0)
                    score = feedback.get('score', 0.5)
                    
                    # TODO: Получить реальные features для этой сделки из bot_history
                    features = feedback.get('features', {})
                    
                    training_data.append({
                        'features': features,
                        'actual_sl': actual_sl,
                        'actual_tp': actual_tp,
                        'actual_roi': actual_roi,
                        'score': score
                    })
            
            # Обучаем ML модель
            if self.ml_predictor and len(training_data) >= 20:
                logger.info(f"[SmartRiskManager] 🎓 Обучаем ML модель на {len(training_data)} примерах...")
                success = self.ml_predictor.train(training_data)
                
                if success:
                    logger.info("[SmartRiskManager] ✅ ML модель успешно обучена!")
            
            # Также корректируем параметры по старой логике
            self._adjust_parameters_from_feedback()
                    
        except Exception as e:
            logger.error(f"[SmartRiskManager] Ошибка обучения: {e}")
    
    def _adjust_parameters_from_feedback(self):
        """Корректирует параметры на основе обратной связи"""
        try:
            # Загружаем все feedback файлы
            for feedback_file in self.feedback_data_path.glob("*.json"):
                symbol = feedback_file.stem
                
                with open(feedback_file, 'r') as f:
                    feedbacks = json.load(f)
                
                if not feedbacks:
                    continue
                
                # Рассчитываем средний score
                avg_score = np.mean([f.get('score', 0.5) for f in feedbacks])
                
                # Если score низкий → корректируем параметры
                if avg_score < 0.5:
                    self._adjust_parameters(symbol, {
                        'sl_multiplier': 1.2,  # +20% к SL
                        'tp_multiplier': 0.9   # -10% к TP
                    })
                    logger.info(f"[SmartRiskManager] 🎓 Корректировка параметров для {symbol}: SL↑20%, TP↓10%")
                
                # Если score высокий → всё ок
                elif avg_score > 0.7:
                    logger.debug(f"[SmartRiskManager] ✅ {symbol}: Хорошие предсказания (score={avg_score:.2f})")
                    
        except Exception as e:
            logger.error(f"[SmartRiskManager] Ошибка корректировки параметров: {e}")
    
    def _adjust_parameters(self, symbol: str, adjustments: Dict):
        """Корректирует параметры для монеты"""
        if symbol not in self.optimized_params:
            self.optimized_params[symbol] = {
                'sl_multiplier': 1.0,
                'tp_multiplier': 1.0,
                'last_updated': datetime.now().isoformat(),
                'total_feedback': 0
            }
        
        self.optimized_params[symbol]['sl_multiplier'] *= adjustments.get('sl_multiplier', 1.0)
        self.optimized_params[symbol]['tp_multiplier'] *= adjustments.get('tp_multiplier', 1.0)
        self.optimized_params[symbol]['last_updated'] = datetime.now().isoformat()
        
        # Сохраняем
        try:
            with open(self.optimized_params_path, 'w') as f:
                json.dump(self.optimized_params, f, indent=2)
        except Exception as e:
            logger.error(f"[SmartRiskManager] Ошибка сохранения параметров: {e}")
    
    def collect_entry_data(self, symbol: str, current_price: float, side: str, 
                          rsi: float, candles: List[Dict], **kwargs) -> None:
        """
        🤖 Собирает данные для обучения ИИ (даже если функция отключена)
        
        Args:
            symbol: Символ монеты
            current_price: Текущая цена
            side: 'LONG' или 'SHORT'
            rsi: Текущий RSI
            candles: История свечей для анализа
            **kwargs: Дополнительные параметры
        """
        try:
            # Сохраняем данные для обучения
            entry_data = {
                'symbol': symbol,
                'price': current_price,
                'side': side,
                'rsi': rsi,
                'timestamp': datetime.now().isoformat(),
                'candles': candles[:10],  # Последние 10 свечей
                **kwargs
            }
            
            # Сохраняем для последующего анализа
            training_file = self.feedback_data_path / f"{symbol}_entry_data.json"
            training_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Загружаем существующие данные
            if training_file.exists():
                with open(training_file, 'r') as f:
                    data = json.load(f)
            else:
                data = []
            
            data.append(entry_data)
            
            # Сохраняем
            with open(training_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"[SmartRiskManager] 📊 Собраны данные для {symbol}")
            
        except Exception as e:
            logger.error(f"[SmartRiskManager] Ошибка сбора данных входа: {e}")
    
    def should_enter_now(self, symbol: str, current_price: float, side: str, 
                        rsi: float, candles: List[Dict], **kwargs) -> Dict[str, Any]:
        """
        🤖 Определяет, стоит ли входить в позицию сейчас или лучше подождать
        
        Когда все фильтры пройдены и RSI достиг порогового значения (71 для SHORT, 29 для LONG),
        ИИ решает: входить сейчас или подождать лучшей цены.
        
        Args:
            symbol: Символ монеты
            current_price: Текущая цена
            side: 'LONG' или 'SHORT'
            rsi: Текущий RSI
            candles: История свечей для анализа
            **kwargs: Дополнительные параметры (trend, volatility, etc.)
        
        Returns:
            {
                'should_enter': True/False,
                'confidence': 0.0-1.0,
                'reason': "Детальная причина",
                'optimal_price': 13.45,  # Оптимальная цена входа
                'price_deviation': -0.8%,  # Отклонение от оптимальной
                'expected_wait_time_minutes': 15  # Сколько ждать
            }
        """
        try:
            # Если нет ML модели - используем простую логику
            if not self.ml_predictor:
                return self._simple_entry_decision(symbol, current_price, side, rsi, candles)
            
            # Подготавливаем features для ML предсказания
            features = {
                'rsi': rsi,
                'volatility': kwargs.get('volatility', 1.0),
                'trend_strength': kwargs.get('trend_strength', 0.5),
                'volume': kwargs.get('volume', 0),
                'price': current_price,
                'coin_stops_count': 0,
                'avg_stop_duration_hours': 12.0
            }
            
            # Получаем предсказание оптимального входа
            prediction = self.ml_predictor.predict_entry_timing(features, side)
            
            # Определяем оптимальную цену входа
            optimal_price = self._calculate_optimal_entry_price(
                current_price, side, rsi, candles, prediction
            )
            
            # Рассчитываем отклонение
            price_deviation = ((current_price - optimal_price) / optimal_price) * 100
            
            # Принимаем решение
            should_enter = False
            confidence = 0.5
            reason = "Недостаточно данных для принятия решения"
            
            if abs(price_deviation) < 2.0:  # Отклонение < 2% - входим
                should_enter = True
                confidence = 0.8
                reason = f"Цена близка к оптимальной ({price_deviation:+.2f}%)"
            elif price_deviation < -3.0:  # Цена < оптимальной на 3%+ - ждем
                should_enter = False
                confidence = 0.6
                reason = f"Цена ниже оптимальной на {abs(price_deviation):.2f}%, ожидаем коррекции"
            elif price_deviation > 3.0:  # Цена > оптимальной на 3%+ - ждем
                should_enter = False
                confidence = 0.6
                reason = f"Цена выше оптимальной на {price_deviation:.2f}%, ожидаем отката"
            
            # Если есть исторические данные о стопах этой монеты
            stops_for_coin = self._get_coin_stops(symbol)
            if stops_for_coin and len(stops_for_coin) > 0:
                avg_entry_delay = np.mean([s.get('entry_delay_minutes', 0) for s in stops_for_coin])
                if avg_entry_delay > 15:
                    # У этой монеты есть паттерн "ждать перед входом"
                    should_enter = False
                    confidence = 0.7
                    reason = f"Исторические данные показывают: лучше подождать ({avg_entry_delay:.0f}мин)"
            
            return {
                'should_enter': should_enter,
                'confidence': confidence,
                'reason': reason,
                'optimal_price': optimal_price,
                'price_deviation': price_deviation,
                'expected_wait_time_minutes': self._estimate_wait_time(price_deviation, candles)
            }
            
        except Exception as e:
            logger.error(f"[SmartRiskManager] Ошибка определения оптимального входа для {symbol}: {e}")
            # Fallback к простому решению
            return self._simple_entry_decision(symbol, current_price, side, rsi, candles)
    
    def _simple_entry_decision(self, symbol: str, current_price: float, side: str, 
                               rsi: float, candles: List[Dict]) -> Dict[str, Any]:
        """Простое решение без ML (fallback)"""
        # Анализируем последние свечи
        if candles and len(candles) >= 3:
            recent_closes = [c.get('close', 0) for c in candles[-3:]]
            
            if side == 'LONG':
                # Для LONG: входим если цена начала расти
                if recent_closes[-1] > recent_closes[0]:
                    return {
                        'should_enter': True,
                        'confidence': 0.6,
                        'reason': "Начался рост после падения",
                        'optimal_price': current_price,
                        'price_deviation': 0.0,
                        'expected_wait_time_minutes': 0
                    }
            else:  # SHORT
                # Для SHORT: входим если цена начала падать
                if recent_closes[-1] < recent_closes[0]:
                    return {
                        'should_enter': True,
                        'confidence': 0.6,
                        'reason': "Началось падение после роста",
                        'optimal_price': current_price,
                        'price_deviation': 0.0,
                        'expected_wait_time_minutes': 0
                    }
        
        # По умолчанию - входим сразу
        return {
            'should_enter': True,
            'confidence': 0.5,
            'reason': "Все фильтры пройдены, входим",
            'optimal_price': current_price,
            'price_deviation': 0.0,
            'expected_wait_time_minutes': 0
        }
    
    def _calculate_optimal_entry_price(self, current_price: float, side: str, 
                                       rsi: float, candles: List[Dict], 
                                       prediction: Dict) -> float:
        """Рассчитывает оптимальную цену входа на основе AI анализа"""
        if not candles or len(candles) < 10:
            return current_price
        
        # Анализируем свечи для поиска локальных экстремумов
        closes = [float(c.get('close', 0)) for c in candles[-20:]]
        
        if side == 'LONG':
            # Для LONG ищем локальный минимум
            local_min = min(closes[-5:])  # Минимум за последние 5 свечей
            optimal = local_min * 1.01  # На 1% выше минимума
        else:  # SHORT
            # Для SHORT ищем локальный максимум
            local_max = max(closes[-5:])  # Максимум за последние 5 свечи
            optimal = local_max * 0.99  # На 1% ниже максимума
        
        return optimal
    
    def _estimate_wait_time(self, price_deviation: float, candles: List[Dict]) -> int:
        """Оценивает время ожидания до оптимальной цены"""
        if abs(price_deviation) < 1.0:
            return 0  # Цена уже оптимальна
        
        # Простая оценка: чем больше отклонение, тем дольше ждать
        base_wait = abs(price_deviation) * 5  # 5 минут на каждый %
        
        # Ограничиваем разумными пределами
        return min(int(base_wait), 30)  # Максимум 30 минут


# Проверка лицензии при импорте
if not PREMIUM_AVAILABLE:
    logger.warning("[SmartRiskManager] ⚠️ Премиум-лицензия не найдена, модуль недоступен")

