#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль интеграции AI. Одна и та же логика для bots.py и ai.py.

- bots.py (реальные сделки): ai_inference (signal_predictor.pkl/scaler.pkl), конфиг из auto_bot_config.
- ai.py (обучение, виртуальные сделки): get_ai_system() и trainer.predict_signal(), конфиг из bot_config.py.
Пороги (ai_min_confidence и т.д.) в шкале 0–1, без пересчёта; конфиг передаётся вызывающим кодом из одного источника.
SMC общий; решение «открывать или нет» — should_open_position_with_ai.
"""

import os
import logging
import threading
import time
from typing import Dict, Optional, Any, List
from datetime import datetime
import pandas as pd

logger = logging.getLogger('AI.Integration')


def _confidence_01(value: float) -> float:
    """Приводит уверенность к шкале 0–1. Конфиг и модель: значение как есть; если > 1 — считаем 0–100 и делим на 100 один раз."""
    if value is None:
        return 0.0
    return (float(value) / 100.0) if float(value) > 1 else float(value)


def _score_from_signal(signal: str, confidence: float) -> float:
    """Score -100..100 из signal + confidence."""
    if signal == 'LONG':
        return confidence * 100.0
    if signal == 'SHORT':
        return -confidence * 100.0
    return 0.0


def _signal_from_score(score: float) -> str:
    """LONG/SHORT/WAIT из score (пороги ±40)."""
    if score >= 40:
        return 'LONG'
    if score <= -40:
        return 'SHORT'
    return 'WAIT'


def _integrate_sentiment_onchain(symbol: str, signal: str, confidence: float) -> tuple:
    """
    Интегрирует sentiment и on-chain в сигнал (опционально по конфигу).
    Возвращает (signal, confidence, sentiment_used, onchain_used).
    """
    current = {'signal': signal, 'confidence': confidence, 'score': _score_from_signal(signal, confidence)}
    sentiment_used = onchain_used = False
    try:
        from bot_engine.ai.sentiment import integrate_sentiment_signal
        current = integrate_sentiment_signal(symbol, current)
        sentiment_used = current.get('sentiment_used', False)
    except Exception as e:
        pass
    try:
        from bot_engine.ai.onchain_analyzer import integrate_onchain_signal
        current = integrate_onchain_signal(symbol, current)
        onchain_used = current.get('onchain_used', False)
    except Exception as e:
        pass
    score = current.get('score', _score_from_signal(signal, confidence))
    out_signal = _signal_from_score(score)
    out_conf = min(1.0, abs(score) / 100.0) if score else confidence
    return (out_signal, out_conf, sentiment_used, onchain_used)


# Глобальный экземпляр AI системы
_ai_system = None
_ai_data_storage = None
_smc_features = None


def _get_ai_data_storage():
    """Ленивая инициализация AIDataStorage (может отсутствовать в некоторых сборках)."""
    global _ai_data_storage
    if _ai_data_storage is not None:
        return _ai_data_storage
    try:
        from bot_engine.ai.ai_data_storage import AIDataStorage
        _ai_data_storage = AIDataStorage()
    except Exception as exc:
        pass
        _ai_data_storage = None
    return _ai_data_storage


def get_ai_system():
    """
    Получить экземпляр AI системы. Вызывается только в процессе ai.py (обучение, виртуальные сделки).
    В bots.py для реальных сделок используется ai_inference (только pkl-модели), не get_ai_system().
    """
    global _ai_system

    if _ai_system is None:
        try:
            from ai import get_ai_system as _get_ai_system
            _ai_system = _get_ai_system()
        except Exception as e:
            pass
            return None

    return _ai_system


def get_smc_features():
    """Получить экземпляр SmartMoneyFeatures (lazy init)"""
    global _smc_features
    
    if _smc_features is None:
        try:
            from bot_engine.ai.smart_money_features import SmartMoneyFeatures
            _smc_features = SmartMoneyFeatures()
            logger.info("SmartMoneyFeatures инициализирован")
        except Exception as e:
            pass
            return None
    
    return _smc_features


def get_smc_signal(candles: List[Dict], current_price: float = None) -> Optional[Dict]:
    """
    Получить сигнал Smart Money Concepts
    
    Args:
        candles: Список свечей с OHLCV данными
        current_price: Текущая цена (опционально)
    
    Returns:
        Сигнал SMC или None
    """
    try:
        smc = get_smc_features()
        if smc is None:
            return None
        
        # Конвертируем в DataFrame
        if isinstance(candles, list):
            df = pd.DataFrame(candles)
        else:
            df = candles
        
        # Проверяем необходимые колонки
        required = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required):
            logger.warning(f"SMC: отсутствуют колонки {required}")
            return None
        
        if len(df) < 10:
            pass
            return None
        
        # Получаем комплексный сигнал
        signal = smc.get_smc_signal(df)
        
        return signal
        
    except Exception as e:
        pass
        return None


def get_smc_analysis(candles: List[Dict]) -> Optional[Dict]:
    """
    Получить детальный SMC анализ
    
    Args:
        candles: Список свечей с OHLCV данными
    
    Returns:
        Детальный анализ SMC или None
    """
    try:
        smc = get_smc_features()
        if smc is None:
            return None
        
        # Конвертируем в DataFrame
        if isinstance(candles, list):
            df = pd.DataFrame(candles)
        else:
            df = candles
        
        if len(df) < 10:
            return None
        
        current_price = df['close'].iloc[-1]
        
        # Собираем все данные SMC
        analysis = {
            'rsi': smc.get_rsi_signal(df),
            'order_blocks': smc.get_active_order_blocks(df, current_price),
            'fvg': smc.get_unfilled_fvg(df, current_price),
            'structure': smc.analyze_market_structure(df),
            'bos': smc.detect_bos(df),
            'choch': smc.detect_choch(df),
            'price_zone': smc.get_price_zone(df),
            'liquidity_zones': smc.find_liquidity_zones(df),
            'signal': smc.get_smc_signal(df)
        }
        
        return analysis
        
    except Exception as e:
        pass
        return None


def should_use_ai_prediction(symbol: str, config: Dict = None) -> bool:
    """
    Проверяет, нужно ли использовать AI предсказания
    
    Args:
        symbol: Символ монеты
        config: Конфигурация бота
    
    Returns:
        True если нужно использовать AI
    """
    try:
        # Проверяем настройку в конфиге
        if config:
            ai_enabled = config.get('ai_enabled', False)
            if not ai_enabled:
                return False
        
        # Проверяем доступность AI системы
        ai_system = get_ai_system()
        if not ai_system:
            return False
        
        # Проверяем, обучены ли модели
        if not ai_system.trainer or not ai_system.trainer.signal_predictor:
            return False
        
        return True
        
    except Exception as e:
        pass
        return False


def get_ai_prediction(symbol: str, market_data: Dict) -> Optional[Dict]:
    """
    Получить предсказание AI для символа
    
    Args:
        symbol: Символ монеты
        market_data: Рыночные данные (RSI, тренд, цена и т.д.)
    
    Returns:
        Предсказание AI или None
    """
    try:
        ai_system = get_ai_system()
        if not ai_system:
            return None
        
        prediction = ai_system.predict_signal(symbol, market_data)
        
        if 'error' in prediction:
            return None
        
        return prediction
        
    except Exception as e:
        pass
        return None


def apply_ai_prediction_to_signal(
    symbol: str,
    original_signal: str,
    market_data: Dict,
    config: Dict = None
) -> Dict:
    """
    Применяет предсказание AI к оригинальному сигналу
    
    Args:
        symbol: Символ монеты
        original_signal: Оригинальный сигнал (LONG/SHORT/WAIT)
        market_data: Рыночные данные
        config: Конфигурация бота
    
    Returns:
        Словарь с результирующим сигналом и информацией об AI
    """
    try:
        # Проверяем, нужно ли использовать AI
        if not should_use_ai_prediction(symbol, config):
            return {
                'signal': original_signal,
                'ai_used': False,
                'reason': 'AI disabled or not available'
            }
        
        # Получаем предсказание AI
        ai_prediction = get_ai_prediction(symbol, market_data)
        
        if not ai_prediction:
            return {
                'signal': original_signal,
                'ai_used': False,
                'reason': 'AI prediction not available'
            }
        
        ai_signal = ai_prediction.get('signal', 'WAIT')
        ai_confidence = ai_prediction.get('confidence', 0)
        ai_signal, ai_confidence, sentiment_used, onchain_used = _integrate_sentiment_onchain(symbol, ai_signal, ai_confidence)

        min_confidence = _confidence_01(config.get('ai_min_confidence', 0.7) if config else 0.7)
        ai_conf_01 = _confidence_01(ai_confidence)

        if ai_conf_01 >= min_confidence:
            return {
                'signal': ai_signal,
                'ai_used': True,
                'ai_confidence': ai_conf_01,
                'ai_prediction': ai_prediction,
                'original_signal': original_signal,
                'sentiment_used': sentiment_used,
                'onchain_used': onchain_used,
                'reason': f'AI signal used (confidence: {ai_conf_01:.2%})'
            }

        return {
            'signal': original_signal,
            'ai_used': True,
            'ai_confidence': ai_conf_01,
            'ai_prediction': ai_prediction,
            'sentiment_used': sentiment_used,
            'onchain_used': onchain_used,
            'reason': f'Original signal used (AI confidence too low: {ai_conf_01:.2%})'
        }
        
    except Exception as e:
        logger.error(f"Ошибка применения AI предсказания для {symbol}: {e}")
        return {
            'signal': original_signal,
            'ai_used': False,
            'error': str(e)
        }


def get_optimized_bot_config(symbol: str) -> Optional[Dict]:
    """
    Получить оптимизированную конфигурацию бота от AI
    
    Args:
        symbol: Символ монеты
    
    Returns:
        Оптимизированная конфигурация или None
    """
    try:
        ai_system = get_ai_system()
        if not ai_system:
            return None
        
        optimized = ai_system.optimize_bot_config(symbol)
        
        if 'error' in optimized:
            return None
        
        return optimized
        
    except Exception as e:
        pass
        return None


def _is_ai_process() -> bool:
    """True, если код выполняется в процессе ai.py (обучение, виртуальные сделки)."""
    return os.environ.get('INFOBOT_AI_PROCESS', '').strip().lower() in ('1', 'true', 'yes')


def _smc_enabled_from_config() -> bool:
    """Читает AI_SMC_ENABLED из configs/bot_config.py с диска (источник истины)."""
    try:
        import bot_engine.config_loader as _bc
        _root = os.path.dirname(os.path.dirname(os.path.abspath(getattr(_bc, '__file__', __file__))))
        _path = os.path.join(_root, 'configs', 'bot_config.py')
        if _path and os.path.isfile(_path):
            with open(_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if 'AI_SMC_ENABLED' in line and '=' in line:
                        val = line.split('=', 1)[-1].strip().upper()
                        return 'FALSE' not in val or val.startswith('TRUE')
    except Exception:
        pass
    try:
        from bot_engine.config_loader import AIConfig
        return getattr(AIConfig, 'AI_SMC_ENABLED', True)
    except Exception:
        pass
    return True


def should_open_position_with_ai(
    symbol: str,
    direction: str,
    rsi: float,
    trend: str,
    price: float,
    config: Dict = None,
    candles: List[Dict] = None
) -> Dict:
    """
    Проверяет, нужно ли открывать позицию с учётом AI и SMC.
    Перед решением смотрит аналитику (ошибки, неудачные монеты) и корректирует поведение.
    """
    try:
        # Аналитика: предварительная проверка (блокировка явно неудачных монет/диапазонов)
        try:
            from bot_engine.ai_analytics import apply_analytics_to_entry_decision
            pre_allowed, _, pre_reason = apply_analytics_to_entry_decision(
                symbol, direction, rsi, trend, base_allowed=True, base_confidence=1.0, base_reason=""
            )
            if not pre_allowed:
                return {
                    'should_open': False, 'ai_used': True, 'smc_used': False,
                    'reason': pre_reason, 'ai_confidence': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as _ax:
            logger.debug("apply_analytics (pre): %s", _ax)
        result = {
            'should_open': True,
            'ai_used': False,
            'smc_used': False,
            'sentiment_used': False,
            'onchain_used': False,
            'reason': 'Default allow',
            'timestamp': datetime.now().isoformat()
        }
        
        # === SMC АНАЛИЗ (если включён и есть свечи) ===
        smc_signal = None
        smc_enabled = _smc_enabled_from_config()
        if smc_enabled and candles and len(candles) >= 10:
            smc_signal = get_smc_signal(candles, price)
            
            if smc_signal:
                result['smc_used'] = True
                result['smc_signal'] = smc_signal['signal']
                result['smc_score'] = smc_signal['score']
                result['smc_confidence'] = smc_signal['confidence']
                result['smc_reasons'] = smc_signal.get('reasons', [])
                result['smc_entry_zone'] = smc_signal.get('entry_zone')
                
                # SMC порог для использования сигнала
                smc_threshold = config.get('smc_min_score', 40) if config else 40
                
                # Проверяем согласованность SMC с направлением
                if direction == 'LONG':
                    if smc_signal['signal'] == 'LONG' and smc_signal['score'] >= smc_threshold:
                        result['reason'] = f"SMC подтверждает LONG (score: {smc_signal['score']})"
                    elif smc_signal['signal'] == 'SHORT' and smc_signal['score'] <= -smc_threshold:
                        result['should_open'] = False
                        result['reason'] = f"SMC против LONG (score: {smc_signal['score']})"
                        return result
                        
                elif direction == 'SHORT':
                    if smc_signal['signal'] == 'SHORT' and smc_signal['score'] <= -smc_threshold:
                        result['reason'] = f"SMC подтверждает SHORT (score: {smc_signal['score']})"
                    elif smc_signal['signal'] == 'LONG' and smc_signal['score'] >= smc_threshold:
                        result['should_open'] = False
                        result['reason'] = f"SMC против SHORT (score: {smc_signal['score']})"
                        return result
        
        # Подготавливаем рыночные данные
        market_data = {
            'rsi': rsi,
            'trend': trend,
            'price': price,
            'direction': direction
        }
        if smc_signal:
            market_data['smc_signal'] = smc_signal['signal']
            market_data['smc_score'] = smc_signal['score']
            market_data['smc_confidence'] = smc_signal['confidence']
        
        # === ПРЕДСКАЗАНИЕ: два разных кода ===
        # В ai.py — полный стек (trainer, обучение, виртуальные сделки)
        # В bots.py — только инференс по сохранённым моделям (ai_inference)
        if _is_ai_process():
            ai_system = get_ai_system()
            if not ai_system or not ai_system.trainer or not ai_system.trainer.signal_predictor:
                if smc_signal:
                    return result
                return {'should_open': True, 'ai_used': False, 'smc_used': result.get('smc_used', False), 'reason': 'AI models not trained yet'}
            prediction = ai_system.predict_signal(symbol, market_data)
        else:
            from bot_engine.ai.ai_inference import predict_signal as inference_predict_signal
            prediction = inference_predict_signal(symbol, market_data)
        
        if 'error' in prediction:
            pass
            if smc_signal:
                return result  # Используем только SMC
            return {'should_open': True, 'ai_used': False, 'smc_used': result.get('smc_used', False), 'reason': f"AI prediction error: {prediction.get('error')}"}
        
        signal = prediction.get('signal')
        confidence = prediction.get('confidence', 0)
        signal, confidence, sentiment_used, onchain_used = _integrate_sentiment_onchain(symbol, signal, confidence)
        ai_conf_01 = _confidence_01(confidence)
        result['ai_used'] = True
        result['ai_signal'] = signal
        result['ai_confidence'] = ai_conf_01
        result['sentiment_used'] = sentiment_used
        result['onchain_used'] = onchain_used

        ai_confidence_threshold = _confidence_01(config.get('ai_min_confidence', 0.65) if config else 0.65)

        # === КОМБИНИРОВАННАЯ ЛОГИКА AI + SMC ===
        should_open = False

        # Если и AI и SMC согласны - высокая уверенность
        if smc_signal:
            ai_agrees = (direction == signal and ai_conf_01 >= ai_confidence_threshold)
            smc_agrees = (
                (direction == 'LONG' and smc_signal['signal'] == 'LONG') or
                (direction == 'SHORT' and smc_signal['signal'] == 'SHORT')
            )
            
            if ai_agrees and smc_agrees:
                should_open = True
                result['reason'] = f"AI + SMC подтверждают {direction}"
            elif smc_agrees and not ai_agrees:
                # SMC согласен, AI нет - используем SMC (приоритет SMC)
                if abs(smc_signal['score']) >= 50:
                    should_open = True
                    result['reason'] = f"SMC подтверждает {direction} (AI не согласен)"
                else:
                    should_open = False
                    result['reason'] = f"AI не подтверждает {direction}, SMC слабый"
            elif ai_agrees and not smc_agrees:
                # AI согласен, SMC нет - блокируем (SMC имеет приоритет)
                if smc_signal['signal'] == 'WAIT':
                    should_open = True
                    result['reason'] = f"AI подтверждает {direction}, SMC нейтрален"
                else:
                    should_open = False
                    result['reason'] = f"SMC против {direction}"
            else:
                should_open = False
                result['reason'] = f"Ни AI ни SMC не подтверждают {direction}"
        else:
            # Только AI (нет свечей для SMC)
            if direction == 'LONG' and signal == 'LONG' and ai_conf_01 >= ai_confidence_threshold:
                should_open = True
                result['reason'] = f"AI подтверждает LONG (confidence: {ai_conf_01:.2%})"
            elif direction == 'SHORT' and signal == 'SHORT' and ai_conf_01 >= ai_confidence_threshold:
                should_open = True
                result['reason'] = f"AI подтверждает SHORT (confidence: {ai_conf_01:.2%})"
            elif signal == 'WAIT':
                should_open = False
                result['reason'] = f"AI рекомендует WAIT"
            elif ai_conf_01 < ai_confidence_threshold:
                should_open = False
                result['reason'] = f"AI confidence too low: {ai_conf_01:.2%}"

        # Применяем аналитику к итоговому решению
        try:
            from bot_engine.ai_analytics import apply_analytics_to_entry_decision
            ai_conf = result.get('ai_confidence', 0.5)
            should_open, ai_conf, reason_adj = apply_analytics_to_entry_decision(
                symbol, direction, rsi, trend,
                base_allowed=should_open, base_confidence=ai_conf,
                base_reason=result.get('reason', ''),
            )
            result['ai_confidence'] = ai_conf
            result['reason'] = reason_adj
            ai_confidence_threshold = _confidence_01(config.get('ai_min_confidence', 0.65) if config else 0.65)
            if should_open and ai_conf < ai_confidence_threshold:
                should_open = False
                result['reason'] = reason_adj + f" | Уверенность {ai_conf:.2%} < порога"
        except Exception as _ax2:
            logger.debug("apply_analytics (post): %s", _ax2)

        result['should_open'] = should_open
        result['model_used'] = 'signal_predictor.pkl + SMC' if smc_signal else 'signal_predictor.pkl'

        # ВАЖНО: Сохраняем решение AI для отслеживания результатов торговли
        if should_open:
            try:
                result['ai_decision_id'] = _track_ai_decision(
                    symbol, direction, rsi, trend, price, signal, ai_conf_01, market_data
                )
            except Exception as e:
                pass
        
        return result
        
    except Exception as e:
        logger.error(f"Ошибка при получении AI/SMC предсказания для {symbol}: {e}")
        import traceback
        pass
        return {'should_open': True, 'ai_used': False, 'smc_used': False, 'reason': f'AI/SMC error: {e}'}

# Глобальная функция для отслеживания решений AI
_ai_decisions_tracking = {}
_ai_decisions_lock = threading.Lock()

def _track_ai_decision(symbol: str, direction: str, rsi: float, trend: str,
                       price: float, ai_signal: str, ai_confidence: float,
                       market_data: Dict) -> str:
    """Отслеживает решение AI для последующего анализа"""
    try:
        decision_id = f"ai_{symbol}_{int(time.time() * 1000)}"
        decision_payload = {
            'id': decision_id,
            'symbol': symbol,
            'direction': direction,
            'rsi': rsi,
            'trend': trend,
            'price': price,
            'ai_signal': ai_signal,
            'ai_confidence': ai_confidence,
            'market_data': market_data.copy(),
            'timestamp': datetime.now().isoformat(),
            'status': 'PENDING'
        }
        
        with _ai_decisions_lock:
            _ai_decisions_tracking[decision_id] = decision_payload

        storage = _get_ai_data_storage()
        if storage:
            try:
                storage.save_ai_decision(decision_id, decision_payload)
            except Exception as storage_error:
                pass
        
        return decision_id
    except:
        return None

def get_ai_decision(decision_id: str) -> Optional[Dict]:
    """Получить решение AI по ID"""
    with _ai_decisions_lock:
        return _ai_decisions_tracking.get(decision_id)

def update_ai_decision_result(decision_id: str, pnl: float, roi: float, is_successful: bool):
    """Обновить результат решения AI после закрытия сделки"""
    try:
        with _ai_decisions_lock:
            if decision_id in _ai_decisions_tracking:
                _ai_decisions_tracking[decision_id]['status'] = 'SUCCESS' if is_successful else 'FAILED'
                _ai_decisions_tracking[decision_id]['pnl'] = pnl
                _ai_decisions_tracking[decision_id]['roi'] = roi
                _ai_decisions_tracking[decision_id]['closed_at'] = datetime.now().isoformat()

                # Сохраняем в файл через тренер для последующего переобучения
                try:
                    ai_system = get_ai_system()
                    if ai_system and ai_system.trainer:
                        ai_system.trainer.update_ai_decision_result(
                            decision_id, pnl, roi, is_successful, {'exit_data': 'from_bot_class'}
                        )
                except Exception as save_error:
                    pass

                # НОВОЕ: Отправляем сделку в систему самообучения
                try:
                    from bot_engine.ai.ai_self_learning import process_trade_for_self_learning
                    trade_result = dict(_ai_decisions_tracking[decision_id])
                    trade_result.update({
                        'pnl': pnl,
                        'roi': roi,
                        'is_successful': is_successful
                    })
                    process_trade_for_self_learning(trade_result)
                    pass
                except Exception as self_learning_error:
                    pass

        storage = _get_ai_data_storage()
        if storage:
            try:
                storage.update_ai_decision(decision_id, {
                    'status': 'SUCCESS' if is_successful else 'FAILED',
                    'pnl': float(pnl) if pnl is not None else None,
                    'roi': float(roi) if roi is not None else None,
                    'updated_at': datetime.now().isoformat(),
                    'closed_at': datetime.now().isoformat()
                })
            except Exception as storage_error:
                pass
    except Exception as e:
        pass


def get_ai_entry_decision(
    symbol: str,
    direction: str,
    candles: List[Dict],
    current_price: float,
    prii_config: Dict = None,
    coin_params: Dict = None,
    rsi: float = None,
    trend: str = None,
) -> Dict[str, Any]:
    """
    Решение ИИ о входе в позицию (FullAI). Используется при full_ai_control.
    Перед решением смотрит аналитику (ошибки, неудачные монеты) и корректирует поведение.
    Возвращает allowed, confidence, reason.
    """
    prii_config = prii_config or {}
    coin_params = coin_params or {}
    min_conf = _confidence_01(prii_config.get('ai_min_confidence', 0.7))
    result = {'allowed': False, 'confidence': 0.0, 'reason': 'AI not available'}
    try:
        # FullAI: модель — основной решатель. Не блокируем до вызова ИИ; аналитика только снижает уверенность.
        from bot_engine.ai import get_ai_manager
        ai_manager = get_ai_manager()
        if not ai_manager or not ai_manager.is_available():
            result['reason'] = 'AI modules or license not available'
            return result
        coin_data = {'current_price': current_price}
        analysis = ai_manager.analyze_coin(symbol, coin_data, candles or [])
        if not analysis.get('available'):
            result['reason'] = analysis.get('reason', 'AI analysis failed')
            return result
        # Голосование: LSTM, pattern, anomaly
        votes_long = 0.0
        votes_short = 0.0
        if analysis.get('lstm_prediction'):
            lp = analysis['lstm_prediction']
            conf = _confidence_01(lp.get('confidence', 0))
            if conf >= min_conf:
                if lp.get('direction', 0) > 0:
                    votes_long += conf
                elif lp.get('direction', 0) < 0:
                    votes_short += conf
        if analysis.get('pattern_analysis') and analysis['pattern_analysis'].get('patterns'):
            pa = analysis['pattern_analysis']
            conf = _confidence_01(pa.get('confidence', 0))
            if conf >= min_conf:
                sig = (pa.get('signal') or '').upper()
                if sig == 'BULLISH':
                    votes_long += conf
                elif sig == 'BEARISH':
                    votes_short += conf
        if analysis.get('anomaly_score') and analysis['anomaly_score'].get('is_anomaly'):
            result['allowed'] = False
            result['reason'] = 'Anomaly detected (block entry)'
            result['confidence'] = 0.0
            return result
        if direction == 'LONG':
            result['allowed'] = votes_long >= min_conf and votes_long >= votes_short
            result['confidence'] = votes_long
        else:
            result['allowed'] = votes_short >= min_conf and votes_short >= votes_long
            result['confidence'] = votes_short
        result['reason'] = f"AI vote LONG={votes_long:.2f} SHORT={votes_short:.2f}"
        # FullAI: аналитика только снижает уверенность (признаки/веса), не блокирует — решение модели приоритет
        try:
            from bot_engine.ai_analytics import apply_analytics_to_entry_decision
            result['allowed'], result['confidence'], result['reason'] = apply_analytics_to_entry_decision(
                symbol, direction, rsi, trend,
                base_allowed=result['allowed'], base_confidence=result['confidence'], base_reason=result['reason'],
                full_ai_mode=True,
            )
            if result['confidence'] < min_conf and result['allowed']:
                result['allowed'] = False
                result['reason'] = result['reason'] + f" | Уверенность {result['confidence']:.2%} < порога {min_conf:.2%}"
        except Exception as _ax2:
            logger.debug("apply_analytics (post): %s", _ax2)
        return result
    except Exception as e:
        logger.exception(f"get_ai_entry_decision {symbol} {direction}: {e}")
        result['reason'] = str(e)
        return result


def get_ai_exit_decision(
    symbol: str,
    position: Dict,
    candles: List[Dict],
    pnl_percent: float,
    prii_config: Dict = None,
    coin_params: Dict = None,
    data_context: Dict = None,
) -> Dict[str, Any]:
    """
    Решение ИИ о закрытии позиции сейчас (FullAI). При full_ai_control решение о выходе принимает только ИИ.
    data_context: полный контекст от get_fullai_data_context (свечи из БД, system: RSI/тренд/сигнал, custom индикаторы).
    Возвращает close_now, reason, confidence.
    """
    prii_config = prii_config or {}
    coin_params = coin_params or {}
    result = {'close_now': False, 'reason': 'Hold', 'confidence': 0.0}
    try:
        # Простая эвристика: сильная прибыль или сильный убыток — закрыть (далее можно заменить на модель)
        tp = float(prii_config.get('take_profit_percent') or coin_params.get('take_profit_percent') or 15)
        sl = float(prii_config.get('max_loss_percent') or coin_params.get('max_loss_percent') or 10)
        if pnl_percent >= tp:
            result['close_now'] = True
            result['reason'] = f'Take profit ({pnl_percent:.2f}% >= {tp}%)'
            result['confidence'] = 1.0
            return result
        if pnl_percent <= -sl:
            result['close_now'] = True
            result['reason'] = f'Stop loss ({pnl_percent:.2f}% <= -{sl}%)'
            result['confidence'] = 1.0
            return result
        # Используем data_context (свечи из БД, индикаторы) если передан
        if not candles and data_context:
            candles = data_context.get('candles') or []
        # Опционально: вызов AIManager для более сложного решения (LSTM/pattern на выход)
        from bot_engine.ai import get_ai_manager
        ai_manager = get_ai_manager()
        if ai_manager and ai_manager.is_available() and candles and len(candles) >= 5:
            coin_data = {'current_price': candles[-1].get('close'), 'in_position': True}
            analysis = ai_manager.analyze_coin(symbol, coin_data, candles)
            if analysis.get('anomaly_score', {}).get('is_anomaly'):
                result['close_now'] = True
                result['reason'] = 'Anomaly: exit'
                result['confidence'] = 0.9
        return result
    except Exception as e:
        logger.exception(f"get_ai_exit_decision {symbol}: {e}")
        return result

