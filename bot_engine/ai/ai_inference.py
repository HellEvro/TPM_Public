#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Инференс для РЕАЛЬНЫХ сделок (используется в bots.py).

Отдельный код от ai.py: здесь только загрузка сохранённых моделей и предсказание.
Никакого обучения, виртуальных сделок, trainer — только signal_predictor.pkl + scaler.pkl.
Модели обучаются и сохраняются в ai.py; bots читают их с диска и предсказывают.
"""

import os
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger('AI.Inference')

# Лениво загружаемые артефакты (без импорта ai.py / trainer)
_signal_predictor = None
_scaler = None
_expected_features = None
_models_dir = None


def _get_models_dir() -> str:
    """Путь к data/ai/models относительно корня проекта."""
    global _models_dir
    if _models_dir is not None:
        return _models_dir
    try:
        from pathlib import Path
        current = Path(__file__).resolve()
        for parent in [current.parent.parent.parent] + list(current.parents):
            if parent and (parent / 'bot_engine').exists():
                d = parent / 'data' / 'ai' / 'models'
                if d.exists() or (parent / 'data').exists():
                    _models_dir = str(d)
                    return _models_dir
        _models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'ai', 'models')
    except Exception:
        _models_dir = 'data/ai/models'
    return _models_dir


def _load_models() -> bool:
    """Загружает signal_predictor.pkl и scaler.pkl. Возвращает True при успехе."""
    global _signal_predictor, _scaler, _expected_features
    if _signal_predictor is not None and _scaler is not None:
        return True
    try:
        import joblib
        base = _get_models_dir()
        signal_path = os.path.normpath(os.path.join(base, 'signal_predictor.pkl'))
        scaler_path = os.path.normpath(os.path.join(base, 'scaler.pkl'))
        if not os.path.exists(signal_path) or not os.path.exists(scaler_path):
            return False
        _signal_predictor = joblib.load(signal_path)
        _scaler = joblib.load(scaler_path)
        if hasattr(_scaler, 'n_features_in_') and _scaler.n_features_in_ is not None:
            _expected_features = _scaler.n_features_in_
        elif hasattr(_scaler, 'mean_') and _scaler.mean_ is not None:
            _expected_features = len(_scaler.mean_)
        elif hasattr(_scaler, 'scale_') and _scaler.scale_ is not None:
            _expected_features = len(_scaler.scale_)
        else:
            _expected_features = 7
        return True
    except Exception as e:
        logger.warning(f"ai_inference: не удалось загрузить модели: {e}")
        return False


def build_features(market_data: Dict) -> list:
    """
    Вектор из 7 признаков в том же порядке, что и ai_trainer._build_signal_features_7.
    Порядок: rsi, volatility, volume_ratio, trend_UP, trend_DOWN, direction_LONG, price/1000.
    Совпадение обязательно — иначе предсказания бессмысленны.
    """
    rsi = market_data.get('rsi', 50)
    trend = market_data.get('trend', 'NEUTRAL')
    price = market_data.get('price', 0)
    direction = market_data.get('direction', 'LONG')
    volatility = market_data.get('volatility', 0)
    volume_ratio = market_data.get('volume_ratio', 1.0)
    features = [
        rsi,
        volatility,
        volume_ratio,
        1.0 if trend == 'UP' else 0.0,
        1.0 if trend == 'DOWN' else 0.0,
        1.0 if direction == 'LONG' else 0.0,
        price / 1000.0 if price > 0 else 0,
    ]
    n = _expected_features if _expected_features is not None else 7
    if len(features) < n:
        while len(features) < n:
            features.append(0.0)
    elif len(features) > n:
        features = features[:n]
    return features


def predict_signal(symbol: str, market_data: Dict) -> Dict[str, Any]:
    """
    Предсказание сигнала для реальной сделки (только инференс, без ai.py/trainer).
    Использует сохранённые signal_predictor.pkl и scaler.pkl из data/ai/models.
    Возвращает {'signal': 'LONG'|'SHORT'|'WAIT', 'confidence': float, ...} или {'error': str}.
    """
    if not _load_models():
        return {'error': 'Models not loaded'}
    if not hasattr(_signal_predictor, 'predict_proba'):
        return {'error': 'signal_predictor does not support predict_proba'}
    try:
        import numpy as np
        features = build_features(market_data)
        features_array = np.array([features])
        features_scaled = _scaler.transform(features_array)
        signal_prob = _signal_predictor.predict_proba(features_scaled)[0]
        rsi = market_data.get('rsi', 50)
        # Интерпретация как в ai_trainer.predict()
        if len(signal_prob) < 2:
            return {'signal': 'WAIT', 'confidence': 0.0, 'error': 'Invalid proba shape'}
        prob_profit = float(signal_prob[1])
        if prob_profit > 0.6:
            signal = 'LONG' if rsi < 35 else ('SHORT' if rsi > 65 else 'WAIT')
        else:
            signal = 'WAIT'
        return {
            'signal': signal,
            'confidence': prob_profit,
            'rsi': rsi,
            'trend': market_data.get('trend', 'NEUTRAL'),
        }
    except Exception as e:
        logger.warning(f"ai_inference predict_signal: {e}")
        return {'error': str(e)}
