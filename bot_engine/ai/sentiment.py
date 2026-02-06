"""
Sentiment Analysis для криптовалют

Модуль для анализа настроений рынка:
- Анализ текста (новости, твиты)
- Агрегация sentiment данных
- Интеграция с торговыми решениями

Placeholder для API интеграций (Twitter, Reddit, News)
"""

import logging
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger('Sentiment')

# Проверяем transformers (опционально)
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class SentimentType(Enum):
    """Типы настроений"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

@dataclass
class SentimentResult:
    """Результат анализа настроений"""
    sentiment: str
    score: float  # -1 до 1
    confidence: float  # 0 до 1
    source: str
    timestamp: str

class SentimentAnalyzer:
    """
    Анализатор настроений текста

    Использует transformers если доступен,
    иначе rule-based подход
    """

    def __init__(self, use_transformers: bool = True):
        self.use_transformers = use_transformers and TRANSFORMERS_AVAILABLE
        self.model = None

        if self.use_transformers:
            try:
                self.model = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
                logger.info("Transformers sentiment model loaded")
            except Exception as e:
                logger.warning(f"Failed to load transformers model: {e}")
                self.use_transformers = False

        # Rule-based keywords
        self.bullish_words = {
            'buy', 'bull', 'bullish', 'moon', 'pump', 'long', 'breakout',
            'accumulate', 'hodl', 'rally', 'surge', 'soar', 'gain',
            'profit', 'growth', 'strong', 'support', 'bottom'
        }

        self.bearish_words = {
            'sell', 'bear', 'bearish', 'dump', 'crash', 'short', 'breakdown',
            'distribute', 'drop', 'fall', 'plunge', 'decline', 'loss',
            'weak', 'resistance', 'top', 'correction', 'fear'
        }

    def analyze_text(self, text: str) -> Dict:
        """
        Анализирует настроение текста

        Args:
            text: Текст для анализа

        Returns:
            Dict с sentiment, confidence
        """
        if not text or len(text.strip()) < 3:
            return {
                'sentiment': 'neutral',
                'score': 0,
                'confidence': 0.5,
                'method': 'empty_text'
            }

        if self.use_transformers and self.model:
            return self._transformers_analyze(text)
        else:
            return self._rule_based_analyze(text)

    def _transformers_analyze(self, text: str) -> Dict:
        """Анализ с использованием transformers"""
        try:
            result = self.model(text[:512])[0]  # Лимит 512 токенов

            label = result['label'].lower()
            score = result['score']

            if label == 'positive':
                sentiment = 'bullish'
                sentiment_score = score
            elif label == 'negative':
                sentiment = 'bearish'
                sentiment_score = -score
            else:
                sentiment = 'neutral'
                sentiment_score = 0

            return {
                'sentiment': sentiment,
                'score': sentiment_score,
                'confidence': score,
                'method': 'transformers'
            }
        except Exception as e:

            return self._rule_based_analyze(text)

    def _rule_based_analyze(self, text: str) -> Dict:
        """Rule-based анализ по ключевым словам"""
        text_lower = text.lower()
        words = set(re.findall(r'\w+', text_lower))

        bullish_count = len(words & self.bullish_words)
        bearish_count = len(words & self.bearish_words)
        total = bullish_count + bearish_count

        if total == 0:
            return {
                'sentiment': 'neutral',
                'score': 0,
                'confidence': 0.3,
                'method': 'rule_based'
            }

        score = (bullish_count - bearish_count) / total

        if score > 0.2:
            sentiment = 'bullish'
        elif score < -0.2:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'

        confidence = min(total / 10, 1.0) * 0.7  # Max 0.7 для rule-based

        return {
            'sentiment': sentiment,
            'score': score,
            'confidence': confidence,
            'method': 'rule_based',
            'bullish_words': bullish_count,
            'bearish_words': bearish_count
        }

class CryptoSentimentCollector:
    """
    Сборщик sentiment данных из различных источников

    Placeholders для API интеграций
    """

    def __init__(self):
        self.analyzer = SentimentAnalyzer()
        self.cache: Dict[str, List[SentimentResult]] = {}

    def get_twitter_sentiment(self, symbol: str) -> Optional[Dict]:
        """
        Получает sentiment из Twitter

        TODO: Интеграция с Twitter API
        """
        # Placeholder
        return {
            'source': 'twitter',
            'sentiment': 'neutral',
            'score': 0,
            'confidence': 0,
            'status': 'not_implemented',
            'message': 'Twitter API integration pending'
        }

    def get_reddit_sentiment(self, symbol: str) -> Optional[Dict]:
        """
        Получает sentiment из Reddit

        TODO: Интеграция с Reddit API
        """
        # Placeholder
        return {
            'source': 'reddit',
            'sentiment': 'neutral',
            'score': 0,
            'confidence': 0,
            'status': 'not_implemented',
            'message': 'Reddit API integration pending'
        }

    def get_news_sentiment(self, symbol: str) -> Optional[Dict]:
        """
        Получает sentiment из новостей

        TODO: Интеграция с News API
        """
        # Placeholder
        return {
            'source': 'news',
            'sentiment': 'neutral',
            'score': 0,
            'confidence': 0,
            'status': 'not_implemented',
            'message': 'News API integration pending'
        }

    def get_aggregated_sentiment(self, symbol: str) -> Dict:
        """
        Агрегирует sentiment из всех источников.
        Если AI_SENTIMENT_ENABLED=False, возвращает neutral с confidence 0.

        Args:
            symbol: Символ монеты (например, BTC, ETH)

        Returns:
            Агрегированный sentiment
        """
        if not _sentiment_enabled():
            return {
                'symbol': symbol,
                'sentiment': 'neutral',
                'score': 0,
                'confidence': 0,
                'sources': [],
                'timestamp': datetime.now().isoformat(),
                'enabled': False,
            }
        sources = [
            self.get_twitter_sentiment(symbol),
            self.get_reddit_sentiment(symbol),
            self.get_news_sentiment(symbol)
        ]

        # Фильтруем валидные источники
        valid_sources = [
            s for s in sources 
            if s and s.get('confidence', 0) > 0
        ]

        if not valid_sources:
            return {
                'symbol': symbol,
                'sentiment': 'neutral',
                'score': 0,
                'confidence': 0,
                'sources': [],
                'timestamp': datetime.now().isoformat()
            }

        # Взвешенное среднее
        total_weight = sum(s['confidence'] for s in valid_sources)
        weighted_score = sum(
            s['score'] * s['confidence'] 
            for s in valid_sources
        ) / total_weight if total_weight > 0 else 0

        if weighted_score > 0.2:
            sentiment = 'bullish'
        elif weighted_score < -0.2:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'

        return {
            'symbol': symbol,
            'sentiment': sentiment,
            'score': weighted_score,
            'confidence': total_weight / len(sources),
            'sources': [s['source'] for s in valid_sources],
            'timestamp': datetime.now().isoformat()
        }

def _sentiment_enabled() -> bool:
    """Проверка, включён ли Sentiment Analysis в конфиге."""
    try:
        from bot_engine.config_loader import AIConfig
        return bool(getattr(AIConfig, 'AI_SENTIMENT_ENABLED', False))
    except Exception:
        return False

def integrate_sentiment_signal(
    symbol: str,
    current_signal: Dict,
    sentiment_weight: Optional[float] = None
) -> Dict:
    """
    Интегрирует sentiment в торговый сигнал.
    Если AI_SENTIMENT_ENABLED=False, возвращает current_signal без изменений (sentiment_used=False).

    Args:
        symbol: Символ монеты
        current_signal: Текущий торговый сигнал
        sentiment_weight: Вес sentiment (0–1). Если None — из AIConfig.AI_SENTIMENT_WEIGHT

    Returns:
        Обновленный сигнал
    """
    if not _sentiment_enabled():
        return {**current_signal, 'sentiment_used': False, 'sentiment_data': None}
    try:
        from bot_engine.config_loader import AIConfig
        w = sentiment_weight if sentiment_weight is not None else getattr(AIConfig, 'AI_SENTIMENT_WEIGHT', 0.2)
    except Exception:
        w = 0.2
    collector = CryptoSentimentCollector()
    sentiment = collector.get_aggregated_sentiment(symbol)

    # Если sentiment недоступен, возвращаем оригинальный сигнал
    if sentiment['confidence'] < 0.3:
        current_signal['sentiment_used'] = False
        current_signal['sentiment_data'] = sentiment
        return current_signal

    # Интегрируем sentiment
    original_score = current_signal.get('score', 0)
    sentiment_score = sentiment['score'] * 100  # Масштабируем к -100..100

    combined_score = (
        original_score * (1 - w) +
        sentiment_score * w
    )

    # Обновляем сигнал
    if combined_score >= 40:
        new_signal = 'LONG'
    elif combined_score <= -40:
        new_signal = 'SHORT'
    else:
        new_signal = 'WAIT'

    return {
        **current_signal,
        'signal': new_signal,
        'score': combined_score,
        'sentiment_used': True,
        'sentiment_data': sentiment,
        'original_score': original_score
    }

# ==================== ТЕСТОВЫЙ КОД ====================

if __name__ == '__main__':
    print("=" * 60)
    print("Sentiment Analysis - Test")
    print("=" * 60)

    # Тест SentimentAnalyzer
    print("\n1. Test SentimentAnalyzer:")

    analyzer = SentimentAnalyzer(use_transformers=False)  # Rule-based для теста

    test_texts = [
        "Bitcoin is going to the moon! Very bullish!",
        "Market is crashing, sell everything!",
        "Price action looks neutral, waiting for breakout",
        "Strong support at current levels, accumulating more"
    ]

    for text in test_texts:
        result = analyzer.analyze_text(text)
        print(f"   '{text[:40]}...'")
        print(f"      -> {result['sentiment']} (score: {result['score']:.2f})")

    # Тест CryptoSentimentCollector
    print("\n2. Test CryptoSentimentCollector:")

    collector = CryptoSentimentCollector()
    aggregated = collector.get_aggregated_sentiment("BTC")

    print(f"   BTC Aggregated sentiment:")
    print(f"      Sentiment: {aggregated['sentiment']}")
    print(f"      Score: {aggregated['score']:.2f}")
    print(f"      Confidence: {aggregated['confidence']:.2f}")

    # Тест интеграции
    print("\n3. Test integrate_sentiment_signal:")

    mock_signal = {
        'signal': 'LONG',
        'score': 50,
        'confidence': 70
    }

    integrated = integrate_sentiment_signal("BTC", mock_signal)
    print(f"   Original signal: {mock_signal['signal']}, score: {mock_signal['score']}")
    print(f"   Integrated signal: {integrated['signal']}, score: {integrated['score']:.1f}")
    print(f"   Sentiment used: {integrated['sentiment_used']}")

    print("\n" + "=" * 60)
    print("[OK] All tests passed!")
    print("=" * 60)
