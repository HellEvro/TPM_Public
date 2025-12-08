"""
Скрипт для обучения Pattern Recognition модели

Использует исторические данные для улучшения распознавания паттернов
с помощью машинного обучения.

Примечание: Большинство паттернов распознаются алгоритмически,
но ML может улучшить точность определения уверенности.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

# Добавляем корневую директорию в путь
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from bot_engine.ai.pattern_detector import PatternDetector


def extract_pattern_features(candles_df: pd.DataFrame, window_size: int = 50) -> np.ndarray:
    """
    Извлекает признаки для ML классификации паттернов
    
    Args:
        candles_df: DataFrame со свечами
        window_size: Размер окна для анализа
    
    Returns:
        Массив признаков
    """
    if len(candles_df) < window_size:
        return None
    
    # Берем последние window_size свечей
    recent = candles_df.tail(window_size)
    
    closes = recent['close'].values
    highs = recent['high'].values
    lows = recent['low'].values
    volumes = recent['volume'].values
    
    # Вычисляем статистические признаки
    features = []
    
    # 1. Волатильность
    volatility = np.std(closes) / np.mean(closes)
    features.append(volatility)
    
    # 2. Тренд (slope цен)
    x = np.arange(len(closes))
    slope = np.polyfit(x, closes, 1)[0]
    features.append(slope)
    
    # 3. Диапазон цен (нормализованный)
    price_range = (max(closes) - min(closes)) / np.mean(closes)
    features.append(price_range)
    
    # 4. Соотношение max/min
    max_min_ratio = max(closes) / min(closes)
    features.append(max_min_ratio)
    
    # 5. Количество экстремумов
    from scipy.signal import argrelextrema
    maxima = len(argrelextrema(closes, np.greater, order=3)[0])
    minima = len(argrelextrema(closes, np.less, order=3)[0])
    features.append(maxima)
    features.append(minima)
    
    # 6. Средний объем
    avg_volume = np.mean(volumes)
    features.append(avg_volume)
    
    # 7. Изменение объема
    volume_change = (volumes[-1] - volumes[0]) / volumes[0] if volumes[0] > 0 else 0
    features.append(volume_change)
    
    # 8. Позиция текущей цены относительно диапазона
    current_position = (closes[-1] - min(closes)) / (max(closes) - min(closes)) if max(closes) != min(closes) else 0.5
    features.append(current_position)
    
    # 9. Сходимость/расходимость (сравнение линий тренда максимумов и минимумов)
    if len(highs) > 10:
        highs_slope = np.polyfit(x, highs, 1)[0]
        lows_slope = np.polyfit(x, lows, 1)[0]
        convergence = highs_slope - lows_slope
        features.append(convergence)
    else:
        features.append(0)
    
    return np.array(features)


def label_patterns(candles_df: pd.DataFrame, future_candles: int = 5) -> str:
    """
    Автоматически определяет паттерн на основе будущего движения цены
    
    Args:
        candles_df: DataFrame со свечами
        future_candles: Сколько свечей вперед смотреть
    
    Returns:
        Метка паттерна: 'BULLISH', 'BEARISH', 'NEUTRAL'
    """
    if len(candles_df) < future_candles + 10:
        return 'NEUTRAL'
    
    current_price = candles_df.iloc[-future_candles-1]['close']
    future_price = candles_df.iloc[-1]['close']
    
    change_percent = ((future_price - current_price) / current_price) * 100
    
    # Классифицируем по изменению цены
    if change_percent > 3:  # +3% и более
        return 'BULLISH'
    elif change_percent < -3:  # -3% и менее
        return 'BEARISH'
    else:
        return 'NEUTRAL'


def prepare_training_data(
    csv_file: str,
    window_size: int = 50,
    future_candles: int = 5
) -> List[Tuple[np.ndarray, str]]:
    """
    Подготавливает обучающие данные из CSV файла
    
    Args:
        csv_file: Путь к CSV файлу
        window_size: Размер окна для признаков
        future_candles: Горизонт для определения метки
    
    Returns:
        Список (признаки, метка)
    """
    try:
        df = pd.read_csv(csv_file)
        
        if len(df) < window_size + future_candles + 20:
            return []
        
        training_samples = []
        
        # Создаем скользящее окно
        for i in range(window_size, len(df) - future_candles):
            window_df = df.iloc[i-window_size:i+future_candles]
            
            # Извлекаем признаки из окна
            features = extract_pattern_features(
                window_df.iloc[:-future_candles],
                window_size=window_size
            )
            
            if features is None:
                continue
            
            # Определяем метку на основе будущего движения
            label = label_patterns(window_df, future_candles=future_candles)
            
            training_samples.append((features, label))
        
        return training_samples
        
    except Exception as e:
        print(f"  [ERROR] File processing error: {e}")
        return []


def load_all_historical_data(
    data_dir: str = "data/ai/historical",
    max_coins: int = 0,
    window_size: int = 50
) -> List[Tuple[np.ndarray, str]]:
    """
    Загружает все исторические данные для обучения
    
    Args:
        data_dir: Директория с CSV файлами
        max_coins: Максимальное количество монет (0 = все)
        window_size: Размер окна для признаков
    
    Returns:
        Список всех обучающих образцов
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"[ERROR] Directory not found: {data_dir}")
        return []
    
    # Получаем список всех CSV файлов
    csv_files = sorted(data_path.glob("*.csv"))
    
    if max_coins > 0:
        csv_files = csv_files[:max_coins]
    
    print(f"\nFound CSV files: {len(csv_files)}")
    print(f"Loading historical data...")
    print("=" * 60)
    
    all_training_data = []
    successful = 0
    failed = 0
    
    for i, csv_file in enumerate(csv_files, 1):
        print(f"[{i}/{len(csv_files)}] {csv_file.stem}")
        
        training_data = prepare_training_data(
            str(csv_file),
            window_size=window_size
        )
        
        if training_data:
            all_training_data.extend(training_data)
            successful += 1
            print(f"  [OK] Prepared {len(training_data)} samples")
        else:
            failed += 1
            print(f"  [SKIP] Not enough data")
    
    print("\n" + "=" * 60)
    print(f"[OK] Successfully processed: {successful} coins")
    print(f"[FAILED] Errors: {failed} coins")
    print(f"[TOTAL] Training samples: {len(all_training_data)}")
    
    return all_training_data


def main():
    """Основная функция обучения"""
    parser = argparse.ArgumentParser(description='Обучение Pattern Detector')
    parser.add_argument('--coins', type=int, default=0, help='Количество монет для обучения (0 = все)')
    parser.add_argument('--window', type=int, default=50, help='Размер окна для анализа')
    args = parser.parse_args()
    
    print("=" * 60)
    print("PATTERN RECOGNITION TRAINING")
    print("=" * 60)
    
    print(f"\nParameters:")
    print(f"  Coins for training: {'all' if args.coins == 0 else args.coins}")
    print(f"  Window size: {args.window} candles")
    
    # Загружаем исторические данные
    training_data = load_all_historical_data(
        max_coins=args.coins,
        window_size=args.window
    )
    
    if not training_data:
        print("\n[ERROR] No training data!")
        print("Run first: python scripts/ai/collect_historical_data.py")
        return 1
    
    # Создаем и обучаем модель
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    detector = PatternDetector()
    
    print("\nTraining pattern recognition model...")
    print(f"Training samples: {len(training_data)}")
    
    result = detector.train(
        training_data=training_data,
        validation_split=0.2
    )
    
    # Выводим результаты
    print("\n" + "=" * 60)
    if result.get('success'):
        print("[SUCCESS] TRAINING COMPLETED!")
        print("=" * 60)
        print(f"Train accuracy: {result['train_accuracy']:.3f}")
        print(f"Validation accuracy: {result['val_accuracy']:.3f}")
        print(f"Training samples: {result['training_samples']}")
        print(f"\n[SAVED] Model saved to: data/ai/models/pattern_detector.pkl")
        return 0
    else:
        print("[ERROR] TRAINING FAILED!")
        print("=" * 60)
        print(f"Error: {result.get('error', 'Unknown')}")
        return 1


if __name__ == "__main__":
    exit(main())

