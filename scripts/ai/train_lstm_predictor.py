"""
Скрипт для обучения LSTM модели на исторических данных

Использует собранные исторические данные для обучения LSTM
предсказывать движение цены на 6 часов вперед.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Добавляем корневую директорию в путь
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from bot_engine.ai.lstm_predictor import LSTMPredictor, TENSORFLOW_AVAILABLE
from utils.rsi_calculator import calculate_rsi


def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Вычисляет EMA"""
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    multiplier = 2 / (period + 1)
    
    for i in range(1, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
    
    return ema


def prepare_training_data(
    csv_file: str,
    sequence_length: int = 60,
    prediction_horizon: int = 1
) -> list:
    """
    Подготавливает данные для обучения из CSV файла
    
    Args:
        csv_file: Путь к CSV файлу с историческими данными
        sequence_length: Длина последовательности (60 свечей)
        prediction_horizon: Горизонт предсказания (1 свеча = 6 часов)
    
    Returns:
        Список (X, y) пар для обучения
    """
    print(f"  Загрузка: {csv_file}")
    
    try:
        # Загружаем данные
        df = pd.read_csv(csv_file)
        
        if len(df) < sequence_length + prediction_horizon + 20:
            print(f"  [SKIP] Not enough data ({len(df)} candles)")
            return []
        
        # Проверяем наличие необходимых колонок
        required_cols = ['close', 'volume', 'high', 'low']
        if not all(col in df.columns for col in required_cols):
            print(f"  [ERROR] Missing required columns")
            return []
        
        # Вычисляем дополнительные признаки
        print("  Calculating features...")
        
        # RSI
        df['rsi'] = calculate_rsi(df['close'].values, period=14)
        
        # EMA
        df['ema_fast'] = calculate_ema(df['close'].values, period=12)
        df['ema_slow'] = calculate_ema(df['close'].values, period=26)
        
        # Удаляем NaN значения
        df = df.dropna()
        
        if len(df) < sequence_length + prediction_horizon + 20:
            print(f"  [SKIP] Not enough data after processing ({len(df)} candles)")
            return []
        
        # Подготавливаем признаки
        features = ['close', 'volume', 'high', 'low', 'rsi', 'ema_fast', 'ema_slow']
        data = df[features].values
        
        # Нормализуем данные (MinMaxScaler будет применен в LSTMPredictor)
        # Здесь мы просто подготавливаем последовательности
        
        training_samples = []
        
        # Создаем скользящее окно
        for i in range(len(data) - sequence_length - prediction_horizon):
            # Входная последовательность (60 свечей)
            X = data[i:i + sequence_length]
            
            # Целевые значения (следующая свеча через prediction_horizon)
            current_close = data[i + sequence_length - 1, 0]  # close
            future_close = data[i + sequence_length + prediction_horizon - 1, 0]
            
            # Вычисляем целевые значения:
            # 1. Направление: 1 (вверх) или -1 (вниз)
            direction = 1.0 if future_close > current_close else -1.0
            
            # 2. Изменение в процентах
            change_percent = ((future_close - current_close) / current_close) * 100
            
            # 3. Уверенность (на основе величины изменения)
            confidence = min(abs(change_percent) / 10, 1.0)  # 0-1
            
            y = np.array([direction, change_percent, confidence])
            
            training_samples.append((X, y))
        
        print(f"  [OK] Prepared samples: {len(training_samples)}")
        return training_samples
        
    except Exception as e:
        print(f"  [ERROR] File processing error: {e}")
        return []


def load_all_historical_data(
    data_dir: str = "data/ai/historical",
    max_coins: int = 0,
    sequence_length: int = 60
) -> list:
    """
    Загружает все исторические данные для обучения
    
    Args:
        data_dir: Директория с CSV файлами
        max_coins: Максимальное количество монет (0 = все)
        sequence_length: Длина последовательности
    
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
        print(f"\n[{i}/{len(csv_files)}] {csv_file.name}")
        
        training_data = prepare_training_data(
            str(csv_file),
            sequence_length=sequence_length
        )
        
        if training_data:
            all_training_data.extend(training_data)
            successful += 1
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"[OK] Successfully processed: {successful} coins")
    print(f"[FAILED] Errors: {failed} coins")
    print(f"[TOTAL] Training samples: {len(all_training_data)}")
    
    return all_training_data


def main():
    """Основная функция обучения"""
    parser = argparse.ArgumentParser(description='Обучение LSTM предиктора')
    parser.add_argument('--coins', type=int, default=0, help='Количество монет для обучения (0 = все)')
    parser.add_argument('--epochs', type=int, default=50, help='Количество эпох обучения')
    parser.add_argument('--batch-size', type=int, default=32, help='Размер батча')
    parser.add_argument('--sequence-length', type=int, default=60, help='Длина последовательности')
    args = parser.parse_args()
    
    print("=" * 60)
    print("LSTM PREDICTOR TRAINING")
    print("=" * 60)
    
    # Проверяем TensorFlow
    if not TENSORFLOW_AVAILABLE:
        print("[ERROR] TensorFlow not installed!")
        print("Install: pip install tensorflow")
        return 1
    
    print(f"\nParameters:")
    print(f"  Coins for training: {'all' if args.coins == 0 else args.coins}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.sequence_length} candles")
    
    # Загружаем исторические данные
    training_data = load_all_historical_data(
        max_coins=args.coins,
        sequence_length=args.sequence_length
    )
    
    if not training_data:
        print("\n[ERROR] No training data!")
        print("Run first: python scripts/ai/collect_historical_data.py")
        return 1
    
    # Создаем и обучаем модель
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    # Создаем новый предиктор БЕЗ загрузки существующей модели
    predictor = LSTMPredictor(
        model_path="data/ai/models/lstm_predictor_new.h5",  # Временный путь
        scaler_path="data/ai/models/lstm_scaler_new.pkl"
    )
    
    # Обучаем модель (она сама нормализует данные внутри)
    print("\nTraining neural network...")
    print(f"Training samples: {len(training_data)}")
    
    result = predictor.train(
        training_data=training_data,  # Передаем ненормализованные данные
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Переименовываем модель в финальную версию
    if result.get('success'):
        import shutil
        final_model = "data/ai/models/lstm_predictor.h5"
        final_scaler = "data/ai/models/lstm_scaler.pkl"
        
        if os.path.exists("data/ai/models/lstm_predictor_new.h5"):
            shutil.move("data/ai/models/lstm_predictor_new.h5", final_model)
        if os.path.exists("data/ai/models/lstm_scaler_new.pkl"):
            shutil.move("data/ai/models/lstm_scaler_new.pkl", final_scaler)
    
    # Выводим результаты
    print("\n" + "=" * 60)
    if result.get('success'):
        print("[SUCCESS] TRAINING COMPLETED!")
        print("=" * 60)
        print(f"Final loss (train): {result['final_loss']:.6f}")
        print(f"Final loss (val): {result['final_val_loss']:.6f}")
        print(f"Epochs trained: {result['epochs_trained']}")
        print(f"Training samples: {result['training_samples']}")
        print(f"\n[SAVED] Model saved to: data/ai/models/lstm_predictor.h5")
        return 0
    else:
        print("[ERROR] TRAINING FAILED!")
        print("=" * 60)
        print(f"Error: {result.get('error', 'Unknown')}")
        return 1


if __name__ == "__main__":
    exit(main())

