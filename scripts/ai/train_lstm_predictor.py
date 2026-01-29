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

from bot_engine.ai.lstm_predictor import LSTMPredictor, PYTORCH_AVAILABLE
from utils.rsi_calculator import calculate_rsi
from utils.memory_utils import force_collect_full


def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Вычисляет EMA"""
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    multiplier = 2 / (period + 1)
    
    for i in range(1, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
    
    return ema


def prepare_training_data_from_candles(
    candles: list,
    symbol: str,
    sequence_length: int = 60,
    prediction_horizon: int = 1
) -> list:
    """
    Подготавливает данные для обучения из списка свечей
    
    Args:
        candles: Список свечей [{'time': int, 'open': float, ...}, ...]
        symbol: Символ монеты
        sequence_length: Длина последовательности (60 свечей)
        prediction_horizon: Горизонт предсказания (1 свеча = 6 часов)
    
    Returns:
        Список (X, y) пар для обучения
    """
    if not candles or len(candles) < sequence_length + prediction_horizon + 20:
        return []
    
    try:
        # Преобразуем в DataFrame
        df = pd.DataFrame(candles)
        df = df.sort_values('time')
        
        # Переименовываем колонки для совместимости
        if 'time' in df.columns:
            df = df.rename(columns={'time': 'timestamp'})
        
        # Проверяем наличие необходимых колонок
        required_cols = ['close', 'volume', 'high', 'low']
        if not all(col in df.columns for col in required_cols):
            return []
        
        # Вычисляем дополнительные признаки
        # RSI
        df['rsi'] = calculate_rsi(df['close'].values, period=14)
        
        # EMA
        df['ema_fast'] = calculate_ema(df['close'].values, period=12)
        df['ema_slow'] = calculate_ema(df['close'].values, period=26)
        
        # Удаляем NaN значения
        df = df.dropna()
        
        if len(df) < sequence_length + prediction_horizon + 20:
            return []
        
        # Подготавливаем признаки
        features = ['close', 'volume', 'high', 'low', 'rsi', 'ema_fast', 'ema_slow']
        data = df[features].values
        
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
        
        return training_samples
        
    except Exception as e:
        print(f"  [ERROR] Processing error for {symbol}: {e}")
        return []


def prepare_training_data(
    csv_file: str,
    sequence_length: int = 60,
    prediction_horizon: int = 1
) -> list:
    """
    Подготавливает данные для обучения из CSV файла (legacy метод)
    
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
    
    ПРИОРИТЕТ: БД (ai_data.db) → CSV файлы (legacy)
    
    Args:
        data_dir: Директория с CSV файлами (используется только если БД пуста)
        max_coins: Максимальное количество монет (0 = все)
        sequence_length: Длина последовательности
    
    Returns:
        Список всех обучающих образцов
    """
    all_training_data = []
    successful = 0
    failed = 0
    
    # ПРИОРИТЕТ 1: Загружаем из БД
    try:
        from bot_engine.ai.ai_database import get_ai_database
        ai_db = get_ai_database()
        
        if ai_db:
            print("\n[INFO] Загрузка данных из БД (ai_data.db)...")
            print("=" * 60)
            
            # Получаем свечи из БД с ограничениями для экономии памяти
            # По умолчанию: максимум 20 символов, 1000 свечей на символ
            default_max_symbols = 20 if max_coins == 0 else max_coins
            candles_dict = ai_db.get_all_candles_dict(
                timeframe='6h',
                max_symbols=default_max_symbols,
                max_candles_per_symbol=1000
            )
            
            if candles_dict:
                symbols = sorted(candles_dict.keys())
                
                if max_coins > 0:
                    symbols = symbols[:max_coins]
                
                print(f"Found {len(symbols)} symbols in database (limited to {default_max_symbols} for memory efficiency)")
                
                # Ограничиваем общий размер обучающих данных (максимум 50000 образцов)
                MAX_TOTAL_SAMPLES = 50000
                
                for i, symbol in enumerate(symbols, 1):
                    candles = candles_dict[symbol]
                    print(f"\n[{i}/{len(symbols)}] {symbol} ({len(candles)} candles)")
                    
                    training_data = prepare_training_data_from_candles(
                        candles,
                        symbol,
                        sequence_length=sequence_length
                    )
                    
                    if training_data:
                        # Проверяем лимит общего количества образцов
                        if len(all_training_data) + len(training_data) > MAX_TOTAL_SAMPLES:
                            remaining = MAX_TOTAL_SAMPLES - len(all_training_data)
                            if remaining > 0:
                                training_data = training_data[:remaining]
                                all_training_data.extend(training_data)
                                print(f"  [OK] Prepared {len(training_data)} samples (limited to {MAX_TOTAL_SAMPLES} total)")
                                print(f"  [WARNING] Достигнут лимит образцов ({MAX_TOTAL_SAMPLES}), остальные символы пропущены")
                                break
                            else:
                                print(f"  [SKIP] Достигнут лимит образцов ({MAX_TOTAL_SAMPLES})")
                                break
                        else:
                            all_training_data.extend(training_data)
                        
                        successful += 1
                        print(f"  [OK] Prepared {len(training_data)} samples (total: {len(all_training_data)})")
                    else:
                        failed += 1
                        print(f"  [SKIP] Not enough data")
                    
                    # Очищаем память после обработки каждого символа
                    del candles, training_data
                    force_collect_full()
                
                print("\n" + "=" * 60)
                print(f"[OK] Successfully processed: {successful} coins")
                print(f"[FAILED] Errors: {failed} coins")
                print(f"[TOTAL] Training samples: {len(all_training_data)}")
                
                if all_training_data:
                    return all_training_data
                else:
                    print("[WARNING] БД пуста или недостаточно данных, пробуем CSV...")
            else:
                print("[WARNING] БД пуста, пробуем CSV файлы...")
        else:
            print("[WARNING] AI Database не доступна, пробуем CSV файлы...")
    except Exception as e:
        print(f"[WARNING] Ошибка загрузки из БД: {e}")
        print("[INFO] Пробуем загрузить из CSV файлов...")
    
    # ПРИОРИТЕТ 2: Загружаем из CSV (legacy)
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"[ERROR] Directory not found: {data_dir}")
        return []
    
    # Получаем список всех CSV файлов
    csv_files = sorted(data_path.glob("*.csv"))
    
    if max_coins > 0:
        csv_files = csv_files[:max_coins]
    
    print(f"\nFound CSV files: {len(csv_files)}")
    print(f"Loading historical data from CSV...")
    print("=" * 60)
    
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
    default_batch = 32
    try:
        from bot_engine.ai.ai_launcher_config import AILauncherConfig
        default_batch = getattr(AILauncherConfig, 'TRAINING_BATCH_SIZE', 32)
    except Exception:
        pass
    parser = argparse.ArgumentParser(description='Обучение LSTM предиктора')
    parser.add_argument('--coins', type=int, default=20, help='Количество монет для обучения (по умолчанию 20 для экономии памяти)')
    parser.add_argument('--epochs', type=int, default=50, help='Количество эпох обучения')
    parser.add_argument('--batch-size', type=int, default=default_batch, help='Размер батча (при лимите ОЗУ берётся из AILauncherConfig)')
    parser.add_argument('--sequence-length', type=int, default=60, help='Длина последовательности')
    args = parser.parse_args()
    
    print("=" * 60)
    print("LSTM PREDICTOR TRAINING")
    print("=" * 60)
    
    # Проверяем PyTorch
    if not PYTORCH_AVAILABLE:
        print("[ERROR] PyTorch not installed!")
        print("Install: pip install torch")
        return 1
    
    # Проверяем и настраиваем GPU NVIDIA (PyTorch)
    gpu_device = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"\n[GPU] Найдено GPU устройств: {gpu_count}")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"  GPU {i}: {gpu_name}")
            
            gpu_device = torch.device('cuda:0')
            print(f"[GPU] GPU NVIDIA будет использоваться для обучения: {torch.cuda.get_device_name(0)}")
        else:
            print("\n[GPU] GPU устройства не найдены, используется CPU")
    except Exception as e:
        print(f"\n[GPU] Ошибка проверки GPU: {e}")
        print("[GPU] Продолжаем с CPU...")
    
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
        model_path="data/ai/models/lstm_predictor_new.keras",  # ✅ Временный путь в Keras 3 формате
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
        final_model = "data/ai/models/lstm_predictor.keras"  # ✅ Keras 3 формат
        final_scaler = "data/ai/models/lstm_scaler.pkl"
        
        if os.path.exists("data/ai/models/lstm_predictor_new.keras"):
            shutil.move("data/ai/models/lstm_predictor_new.keras", final_model)
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
        print(f"\n[SAVED] Model saved to: data/ai/models/lstm_predictor.keras")
        return 0
    else:
        print("[ERROR] TRAINING FAILED!")
        print("=" * 60)
        print(f"Error: {result.get('error', 'Unknown')}")
        return 1


if __name__ == "__main__":
    exit(main())

