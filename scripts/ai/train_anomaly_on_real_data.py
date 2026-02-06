"""
Обучение Anomaly Detector на реальных исторических данных

Использует собранные данные из БД ai_data.db (таблица candles_history) для обучения модели.
"""

import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь (устойчиво к запуску из любого cwd)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import os
os.environ['AI_DEV_MODE'] = '1'

# ВАЖНО: до любых импортов sklearn — подавляет UserWarning delayed/Parallel и фиксирует joblib
import utils.sklearn_parallel_config  # noqa: F401

from bot_engine.ai.anomaly_detector import AnomalyDetector
from bot_engine.config_loader import AIConfig
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _build_training_windows(candles, window_size: int, step: int):
    """Преобразует список свечей в набор скользящих окон для обучения."""
    if not candles or len(candles) < window_size:
        return []
    windows = []
    for i in range(0, len(candles) - window_size + 1, step):
        windows.append(candles[i:i + window_size])
    return windows


def load_historical_data(timeframe: str = '6h', window_size: int = 50, step: int = 25, max_symbols: int = 50, max_candles_per_symbol: int = 3000):
    """
    Загружает исторические данные из БД (ai_data.db) и создаёт скользящие окна.
    
    Args:
        timeframe: Таймфрейм свечей
        window_size: Размер окна (количество свечей)
        step: Шаг окна (сколько свечей пропускать)
        max_symbols: Максимум символов для выборки (защита от переполнения памяти)
        max_candles_per_symbol: Максимум свечей на символ
    
    Returns:
        Список списков свечей (каждый список = window_size свечей)
    """
    try:
        from bot_engine.ai.ai_database import get_ai_database
        ai_db = get_ai_database()
        if not ai_db:
            logger.error("AI Database недоступна (get_ai_database вернул None)")
            return []
    except Exception as e:
        logger.error(f"Ошибка инициализации AI Database: {e}")
        return []

    logger.info(f"Загружаем свечи из БД: timeframe={timeframe}, max_symbols={max_symbols}, max_candles_per_symbol={max_candles_per_symbol}")
    logger.info(f"Параметры окна: размер={window_size}, шаг={step}")

    candles_by_symbol = ai_db.get_all_candles_dict(
        timeframe=timeframe,
        max_symbols=max_symbols,
        max_candles_per_symbol=max_candles_per_symbol
    ) or {}

    if not candles_by_symbol:
        logger.error(f"В БД нет данных свечей для timeframe={timeframe}")
        logger.info("Сначала запустите: python scripts/ai/collect_historical_data.py --limit 20")
        return []

    training_data = []
    required_fields = ['close', 'high', 'low', 'volume']

    for symbol, candles in candles_by_symbol.items():
        try:
            if not candles:
                continue
            if not all(field in candles[0] for field in required_fields):
                logger.warning(f"  ⚠️ {symbol}: отсутствуют необходимые поля")
                continue

            windows = _build_training_windows(candles, window_size=window_size, step=step)
            if windows:
                training_data.extend(windows)
            logger.info(f"  ✅ {symbol}: {len(candles)} свечей → {len(windows)} окон")
        except Exception as e:
            logger.error(f"  ❌ Ошибка подготовки данных для {symbol}: {e}")

    logger.info(f"Всего создано {len(training_data)} тренировочных примеров")
    return training_data


def train_on_real_data():
    """Обучает Anomaly Detector на реальных данных"""
    
    print("=" * 60)
    print("TRAINING ANOMALY DETECTOR ON REAL DATA")
    print("=" * 60)
    print()
    
    # Загружаем данные
    print("Step 1/4: Loading historical data...")
    print("-" * 60)
    training_data = load_historical_data(timeframe='6h')
    
    if not training_data:
        print("[ERROR] No training data found!")
        print()
        print("Please run first:")
        print("  python scripts/ai/collect_historical_data.py --limit 20")
        return
    
    # Подсчитываем общее количество свечей
    total_candles = sum(len(candles) for candles in training_data)
    
    print()
    print(f"[OK] Loaded {len(training_data)} training examples")
    print(f"[OK] Total candles: {total_candles:,}")
    print()
    
    # Создаем детектор
    print("Step 2/4: Creating detector...")
    print("-" * 60)
    detector = AnomalyDetector()
    print(f"[OK] Detector created: {detector.get_status()}")
    print()
    
    # Обучаем
    print("Step 3/4: Training model...")
    print("-" * 60)
    print(f"Training on {len(training_data)} examples...")
    print("This may take 1-5 minutes depending on data volume...")
    print()
    
    import time
    train_start = time.time()
    
    success = detector.train(training_data)
    
    train_time = time.time() - train_start
    
    if not success:
        print("[ERROR] Training failed!")
        return
    
    print()
    print(f"[OK] Training completed in {train_time:.1f} seconds")
    print()
    
    # Сохраняем модель
    print("Step 4/4: Saving model...")
    print("-" * 60)
    
    model_path = AIConfig.AI_ANOMALY_MODEL_PATH
    scaler_path = AIConfig.AI_ANOMALY_SCALER_PATH
    
    detector.save_model(model_path, scaler_path)
    
    # Проверяем что модель сохранилась
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model_size = os.path.getsize(model_path) / 1024  # KB
        scaler_size = os.path.getsize(scaler_path) / 1024  # KB
        
        print(f"[OK] Model saved: {model_path} ({model_size:.1f} KB)")
        print(f"[OK] Scaler saved: {scaler_path} ({scaler_size:.1f} KB)")
    else:
        print("[ERROR] Failed to save files!")
        return
    
    print()
    
    # Тестирование на реальных данных
    print("Testing model...")
    print("-" * 60)
    
    # Берем несколько случайных примеров для теста
    import random
    test_samples = random.sample(training_data, min(10, len(training_data)))
    
    anomalies_detected = 0
    high_severity = 0
    
    for i, candles in enumerate(test_samples, 1):
        result = detector.detect(candles)
        
        severity = result['severity']
        is_anomaly = result['is_anomaly']
        
        # Прогресс бар
        progress = "#" * int(severity * 20)
        
        status = "[ANOMALY]" if is_anomaly else "[NORMAL]"
        print(f"  Test {i:2d}: {status:12s} severity={severity:6.1%} {progress}")
        
        if is_anomaly:
            anomalies_detected += 1
            if severity > 0.7:
                high_severity += 1
    
    print()
    print(f"[OK] Anomalies detected: {anomalies_detected}/{len(test_samples)}")
    print(f"[OK] High severity (>70%): {high_severity}/{len(test_samples)}")
    print()
    
    # Итоговые инструкции
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print()
    print("Модель обучена и готова к использованию!")
    print()
    print("Чтобы активировать:")
    print()
    print("1. Отредактируйте configs/bot_config.py:")
    print()
    print("   class AIConfig:")
    print("       AI_ENABLED = True")
    print("       AI_ANOMALY_DETECTION_ENABLED = True")
    print()
    print("2. Включите режим разработки:")
    print()
    print("   set AI_DEV_MODE=1  # Windows")
    print("   export AI_DEV_MODE=1  # Linux/Mac")
    print()
    print("3. Перезапустите бота:")
    print()
    print("   python bots.py")
    print()
    print("4. Мониторьте логи на наличие:")
    print()
    print("   [AI] [OK] Anomaly Detector loaded")
    print("   [EXIT_SCAM] ... [BLOCKED] (AI): Anomaly detected ...")
    print()


if __name__ == '__main__':
    train_on_real_data()

