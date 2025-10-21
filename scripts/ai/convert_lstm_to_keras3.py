"""
Скрипт для конвертации LSTM модели из формата .h5 в .keras (Keras 3)
"""
import os
import sys

# Устанавливаем UTF-8 для консоли Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from tensorflow import keras
    print("[OK] TensorFlow imported")
except ImportError:
    print("[ERROR] TensorFlow not installed")
    sys.exit(1)


def convert_model():
    """Конвертирует модель из .h5 в .keras"""
    old_model_path = "data/ai/models/lstm_predictor.h5"
    new_model_path = "data/ai/models/lstm_predictor.keras"
    
    print("\n" + "=" * 60)
    print("LSTM MODEL CONVERSION")
    print("=" * 60)
    print(f"From: {old_model_path}")
    print(f"To:   {new_model_path}")
    print()
    
    # Проверяем, существует ли старая модель
    if not os.path.exists(old_model_path):
        print(f"[ERROR] File not found: {old_model_path}")
        return False
    
    try:
        # Загружаем старую модель
        print(f"[1/3] Loading model from {old_model_path}...")
        
        # Пробуем загрузить с compile=False, чтобы избежать ошибок с метриками
        model = keras.models.load_model(old_model_path, compile=False)
        print("[OK] Model loaded (without compilation)")
        
        # Перекомпилируем модель с правильными метриками
        print("[2/3] Recompiling model...")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.00025),
            loss='mse',
            metrics=['mae']
        )
        print("[OK] Model recompiled")
        
        # Сохраняем в новом формате
        print(f"[3/3] Saving model to {new_model_path}...")
        model.save(new_model_path)
        print("[OK] Model saved")
        
        # Проверяем размер файлов
        old_size = os.path.getsize(old_model_path) / (1024 * 1024)  # MB
        new_size = os.path.getsize(new_model_path) / (1024 * 1024)  # MB
        
        print("\n" + "=" * 60)
        print("[SUCCESS] CONVERSION COMPLETED!")
        print("=" * 60)
        print(f"Old model size: {old_size:.2f} MB")
        print(f"New model size: {new_size:.2f} MB")
        print(f"\nNew model: {new_model_path}")
        
        # Тестируем загрузку новой модели
        print("\n[TEST] Testing new model loading...")
        test_model = keras.models.load_model(new_model_path)
        print("[OK] New model loads correctly")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = convert_model()
    sys.exit(0 if success else 1)

