#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Диагностический скрипт для проверки работы AI обучения

Проверяет:
1. Инициализацию AITrainer
2. Загрузку сделок из БД
3. Количество сделок для обучения
4. Доступность моделей
"""

import sys
import os
from pathlib import Path

# Настройка кодировки для Windows консоли
if os.name == 'nt':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 80)
print("ДИАГНОСТИКА AI ОБУЧЕНИЯ")
print("=" * 80)

# 1. Проверка инициализации AITrainer
print("\n[1] Проверка инициализации AITrainer...")
try:
    from bot_engine.ai.ai_trainer import AITrainer
    trainer = AITrainer()
    print("   [OK] AITrainer успешно инициализирован")
except Exception as e:
    print(f"   [ERROR] Ошибка инициализации AITrainer: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2. Проверка загрузки сделок
print("\n[2] Проверка загрузки сделок из БД...")
try:
    trades_count = trainer.get_trades_count()
    print(f"   [OK] Количество сделок для обучения: {trades_count}")
    
    if trades_count == 0:
        print("   [WARNING] ВНИМАНИЕ: Нет сделок для обучения!")
        print("   [INFO] Проверьте:")
        print("      - Есть ли сделки в bots_data.db -> bot_trades_history")
        print("      - Есть ли сделки в ai_data.db -> bot_trades, exchange_trades")
    elif trades_count < 10:
        print(f"   [WARNING] Мало сделок для обучения (нужно >= 10, есть {trades_count})")
    else:
        print(f"   [OK] Достаточно сделок для обучения ({trades_count} >= 10)")
except Exception as e:
    print(f"   [ERROR] Ошибка загрузки сделок: {e}")
    import traceback
    traceback.print_exc()

# 3. Проверка загрузки истории напрямую
print("\n[3] Проверка _load_history_data()...")
try:
    trades = trainer._load_history_data()
    print(f"   [OK] Загружено {len(trades)} сделок через _load_history_data()")
    
    if trades:
        sample = trades[0]
        print(f"   [INFO] Пример сделки:")
        print(f"      - Symbol: {sample.get('symbol')}")
        print(f"      - PnL: {sample.get('pnl')}")
        print(f"      - RSI: {sample.get('rsi')}")
        print(f"      - Trend: {sample.get('trend')}")
        print(f"      - Source: {sample.get('decision_source', 'UNKNOWN')}")
except Exception as e:
    print(f"   [ERROR] Ошибка _load_history_data(): {e}")
    import traceback
    traceback.print_exc()

# 4. Проверка БД напрямую
print("\n[4] Проверка БД напрямую...")
try:
    from bot_engine.ai.ai_database import get_ai_database
    ai_db = get_ai_database()
    
    # Проверяем сделки из разных источников
    db_trades = ai_db.get_trades_for_training(
        include_simulated=False,
        include_real=True,
        include_exchange=True,
        min_trades=0,
        limit=None
    )
    print(f"   [OK] get_trades_for_training(): {len(db_trades)} сделок")
    
    # Проверяем bots_data.db
    from bot_engine.bots_database import get_bots_database
    bots_db = get_bots_database()
    bots_trades = bots_db.get_bot_trades_history(
        status='CLOSED',
        limit=None
    )
    print(f"   [OK] bots_data.db -> bot_trades_history: {len(bots_trades)} сделок")
    
except Exception as e:
    print(f"   [ERROR] Ошибка проверки БД: {e}")
    import traceback
    traceback.print_exc()

# 5. Проверка моделей
print("\n[5] Проверка моделей...")
try:
    models_dir = Path('data/ai/models')
    if not models_dir.exists():
        print(f"   [WARNING] Директория моделей не найдена: {models_dir}")
    else:
        signal_model = models_dir / 'signal_predictor.pkl'
        profit_model = models_dir / 'profit_predictor.pkl'
        
        if signal_model.exists():
            print(f"   [OK] Модель сигналов найдена: {signal_model}")
        else:
            print(f"   [WARNING] Модель сигналов не найдена: {signal_model}")
        
        if profit_model.exists():
            print(f"   [OK] Модель прибыли найдена: {profit_model}")
        else:
            print(f"   [WARNING] Модель прибыли не найдена: {profit_model}")
except Exception as e:
    print(f"   [ERROR] Ошибка проверки моделей: {e}")

# 6. Проверка лицензии (если доступна)
print("\n[6] Проверка лицензии...")
try:
    # Пытаемся импортировать через sys.path
    import importlib.util
    license_path = PROJECT_ROOT / 'license_generator' / 'source' / '@source' / 'ai_launcher_source.py'
    if license_path.exists():
        spec = importlib.util.spec_from_file_location("ai_launcher_source", str(license_path))
        ai_launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ai_launcher)
        
        license_status = ai_launcher.ensure_license_available()
        if license_status.get('valid'):
            print("   [OK] Лицензия валидна")
            features = license_status.get('info', {}).get('features', {})
            if features.get('ai_training'):
                print("   [OK] Функция 'ai_training' включена в лицензию")
            else:
                print("   [WARNING] Функция 'ai_training' НЕ включена в лицензию")
        else:
            print("   [WARNING] Лицензия невалидна или отсутствует")
    else:
        print(f"   [WARNING] Файл лицензии не найден: {license_path}")
except Exception as e:
    print(f"   [WARNING] Не удалось проверить лицензию: {e}")

# 7. Проверка конфигурации AI системы
print("\n[7] Проверка конфигурации AI системы...")
try:
    from bot_engine.config_loader import AIConfig
    print(f"   [INFO] AI_ENABLED: {AIConfig.AI_ENABLED}")
    print(f"   [INFO] AI_AUTO_TRAIN_ENABLED: {AIConfig.AI_AUTO_TRAIN_ENABLED}")
except Exception as e:
    print(f"   [WARNING] Не удалось проверить конфигурацию: {e}")

# 8. Проверка запущенных процессов AI
print("\n[8] Проверка запущенных процессов AI...")
try:
    import psutil
    ai_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline and any('ai.py' in str(arg) for arg in cmdline):
                mode = 'unknown'
                for i, arg in enumerate(cmdline):
                    if arg == '--mode' and i + 1 < len(cmdline):
                        mode = cmdline[i + 1]
                        break
                ai_processes.append({
                    'pid': proc.info['pid'],
                    'mode': mode,
                    'cmdline': ' '.join(cmdline)
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if ai_processes:
        print(f"   [INFO] Найдено {len(ai_processes)} процессов AI:")
        for proc in ai_processes:
            print(f"      - PID {proc['pid']}: режим '{proc['mode']}'")
            if proc['mode'] == 'data-service':
                print("      [WARNING] Режим 'data-service' - обучение ОТКЛЮЧЕНО!")
            elif proc['mode'] == 'train':
                print("      [OK] Режим 'train' - обучение ВКЛЮЧЕНО")
            elif proc['mode'] == 'all':
                print("      [OK] Режим 'all' - должны быть запущены все процессы (data-service, train, scheduler)")
    else:
        print("   [WARNING] Процессы AI не найдены!")
        print("   [INFO] Запустите: python ai.py --mode all")
except ImportError:
    print("   [INFO] psutil не установлен, пропускаем проверку процессов")
except Exception as e:
    print(f"   [WARNING] Ошибка проверки процессов: {e}")

# 9. Проверка статуса data-service в БД
print("\n[9] Проверка статуса data-service в БД...")
try:
    from bot_engine.ai.ai_database import get_ai_database
    ai_db = get_ai_database()
    
    # Проверяем статус data-service
    status_result = ai_db.get_data_service_status('data_service')
    if status_result:
        # status_result содержит поле 'status' с вложенным словарем
        status = status_result.get('status', {})
        if not isinstance(status, dict):
            status = status_result
            
        print(f"   [INFO] Последний сбор данных: {status.get('last_collection', status.get('timestamp', 'N/A'))}")
        print(f"   [INFO] Сделок в БД: {status.get('trades_count', status.get('total_trades', 0))}")
        print(f"   [INFO] Свечей в БД: {status.get('candles_count', status.get('total_candles', 0))}")
        print(f"   [INFO] Готовность: {'Да' if status.get('ready', status.get('data_ready', False)) else 'Нет'}")
        print(f"   [INFO] История загружена: {'Да' if status.get('history_loaded', False) else 'Нет'}")
        print(f"   [INFO] Обновлено: {status_result.get('updated_at', 'N/A')}")
    else:
        print("   [WARNING] Статус data-service не найден")
        print("   [INFO] Это нормально, если data-service еще не запускался")
except Exception as e:
    print(f"   [WARNING] Ошибка проверки статуса data-service: {e}")

# 10. Проверка свечей в БД
print("\n[10] Проверка свечей в БД...")
try:
    from bot_engine.ai.ai_database import get_ai_database
    ai_db = get_ai_database()
    
    candles_dict = ai_db.get_all_candles_dict()
    total_candles = sum(len(candles) for candles in candles_dict.values())
    symbols_count = len(candles_dict)
    
    print(f"   [INFO] Монет со свечами: {symbols_count}")
    print(f"   [INFO] Всего свечей: {total_candles:,}")
    
    if total_candles == 0:
        print("   [WARNING] Нет свечей в БД!")
        print("   [ACTION] Запустите сбор данных: python ai.py --mode data-service")
    elif total_candles < 1000:
        print(f"   [WARNING] Мало свечей ({total_candles} < 1000)")
    else:
        print(f"   [OK] Достаточно свечей для обучения")
except Exception as e:
    print(f"   [WARNING] Ошибка проверки свечей: {e}")

# 11. Проверка статистики по сделкам
print("\n[11] Детальная статистика по сделкам...")
try:
    if 'trades' in locals() and trades:
        # Статистика по источникам
        sources = {}
        for trade in trades:
            source = trade.get('decision_source', 'UNKNOWN')
            sources[source] = sources.get(source, 0) + 1
        
        print(f"   [INFO] Сделки по источникам:")
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            print(f"      - {source}: {count} сделок")
        
        # Статистика по успешности
        successful = sum(1 for t in trades if t.get('is_successful') or (t.get('pnl', 0) > 0))
        losing = len(trades) - successful
        win_rate = (successful / len(trades) * 100) if trades else 0
        
        print(f"   [INFO] Успешных сделок: {successful} ({win_rate:.1f}%)")
        print(f"   [INFO] Убыточных сделок: {losing}")
        
        # Проверка наличия RSI и Trend данных
        trades_with_rsi = sum(1 for t in trades if t.get('rsi') is not None or t.get('entry_rsi') is not None)
        trades_with_trend = sum(1 for t in trades if t.get('trend') is not None or t.get('entry_trend') is not None)
        
        print(f"   [INFO] Сделок с RSI данными: {trades_with_rsi} ({trades_with_rsi/len(trades)*100:.1f}%)")
        print(f"   [INFO] Сделок с Trend данными: {trades_with_trend} ({trades_with_trend/len(trades)*100:.1f}%)")
        
        if trades_with_rsi < len(trades) * 0.5:
            print("   [WARNING] Меньше 50% сделок имеют RSI данные!")
            print("   [ACTION] AI нужны RSI данные для обучения. Проверьте загрузку индикаторов.")
        
        if trades_with_trend < len(trades) * 0.5:
            print("   [WARNING] Меньше 50% сделок имеют Trend данные!")
            print("   [ACTION] AI нужны Trend данные для обучения. Проверьте загрузку индикаторов.")
        
        # Статистика по символам
        symbols = {}
        for trade in trades:
            symbol = trade.get('symbol', 'UNKNOWN')
            symbols[symbol] = symbols.get(symbol, 0) + 1
        
        print(f"   [INFO] Уникальных символов: {len(symbols)}")
        if len(symbols) > 0:
            top_symbols = sorted(symbols.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"   [INFO] Топ-5 символов по количеству сделок:")
            for symbol, count in top_symbols:
                print(f"      - {symbol}: {count} сделок")
    else:
        print("   [WARNING] Нет сделок для статистики")
except Exception as e:
    print(f"   [WARNING] Ошибка статистики: {e}")

# 12. Проверка последних обучений
print("\n[12] Проверка последних обучений...")
try:
    from bot_engine.ai.ai_data_storage import AIDataStorage
    storage = AIDataStorage()
    last_trainings = storage.get_training_history(limit=3)
    
    if last_trainings:
        print(f"   [INFO] Найдено {len(last_trainings)} последних обучений:")
        for idx, training in enumerate(last_trainings, 1):
            print(f"   [INFO] Обучение #{idx}:")
            print(f"      - Дата: {training.get('timestamp', 'N/A')}")
            print(f"      - Монет обработано: {training.get('total_trained_coins', 0)}")
            print(f"      - Моделей сохранено: {training.get('total_models_saved', 0)}")
            print(f"      - Win Rate: {training.get('overall_win_rate', 0):.1f}%")
            print(f"      - PnL: {training.get('overall_pnl', 0):.2f} USDT")
    else:
        print("   [WARNING] Обучений не найдено!")
        print("   [ACTION] Запустите обучение: python ai.py --mode train")
except Exception as e:
    print(f"   [WARNING] Ошибка проверки обучений: {e}")

# 13. Проверка параметров обучения
print("\n[13] Проверка параметров обучения...")
try:
    from bot_engine.config_loader import AIConfig
    print(f"   [INFO] AI_ENABLED: {AIConfig.AI_ENABLED}")
    print(f"   [INFO] AI_AUTO_TRAIN_ENABLED: {AIConfig.AI_AUTO_TRAIN_ENABLED}")
    print(f"   [INFO] AI_RETRAIN_INTERVAL: {getattr(AIConfig, 'AI_RETRAIN_INTERVAL', 'N/A')} сек")
    print(f"   [INFO] AI_MIN_TRADES_FOR_TRAINING: {getattr(AIConfig, 'AI_MIN_TRADES_FOR_TRAINING', 'N/A')}")
except Exception as e:
    print(f"   [WARNING] Ошибка проверки параметров: {e}")

# 14. Проверка симуляций
print("\n[14] Проверка симулированных сделок...")
try:
    from bot_engine.ai.ai_database import get_ai_database
    ai_db = get_ai_database()
    
    simulated_trades = ai_db.get_trades_for_training(
        include_simulated=True,
        include_real=False,
        include_exchange=False,
        min_trades=0,
        limit=None
    )
    print(f"   [INFO] Симулированных сделок: {len(simulated_trades)}")
    
    if len(simulated_trades) > 0:
        print("   [OK] Есть симуляции для обучения")
    else:
        print("   [WARNING] Нет симуляций!")
        print("   [INFO] Симуляции создаются автоматически при обучении на исторических данных")
except Exception as e:
    print(f"   [WARNING] Ошибка проверки симуляций: {e}")

# 15. Рекомендации
print("\n[15] РЕКОМЕНДАЦИИ:")
if 'trades_count' in locals():
    if trades_count == 0:
        print("   [ACTION] Нет сделок для обучения!")
        print("   [ACTION] Запустите: python scripts/rebuild_bot_history_from_exchange.py")
    elif trades_count < 10:
        print(f"   [ACTION] Мало сделок ({trades_count} < 10)")
        print("   [ACTION] Накопите больше сделок или используйте симуляции")
    else:
        print("   [OK] Достаточно сделок для обучения")
        print("   [ACTION] Убедитесь, что AI система запущена в режиме 'all' или 'train':")
        print("           python ai.py --mode all")
        print("   [ACTION] Или только обучение:")
        print("           python ai.py --mode train")

print("\n" + "=" * 80)
print("[OK] ДИАГНОСТИКА ЗАВЕРШЕНА")
print("=" * 80)
print("\n[INFO] Для запуска обучения:")
print("   python ai.py --mode all      # Все сервисы (рекомендуется)")
print("   python ai.py --mode train    # Только обучение")
print("\n[INFO] Для проверки статуса:")
print("   python scripts/check_databases.py")
print("   python scripts/verify_ai_ready.py")

