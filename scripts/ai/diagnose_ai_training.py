#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Диагностика проблем с обучением AI

Проверяет:
1. Количество сделок для обучения
2. Состояние моделей (обучены ли)
3. Качество моделей (точность, метрики)
4. Доступность данных (свечи, сделки)
5. Проблемы с обучением
"""

import sys
import os
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import logging
from datetime import datetime, timedelta
from bot_engine.ai.ai_database import get_ai_database
from bot_engine.ai import get_ai_system

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AI.Diagnose')


def diagnose_ai_training():
    """Диагностика проблем с обучением AI"""
    print("=" * 80)
    print("🔍 ДИАГНОСТИКА ПРОБЛЕМ С ОБУЧЕНИЕМ AI")
    print("=" * 80)
    print()
    
    issues = []
    warnings = []
    info = []
    
    # 1. Проверка подключения к БД
    print("📊 Проверка подключения к БД...")
    try:
        ai_db = get_ai_database()
        if ai_db:
            print("   ✅ БД подключена")
            info.append("БД подключена")
        else:
            print("   ❌ БД не подключена")
            issues.append("БД не подключена")
    except Exception as e:
        print(f"   ❌ Ошибка подключения к БД: {e}")
        issues.append(f"Ошибка подключения к БД: {e}")
    
    # 2. Проверка количества сделок
    print("\n📈 Проверка количества сделок...")
    try:
        if ai_db:
            bot_trades = ai_db.get_bot_trades(status='CLOSED', limit=None)
            exchange_trades_count = ai_db.count_exchange_trades()
            
            total_trades = len(bot_trades) + exchange_trades_count
            
            print(f"   📊 Сделки ботов: {len(bot_trades)}")
            print(f"   📊 Сделки биржи: {exchange_trades_count}")
            print(f"   📊 Всего сделок: {total_trades}")
            
            if total_trades < 10:
                print(f"   ⚠️ Недостаточно сделок для обучения (нужно минимум 10, есть {total_trades})")
                warnings.append(f"Недостаточно сделок для обучения: {total_trades} < 10")
            else:
                print(f"   ✅ Достаточно сделок для обучения")
                info.append(f"Достаточно сделок: {total_trades}")
    except Exception as e:
        print(f"   ❌ Ошибка проверки сделок: {e}")
        issues.append(f"Ошибка проверки сделок: {e}")
    
    # 3. Проверка свечей
    print("\n🕯️ Проверка свечей...")
    try:
        if ai_db:
            candles_count = ai_db.count_candles()
            symbols_count = ai_db.count_symbols_with_candles()
            
            print(f"   📊 Всего свечей: {candles_count}")
            print(f"   📊 Монет со свечами: {symbols_count}")
            
            if candles_count < 1000:
                print(f"   ⚠️ Мало свечей для обучения (есть {candles_count})")
                warnings.append(f"Мало свечей: {candles_count}")
            else:
                print(f"   ✅ Достаточно свечей")
                info.append(f"Достаточно свечей: {candles_count}")
            
            if symbols_count < 10:
                print(f"   ⚠️ Мало монет со свечами (есть {symbols_count})")
                warnings.append(f"Мало монет со свечами: {symbols_count}")
    except Exception as e:
        print(f"   ❌ Ошибка проверки свечей: {e}")
        issues.append(f"Ошибка проверки свечей: {e}")
    
    # 4. Проверка моделей
    print("\n🤖 Проверка моделей...")
    try:
        ai_system = get_ai_system()
        if not ai_system:
            print("   ❌ AI система не инициализирована")
            issues.append("AI система не инициализирована")
        else:
            trainer = ai_system.trainer
            if not trainer:
                print("   ❌ Trainer не инициализирован")
                issues.append("Trainer не инициализирован")
            else:
                # Проверка signal_predictor
                if trainer.signal_predictor:
                    print("   ✅ signal_predictor обучен")
                    info.append("signal_predictor обучен")
                else:
                    print("   ❌ signal_predictor не обучен")
                    issues.append("signal_predictor не обучен")
                
                # Проверка profit_predictor
                if trainer.profit_predictor:
                    print("   ✅ profit_predictor обучен")
                    info.append("profit_predictor обучен")
                else:
                    print("   ❌ profit_predictor не обучен")
                    issues.append("profit_predictor не обучен")
                
                # Проверка ParameterQualityPredictor
                if trainer.param_quality_predictor:
                    if trainer.param_quality_predictor.is_trained:
                        print("   ✅ ParameterQualityPredictor обучен")
                        info.append("ParameterQualityPredictor обучен")
                    else:
                        print("   ⚠️ ParameterQualityPredictor не обучен")
                        warnings.append("ParameterQualityPredictor не обучен")
                else:
                    print("   ⚠️ ParameterQualityPredictor не инициализирован")
                    warnings.append("ParameterQualityPredictor не инициализирован")
    except Exception as e:
        print(f"   ❌ Ошибка проверки моделей: {e}")
        issues.append(f"Ошибка проверки моделей: {e}")
    
    # 5. Проверка необходимости переобучения
    print("\n🔄 Проверка необходимости переобучения...")
    try:
        if ai_system and ai_system.trainer:
            trainer = ai_system.trainer
            
            # Проверка переобучения на реальных сделках
            should_retrain = trainer._should_retrain_real_trades_models()
            print(f"   📊 Переобучение на реальных сделках: {'✅ Нужно' if should_retrain['retrain'] else '❌ Не нужно'}")
            print(f"   📝 Причина: {should_retrain['reason']}")
            print(f"   📊 Сделок: {should_retrain['trades_count']}")
            
            if should_retrain['retrain']:
                warnings.append(f"Требуется переобучение на реальных сделках: {should_retrain['reason']}")
            
            # Проверка переобучения ParameterQualityPredictor
            if trainer.param_quality_predictor:
                should_retrain_param = trainer._should_retrain_parameter_quality_model()
                print(f"   📊 Переобучение ParameterQualityPredictor: {'✅ Нужно' if should_retrain_param['retrain'] else '❌ Не нужно'}")
                print(f"   📝 Причина: {should_retrain_param['reason']}")
                
                if should_retrain_param['retrain']:
                    warnings.append(f"Требуется переобучение ParameterQualityPredictor: {should_retrain_param['reason']}")
    except Exception as e:
        print(f"   ❌ Ошибка проверки переобучения: {e}")
        issues.append(f"Ошибка проверки переобучения: {e}")
    
    # Итоговый отчет
    print("\n" + "=" * 80)
    print("📋 ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 80)
    
    if issues:
        print(f"\n❌ КРИТИЧЕСКИЕ ПРОБЛЕМЫ ({len(issues)}):")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    
    if warnings:
        print(f"\n⚠️ ПРЕДУПРЕЖДЕНИЯ ({len(warnings)}):")
        for i, warning in enumerate(warnings, 1):
            print(f"   {i}. {warning}")
    
    if info:
        print(f"\n✅ ИНФОРМАЦИЯ ({len(info)}):")
        for i, item in enumerate(info, 1):
            print(f"   {i}. {item}")
    
    if not issues and not warnings:
        print("\n✅ Все проверки пройдены успешно!")
    
    print("\n" + "=" * 80)
    
    # Рекомендации
    if issues or warnings:
        print("\n💡 РЕКОМЕНДАЦИИ:")
        if issues:
            print("   1. Исправьте критические проблемы перед обучением")
        if warnings:
            print("   2. Устраните предупреждения для улучшения качества обучения")
        print("   3. Запустите обучение: python scripts/ai/train_on_real_trades.py")
        print("   4. Проверьте логи: logs/ai.log")
    
    return len(issues) == 0


if __name__ == '__main__':
    try:
        success = diagnose_ai_training()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
