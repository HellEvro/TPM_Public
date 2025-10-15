#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для синхронизации данных об оптимальных EMA
Загружает данные из optimal_ema.py и обновляет их в bots.py
"""

import os
import sys
import json
import logging

# Добавляем путь к корню проекта для импорта модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def sync_optimal_ema_data():
    """Синхронизирует данные об оптимальных EMA между скриптами"""
    try:
        # Пути к файлам
        optimal_ema_file = 'data/optimal_ema.json'
        
        # Проверяем существование файла
        if not os.path.exists(optimal_ema_file):
            logger.warning(f"Файл {optimal_ema_file} не найден")
            return False
        
        # Загружаем данные из файла
        with open(optimal_ema_file, 'r', encoding='utf-8') as f:
            optimal_ema_data = json.load(f)
        
        if not optimal_ema_data:
            logger.warning("Файл optimal_ema.json пустой")
            return False
        
        logger.info(f"Загружено {len(optimal_ema_data)} записей об оптимальных EMA")
        
        # Импортируем функцию обновления из bots.py
        try:
            from bots import update_optimal_ema_data
            success = update_optimal_ema_data(optimal_ema_data)
            
            if success:
                logger.info("✅ Данные об оптимальных EMA успешно синхронизированы")
                return True
            else:
                logger.error("❌ Ошибка синхронизации данных")
                return False
                
        except ImportError as e:
            logger.error(f"❌ Не удалось импортировать функцию обновления: {e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Ошибка синхронизации: {e}")
        return False

def main():
    """Основная функция"""
    logger.info("🔄 Начинаем синхронизацию данных об оптимальных EMA...")
    
    success = sync_optimal_ema_data()
    
    if success:
        logger.info("✅ Синхронизация завершена успешно")
        sys.exit(0)
    else:
        logger.error("❌ Синхронизация завершена с ошибками")
        sys.exit(1)

if __name__ == '__main__':
    main()
